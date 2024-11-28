from transformers_openai.env import args
from transformers_openai.base_model import (
    Segment,
    TranscriptionVerboseJsonResponse,
    TranscriptionJsonResponse,
)
from transformers_openai.function import (
    sample,
    pad_hidden_encoder,
    efficient_attention_mask,
    format_timestamp,
)
from transformers_openai.function_hf import (
    load_hf_tokenizer,
    load_hf_processor,
    load_hf_model,
    decode,
)
from transformers_openai.cache import (
    StaticLengthEncoderDecoderCache,
    DynamicLengthEncoderDecoderCache
)
from fastapi import Request
from sse_starlette import ServerSentEvent
from torchaudio.io import StreamReader
from datetime import datetime
import re
import numpy as np
import torch
import json
import time
import asyncio
import logging
import traceback
import gc

if args.hqq:
    from hqq.models.hf.base import AutoHQQHFModel
    from hqq.core.quantize import *
    from hqq.utils.patching import prepare_for_inference
    import hqq.models.base as hqq_base
    hqq_base._QUANT_LAYERS = [torch.nn.Linear, HQQLinear]
    quant_config = BaseQuantizeConfig(
        nbits=4,
        group_size=64,
        quant_scale=False,
        quant_zero=False,
        axis=1
    )
    HQQLinear.set_backend(HQQBackend.PYTORCH)

buffer_size = 4096
sample_rate = 16000
segment_length = sample_rate * 1
maxlen = 30
replaces = ['<|startoftranscript|>', '<|endoftext|>', '<|transcribe|>']
pattern = r'<\|\-?\d+\.?\d*\|>'
pattern_pair = r'<\|(\d+\.\d+)\|>(.*?)<\|(\d+\.\d+)\|>'

model = None
processor = None
tokenizer = None
no_speech_token = None
global_cache = None

torch_dtype = getattr(torch, args.torch_dtype)
device = args.device

prefill_queue = asyncio.Queue()
step_queue = asyncio.Queue()

def load_model():
    global model, processor, tokenizer, no_speech_token, global_cache
    model = load_hf_model()
    processor = load_hf_processor()
    tokenizer = load_hf_tokenizer()
    if args.static_cache:
        logging.info('use static cache')
        global_cache = StaticLengthEncoderDecoderCache(
            batch_size = args.continuous_batching_microsleep, 
            encoder_max_length = args.static_cache_encoder_max_length,
            decoder_max_length = args.static_cache_decoder_max_length,
            encoder_head_size = model.config.encoder_attention_heads,
            decoder_head_size = model.config.decoder_attention_heads,
            encoder_dim_size = model.config.d_model,
            decoder_dim_size = model.config.d_model,
            encoder_hidden_layers = model.config.encoder_layers,
            decoder_hidden_layers = model.config.decoder_layers,
            dtype = torch_dtype,
            device = device,
            whisper_mode=True,
        )
    else:
        logging.info('use dynamic cache')
        global_cache = DynamicLengthEncoderDecoderCache(whisper_mode = True)
    try:
        no_speech_token = tokenizer.convert_tokens_to_ids(['<|nospeech|>'])[0]
    except BaseException:
        pass

    if args.hqq:
        AutoHQQHFModel.quantize_model(
            model.model.encoder,
            quant_config=quant_config,
            compute_dtype=compute_dtype,
            device=device
        )
        AutoHQQHFModel.quantize_model(
            model.model.decoder,
            quant_config=quant_config,
            compute_dtype=compute_dtype,
            device=device
        )
        AutoHQQHFModel.set_auto_linear_tags(model.model.encoder)
        prepare_for_inference(model.model.encoder)
        AutoHQQHFModel.set_auto_linear_tags(model.model.decoder)
        prepare_for_inference(model.model.decoder, backend='torchao_int4')
    
    if args.torch_compile:
        model.model.encoder.forward = torch.compile(
            model.model.encoder.forward,
            mode='reduce-overhead',
            fullgraph=True
        )


async def prefill():
    need_sleep = True
    while True:
        if need_sleep:
            await asyncio.sleep(args.continuous_batching_microsleep)
        try:
            need_sleep = True
            batch = []
            while not prefill_queue.empty():
                try:
                    request = await asyncio.wait_for(prefill_queue.get(), timeout=1e-4)
                    batch.append(request)
                    if len(batch) >= args.continuous_batching_batch_size:
                        need_sleep = False
                        break
                except asyncio.TimeoutError:
                    break

            if not len(batch):
                continue

            logging.debug(f'{str(datetime.now())} prefill batch size of {len(batch)}')

            futures = [batch[i][0] for i in range(len(batch))]
            langs = [batch[i][1] for i in range(len(batch))]
            inputs = [batch[i][2] for i in range(len(batch))]
            uuids = [batch[i][5] for i in range(len(batch))]

            with torch.no_grad():
                inputs = processor(
                    inputs,
                    sampling_rate=sample_rate,
                    device = device,
                )
                inputs = inputs['input_features'].type(model.dtype)
                out_encoder = model.model.encoder(inputs)
                out_encoder = out_encoder[0]
                langs_none_map, langs_none = {}, []
                index = 0
                for i in range(len(langs)):
                    if langs[i] is None:
                        langs_none_map[index] = i
                        langs_none.append(out_encoder[i:i + 1])
                        index += 1

                if len(langs_none):
                    labels = tokenizer(
                        '<|startoftranscript|>',
                        add_special_tokens=False,
                        return_tensors='pt',
                    ).to('cuda')['input_ids']

                    labels = labels.repeat(len(langs_none), 1)
                    langs_none = torch.concat(langs_none, dim=0)

                    out_decoder = model.model.decoder(
                        labels,
                        encoder_hidden_states=langs_none,
                        return_dict=False,
                    )
                    proj = model.proj_out(out_decoder[0][:, -1:]).argmax(-1)

                    langs_none = tokenizer.batch_decode(proj)
                    langs_none = [l[2:-2] for l in langs_none]
                    for k, v in langs_none_map.items():
                        langs[v] = langs_none[k]

                prompt_ids = []
                for lang in langs:
                    lang_token = tokenizer.encode(f'<|{lang}|>', add_special_tokens=False)[0]
                    prompt_ids.append([50258, lang_token, 50360, 50365])

                inputs = torch.tensor(prompt_ids).to('cuda')

                out = model.model.decoder(
                    inputs,
                    encoder_hidden_states=out_encoder,
                    past_key_values=None,
                    position_ids=None,
                    use_cache=True,
                    return_dict=False,
                )
                out_logits = model.proj_out(out[0][:, -1:])
                out_caches = out[1]

                cache_exists = len(global_cache.key_cache) > 0

                for k in range(len(out_caches)):
                    key_cache = {}
                    value_cache = {}
                    cross_key_cache = {}
                    cross_value_cache = {}

                    for i in range(len(batch)):
                        key_cache[uuids[i]] = out_caches[k][0][i: i + 1]
                        value_cache[uuids[i]] = out_caches[k][1][i: i + 1]
                        cross_key_cache[uuids[i]] = out_caches[k][2][i: i + 1]
                        cross_value_cache[uuids[i]] = out_caches[k][3][i: i + 1]

                    if cache_exists:
                        global_cache.key_cache[k].update(key_cache)
                        global_cache.value_cache[k].update(value_cache)
                        global_cache.cross_key_cache[k].update(cross_key_cache)
                        global_cache.cross_value_cache[k].update(cross_value_cache)
                    else:
                        global_cache.key_cache.append(key_cache)
                        global_cache.value_cache.append(value_cache)
                        global_cache.cross_key_cache.append(cross_key_cache)
                        global_cache.cross_value_cache.append(cross_value_cache)

                for i in range(len(futures)):
                    futures[i].set_result((out_logits[i: i + 1], out_encoder[i:i + 1]))

        except Exception as e:
            logging.warning(f"Error in prefill: {e}")
            futures = [batch[i][0] for i in range(len(batch))]
            for i in range(len(futures)):
                if not futures[i].done():
                    futures[i].set_exception(e)


async def step():
    need_sleep = True
    while True:
        if need_sleep:
            await asyncio.sleep(args.continuous_batching_microsleep)
        try:
            need_sleep = True
            batch = []
            while not step_queue.empty():
                try:
                    request = await asyncio.wait_for(step_queue.get(), timeout=1e-6)
                    batch.append(request)
                    if len(batch) >= args.continuous_batching_batch_size:
                        need_sleep = False
                        break
                except asyncio.TimeoutError:
                    break

            if not len(batch):
                continue

            logging.debug(f'{str(datetime.now())} step batch size of {len(batch)}')

            futures = [batch[i][0] for i in range(len(batch))]
            langs = [batch[i][1] for i in range(len(batch))]
            inputs = [batch[i][2] for i in range(len(batch))]
            out_encoders = [batch[i][3] for i in range(len(batch))]
            lengths = [batch[i][4] for i in range(len(batch))]
            uuids = [batch[i][5] for i in range(len(batch))]

            global_cache.current_uuid = uuids
            max_len = max(lengths)

            with torch.no_grad():
                inputs = torch.concat(inputs, dim=0)
                out_encoder = pad_hidden_encoder(out_encoders)
                attention_mask = efficient_attention_mask(
                    batch_size=len(lengths),
                    max_len=max_len,
                    lengths=lengths,
                    device=device,
                    dtype=torch_dtype,
                )
                position_ids = torch.tensor([[l + 3 for l in lengths]]).T.cuda()
                out = model.model.decoder(
                    inputs,
                    encoder_hidden_states=out_encoder,
                    past_key_values=global_cache,
                    position_ids=position_ids,
                    use_cache=True,
                    return_dict=False,
                )
                out_logits = model.proj_out(out[0][:, -1:])

                for i in range(len(futures)):
                    futures[i].set_result((out_logits[i: i + 1],))

                out = list(out)

        except Exception as e:
            print(traceback.format_exc())
            logging.warning(f"Error in step: {e}")
            futures = [batch[i][0] for i in range(len(batch))]
            for i in range(len(futures)):
                if not futures[i].done():
                    futures[i].set_exception(e)


async def generate(
    wav_data,
    language,
    last_timestamp,
    last_i,
    response_format,
    repetition_penalty,
    temperature,
    top_p,
    top_k,
    request,
):
    no_speech_prob = 0.0
    mask_penalty = torch.ones((1, model.config.vocab_size)).cuda()
    inputs = wav_data

    out_encoder = None
    texts = f'<|{language}|><|{last_timestamp}|>'

    if response_format != 'srt':
        text = texts
        if response_format == 'json':
            text = json.dumps({'token': texts})
        yield text

    if isinstance(request, dict):
        uuid = request['uuid']
    else:
        uuid = request.scope['request']['uuid']
    uuid = f'{uuid}_{last_i}'

    # minus 4 because ['<|startoftranscript|>', lang token, '<|transcribe|>', '<|0.0|>'] tokens
    try:
        for k in range(model.config.max_length - 4):
            if k == 0:
                q = prefill_queue
            else:
                q = step_queue

            future = asyncio.Future()
            await q.put((future, language, inputs, out_encoder, k, uuid))
            out = await future

            logits = out[0]

            if out_encoder is None:
                out_encoder = out[1]

            idx_next, probs = sample(
                logits,
                mask_penalty,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            mask_penalty[0, idx_next[0]] = repetition_penalty
            token = tokenizer.decode(idx_next, decode_with_timestamps=True)

            if k == 0 and 'time_first_token' not in request.scope['request']:
                request.scope['request']['time_first_token'] = time.time()

            ids = idx_next[0].tolist()
            if ids == model.config.eos_token_id:
                break

            inputs = idx_next.unsqueeze(0)

            for r in replaces:
                token = token.replace(r, '')

            matches = re.findall(pattern, token)
            for match in matches:
                timestamp = float(match.split('|')[1])
                timestamp += last_timestamp
                timestamp = f'<|{timestamp}|>'
                token = token.replace(match, timestamp)
            if len(token):
                texts += token
                matches = re.findall(pattern_pair, texts)
                if response_format == 'srt':
                    if len(matches):
                        match = matches[0]
                        if len(match[1]) > 2:
                            start = float(match[0]) + last_timestamp
                            end = float(match[-1]) + last_timestamp
                            text_between = match[1].strip()
                            ids = f"{last_i + 1}\n"
                            r = [
                                ids,
                                f"{format_timestamp(start, always_include_hours=True, decimal_marker=',')} --> ",
                                f"{format_timestamp(end, always_include_hours=True, decimal_marker=',')}\n",
                                f"{text_between.replace('-->', '->')}\n"]

                            combined = ''.join(r) + '\n'
                            last_i += 1
                            yield combined

                        texts = token.split('|>')[-2] + '|>'
                else:
                    if response_format == 'json':
                        token = json.dumps({'token': token})

                    yield token
    
    except asyncio.CancelledError as e:
        logging.warning(f"model step cancelled {uuid}")
        yield ServerSentEvent(**{"data": str(e)})
    
    except Exception as e:
        logging.error(f"model step exception {e} {uuid}")
        yield ServerSentEvent(**{"data": str(e)})

    finally:
        logging.debug(f'purging {uuid} KV cache')
        for i in range(len(global_cache.key_cache)):
            key_cache = global_cache.key_cache[i].pop(uuid, None)
            value_cache = global_cache.value_cache[i].pop(uuid, None)
            cross_key_cache = global_cache.cross_key_cache[i].pop(uuid, None)
            cross_value_cache = global_cache.cross_value_cache[i].pop(uuid, None)
        torch.cuda.empty_cache()
        gc.collect()


async def audio(file, language, response_format, repetition_penalty, temperature, top_p, top_k, request):
    wav_data = np.array([], dtype=np.float32)
    last_i = 0
    last_timestamp = 0.0
    try:
        streamer = StreamReader(
            src=file,
            format=None,
            option=None,
            buffer_size=buffer_size
        )
        streamer.add_basic_audio_stream(
            frames_per_chunk=segment_length,
            sample_rate=sample_rate
        )
        stream_iterator = streamer.stream()
        for chunk in stream_iterator:
            frame = chunk[0][:, 0].numpy()
            wav_data = np.concatenate([wav_data, frame])
            audio_len = len(wav_data) / sample_rate
            if audio_len >= maxlen:
                async for t in generate(
                    wav_data=wav_data,
                    language=language,
                    last_timestamp=last_timestamp,
                    last_i=last_i,
                    response_format=response_format,
                    repetition_penalty=repetition_penalty,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    request=request,
                ):
                    yield t
                    await asyncio.sleep(0)
                    last_i += 1

                last_timestamp += audio_len
                wav_data = np.array([], dtype=np.float32)

        if len(wav_data):
            async for t in generate(
                wav_data=wav_data,
                language=language,
                last_timestamp=last_timestamp,
                last_i=last_i,
                response_format=response_format,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                request=request,
            ):
                yield t
                await asyncio.sleep(0)
                last_i += 1

        audio_len = len(wav_data) / sample_rate
        last_timestamp += audio_len

        request.scope['request']['time_max_tokens'] = time.time()
        request.scope['request']['total_tokens'] = last_i
        request.scope['request']['total_seconds'] = last_timestamp

    except asyncio.CancelledError as e:
        logging.warning(f"model step cancelled {request.scope['request']['uuid']}")
        yield ServerSentEvent(**{"data": str(e)})


async def audio_completions(
    file,
    language,
    response_format,
    timestamp_granularities,
    stream,
    repetition_penalty,
    temperature,
    top_p,
    top_k,
    request: Request = None,
):
    if model is None:
        load_model()

    func = audio(
        file=file,
        language=language,
        response_format='json' if not stream else response_format,
        repetition_penalty=repetition_penalty,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        request=request,
    )

    if stream:
        return func
    else:
        tokens = []
        async for data in func:
            if isinstance(data, ServerSentEvent):
                continue
            data = json.loads(data)
            tokens.append(data['token'])

        tokens = ''.join(tokens)
        lang = tokens.split('|')[1]
        matches = re.findall(pattern_pair, tokens)
        segments = []
        all_texts = []
        for no, (start, substring, end) in enumerate(matches):
            start_timestamp = float(start)
            end_timestamp = float(end)
            segment = Segment(
                id=no,
                seek=0,
                start=start_timestamp,
                end=end_timestamp,
                text=substring.strip(),
                tokens=tokenizer.encode(substring.strip(), add_special_tokens=False),
                temperature=temperature,
                avg_logprob=0.0,
                compression_ratio=1.0,
                no_speech_prob=0.0,
            )
            segments.append(segment)
            all_texts.append(substring)

        all_texts = ''.join(all_texts).strip()
        if response_format == 'verbose_json':
            return TranscriptionVerboseJsonResponse(
                task='transcribe',
                language=lang,
                duration=segments[-1].end,
                text=all_texts,
                segments=segments
            )
        elif response_format == 'json':
            return TranscriptionJsonResponse(
                text=all_texts
            )
        else:
            return all_texts
