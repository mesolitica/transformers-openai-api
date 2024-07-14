from transformers_openai.env import args
from transformers_openai.base_model import (
    Segment,
    TranscriptionVerboseJsonResponse,
    TranscriptionJsonResponse,
)
from transformers_openai.function import (
    sample,
    decode,
    load_hf_processor,
    load_hf_model,
    pad_hidden_encoder,
    efficient_attention_mask,
    format_timestamp,
    cleanup_cache,
)
from transformers_openai.cache import DynamicLengthEncoderDecoderCache
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

buffer_size = 4096
sample_rate = 16000
segment_length = sample_rate * 1
maxlen = 30
replaces = ['<|startoftranscript|>', '<|endoftext|>', '<|transcribe|>']
pattern = r'<\|\-?\d+\.?\d*\|>'
pattern_pair = r'<\|(\d+\.\d+)\|>(.*?)<\|(\d+\.\d+)\|>'


model = None
processor = None
no_speech_token = None

prefill_queue = asyncio.Queue()
step_queue = asyncio.Queue()


def load_model():
    global model, processor, no_speech_token
    model = load_hf_model()
    processor = load_hf_processor()
    try:
        no_speech_token = processor.tokenizer.convert_tokens_to_ids(['<|nospeech|>'])[0]
    except BaseException:
        pass


async def prefill():
    while True:
        await asyncio.sleep(args.continous_batching_microsleep)
        try:
            batch = []
            while not prefill_queue.empty():
                try:
                    request = await asyncio.wait_for(prefill_queue.get(), timeout=1e-4)
                    batch.append(request)
                    if len(batch) >= args.continous_batching_batch_size:
                        break
                except asyncio.TimeoutError:
                    break

            if not len(batch):
                continue

            logging.debug(f'{str(datetime.now())} prefill batch size of {len(batch)}')
            futures = [batch[i][0] for i in range(len(batch))]
            langs = [batch[i][1] for i in range(len(batch))]
            inputs = [batch[i][2] for i in range(len(batch))]

            inputs = torch.concat(inputs)
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
                labels = processor.tokenizer(
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

                langs_none = processor.tokenizer.batch_decode(proj)
                langs_none = [l[2:-2] for l in langs_none]
                for k, v in langs_none_map.items():
                    langs[v] = langs_none[k]

                del labels, langs_none, out_decoder[0], proj

            prompt_ids = []
            for lang in langs:
                lang_token = processor.tokenizer.encode(f'<|{lang}|>', add_special_tokens=False)[0]
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

            caches = []
            for i in range(len(batch)):
                cache = []
                for k in range(len(out_caches)):
                    cache_ = [
                        out_caches[k][0][i:i + 1],
                        out_caches[k][1][i:i + 1],
                        out_caches[k][2][i:i + 1],
                        out_caches[k][3][i:i + 1],
                    ]
                    cache.append(cache_)
                caches.append(cache)

            for i in range(len(futures)):
                futures[i].set_result((out_logits[i: i + 1], caches[i], out_encoder[i:i + 1]))

            del out_logits, out_encoder

            for k in range(len(out_caches)):
                temp = list(out_caches[k])
                for j in range(len(out_caches[k])):
                    del temp[0]

            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error in prefill: {e}")
            futures = [batch[i][0] for i in range(len(batch))]
            for i in range(len(futures)):
                if not futures[i].done():
                    futures[i].set_exception(e)


async def step():
    while True:
        await asyncio.sleep(args.continous_batching_microsleep)
        try:
            batch = []
            while not step_queue.empty():
                try:
                    request = await asyncio.wait_for(step_queue.get(), timeout=1e-4)
                    batch.append(request)
                    if len(batch) >= args.continous_batching_batch_size:
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
            caches = [batch[i][4] for i in range(len(batch))]
            lengths = [batch[i][5] for i in range(len(batch))]

            cache_dtype = caches[0][0][0].dtype
            cache_device = caches[0][0][0].device

            kv_len = [caches[i][0][0].shape[2] for i in range(len(batch))]
            max_len = max(kv_len)
            max_len_lengths = max(lengths)

            cache = DynamicLengthEncoderDecoderCache(lengths=lengths, whisper_mode=True)

            len_cache = len(caches[0])
            len_kv = len(caches[0][0])

            for n in range(len_cache):

                key_cache = []
                value_cache = []
                for i in range(len(batch)):
                    key_cache.append(caches[i][n][0])
                    value_cache.append(caches[i][n][1])

                cache.key_cache.append(key_cache)
                cache.value_cache.append(value_cache)

                key_cache = []
                value_cache = []
                for i in range(len(batch)):
                    key_cache.append(caches[i][n][2])
                    value_cache.append(caches[i][n][3])

                key_cache = torch.concat(key_cache)
                value_cache = torch.concat(value_cache)

                cache.cross_key_cache.append(key_cache)
                cache.cross_value_cache.append(value_cache)
                del key_cache, value_cache

            inputs = torch.concat(inputs, dim=0)
            out_encoder = pad_hidden_encoder(out_encoders)
            position_ids = torch.tensor([[l + 3 for l in lengths]]).T.cuda()
            out = model.model.decoder(
                inputs,
                encoder_hidden_states=out_encoder,
                past_key_values=cache,
                position_ids=position_ids,
                use_cache=True,
                return_dict=False,
            )
            out_logits = model.proj_out(out[0][:, -1:])

            caches = []
            for i in range(len(batch)):
                new_cache = []
                for k in range(len(cache)):
                    keys = cache.key_cache[k]
                    values = cache.value_cache[k]
                    v = [keys[i], values[i]]
                    keys = cache.cross_key_cache[k]
                    values = cache.cross_value_cache[k]
                    v.extend([keys[i: i + 1], values[i: i + 1]])
                    new_cache.append(v)
                caches.append(new_cache)

            for i in range(len(futures)):
                futures[i].set_result((out_logits[i: i + 1], caches[i]))

            out = list(out)
            del inputs, out_encoder, out[0], out_logits

            for k in range(len(cache)):
                temp = list(cache[k])
                for j in range(len(temp)):
                    del temp[0]

            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error in step: {e}")
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

    if 'cache' in request.scope and request.scope['cache'] is not None:
        cleanup_cache(request.scope['cache'])

        torch.cuda.empty_cache()

    cache = None
    no_speech_prob = 0.0

    mask_penalty = torch.ones((1, model.config.vocab_size)).cuda()

    with torch.no_grad():

        inputs = processor(
            [wav_data],
            return_tensors='pt',
            sampling_rate=sample_rate).to(
            model.device)
        inputs = inputs['input_features'].type(model.dtype)

        if not args.continous_batching:

            out_encoder = model.model.encoder(inputs)
            out_encoder = out_encoder[0]
            if language is None:
                labels = processor.tokenizer(
                    '<|startoftranscript|>',
                    add_special_tokens=False,
                    return_tensors='pt',
                ).to('cuda')['input_ids']
                out_decoder = model.model.decoder(labels, encoder_hidden_states=out_encoder)
                proj = model.proj_out(out_decoder.last_hidden_state[:, -1:])
                lang = processor.tokenizer.decode(proj.argmax(-1)[0])
                language = lang[2:-2]

            lang_token = processor.tokenizer.encode(f'<|{language}|>', add_special_tokens=False)[0]
            prompt_ids = [50258, lang_token, 50360, 50365]
            inputs = torch.tensor([prompt_ids]).to('cuda')

        else:
            out_encoder = None

        texts = f'<|{language}|><|{last_timestamp}|>'

        if response_format != 'srt':
            text = texts
            if response_format == 'json':
                text = json.dumps({'token': texts})

            yield text

        # minus 4 because ['<|startoftranscript|>', lang token, '<|transcribe|>', '<|0.0|>'] tokens
        for k in range(model.config.max_length - 4):
            if args.continous_batching:
                if k == 0:
                    q = prefill_queue
                else:
                    q = step_queue

                future = asyncio.Future()
                await q.put((future, language, inputs, out_encoder, cache, k))
                out = await future
            else:
                if k > 0:
                    position_ids = torch.tensor([[k + 3]]).cuda()
                else:
                    position_ids = None
                out = model.model.decoder(
                    inputs,
                    encoder_hidden_states=out_encoder,
                    past_key_values=cache,
                    position_ids=position_ids,
                    use_cache=True,
                    return_dict=False,
                )
                out = list(out)
                logits = model.proj_out(out[0][:, -1:])
                out[0] = logits

            logits = out[0]
            cache = out[1]
            request.scope['cache'] = cache

            if out_encoder is None:
                out_encoder = out[2]

            idx_next, probs = sample(
                logits,
                mask_penalty,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            mask_penalty[0, idx_next[0]] = repetition_penalty
            token = processor.tokenizer.decode(idx_next, decode_with_timestamps=True)

            if k == 0 and 'time_first_token' not in request.scope['request']:
                request.scope['request']['time_first_token'] = time.time()

            ids = idx_next[0].tolist()

            if ids == model.config.eos_token_id:
                break

            del logits, probs, inputs
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
                tokens=processor.tokenizer.encode(substring.strip(), add_special_tokens=False),
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
