from transformers_openai.env import args
from transformers_openai.function import (
    sample,
    pad_attention_mask,
    pad_hidden_encoder,
    prefill_attention_mask,
    efficient_attention_mask,
)
from transformers_openai.function_hf import (
    load_hf_tokenizer,
    load_hf_model,
    decode,
)
from transformers_openai.base_model import ChatCompletionForm
from transformers_openai.cache import (
    DynamicLengthDecoderCache,
    DynamicLengthEncoderDecoderCache,
)
from fastapi import Request
from sse_starlette import ServerSentEvent
from datetime import datetime
from contextlib import nullcontext
import asyncio
import time
import logging
import torch
import json
import os, sys
import traceback
import gc

model = None
tokenizer = None
global_cache = None

torch_dtype = getattr(torch, args.torch_dtype)
device = args.device

prefill_queue = asyncio.Queue()
step_queue = asyncio.Queue()

def load_model():
    global model, tokenizer, global_cache

    model = load_hf_model()
    tokenizer = load_hf_tokenizer()

    if args.architecture_type == 'encoder-decoder':
        global_cache = DynamicLengthEncoderDecoderCache()
    else:
        if args.static_cache:
            logging.info('initializing static cache')
            global_cache = StaticLengthDecoderCache(
                max_length = args.static_cache_max_length,
                device = device,
                head_size = model.config.num_attention_heads,
                dim_size = model.config.hidden_size // model.config.num_attention_heads,
                num_hidden_layers = model.config.num_hidden_layers,
                dtype = model.dtype,
            )
        else:
            logging.info('initializing dynamic cache')
            global_cache = DynamicLengthDecoderCache()

profiler = lambda: torch.autograd.profiler.profile(use_cuda = True, use_kineto = True, use_cpu = False)

@torch.no_grad()
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
                    request = await asyncio.wait_for(prefill_queue.get(), timeout=1e-6)
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
            inputs = [batch[i][1] for i in range(len(batch))]
            lengths = [batch[i][3] for i in range(len(batch))]
            uuids = [batch[i][4] for i in range(len(batch))]

            max_len = max(lengths)

            context = profiler() if args.torch_autograd_profiling else nullcontext()

            with context as prof:
                inputs = [{'input_ids': inputs[i][0]} for i in range(len(inputs))]
                input_ids = tokenizer.pad(inputs, padding=True, return_tensors='pt').to(device)

                if args.architecture_type == 'encoder-decoder':
                    out_encoder = model.encoder(**input_ids, return_dict=False)
                    inputs = torch.tensor([[model.config.decoder_start_token_id]] * len(batch), device=device)
                    out = model(
                        attention_mask=input_ids['attention_mask'],
                        decoder_input_ids=inputs,
                        encoder_outputs=out_encoder,
                        past_key_values=None,
                        use_cache=True,
                        return_dict=False
                    )
                    out_encoder = out_encoder[0]
                else:
                    print(input_ids, lengths)
                    attention_mask = prefill_attention_mask(
                        batch_size=len(lengths),
                        max_len=max_len,
                        lengths=lengths,
                        device=device,
                        dtype=torch_dtype,
                    )
                    out = model(
                        input_ids=input_ids['input_ids'],
                        attention_mask=attention_mask,
                        past_key_values=None,
                        use_cache=True,
                        return_dict=False
                    )

                out_logits = out[0]
                out_caches = out[1]

                cache_exists = len(global_cache.key_cache) > 0
                print(cache_exists, lengths)

                for k in range(len(out_caches)):
                    key_cache = {}
                    value_cache = {}
                    cross_key_cache = {}
                    cross_value_cache = {}
                    for i in range(len(batch)):
                        key_cache[uuids[i]] = out_caches[k][0][i: i + 1, :, :lengths[i]]
                        value_cache[uuids[i]] = out_caches[k][1][i: i + 1, :, :lengths[i]]
                        if args.architecture_type == 'encoder-decoder':
                            cross_key_cache[uuids[i]] = out_caches[k][2][i: i + 1, :, :lengths[i]]
                            cross_value_cache[uuids[i]] = out_caches[k][3][i: i + 1, :, :lengths[i]]

                    if cache_exists:
                        global_cache.key_cache[k].update(key_cache)
                        global_cache.value_cache[k].update(value_cache)
                        if args.architecture_type == 'encoder-decoder':
                            global_cache.cross_key_cache[k].update(cross_key_cache)
                            global_cache.cross_value_cache[k].update(cross_value_cache)
                    else:
                        global_cache.key_cache.append(key_cache)
                        global_cache.value_cache.append(value_cache)
                        if args.architecture_type == 'encoder-decoder':
                            global_cache.cross_key_cache.append(cross_key_cache)
                            global_cache.cross_value_cache.append(cross_value_cache)

                if args.architecture_type == 'encoder-decoder':
                    last = []
                    for i in range(len(batch)):
                        last.append((
                            input_ids['attention_mask'][i:i + 1, :lengths[i]], 
                            out_encoder[i:i + 1, :lengths[i]]))
                else:
                    last = [None] * len(batch)

                print(lengths, out_logits.shape)
                for i in range(len(futures)):
                    # if
                    # else:
                    o = out_logits[i:i + 1, lengths[i] - 1:lengths[i]]
                    futures[i].set_result((o, last[i]))
            
            if args.torch_autograd_profiling:
                print(prof.key_averages().table(sort_by='self_cpu_time_total'))

            for k in range(len(out_caches)):
                temp = list(out_caches[k])
                for j in range(len(out_caches[k])):
                    del temp[0]

        except Exception as e:
            logging.warning(f"Error in prefill: {e}")
            futures = [batch[i][0] for i in range(len(batch))]
            for i in range(len(futures)):
                if not futures[i].done():
                    futures[i].set_exception(e)

@torch.no_grad()
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
            inputs = [batch[i][1] for i in range(len(batch))]
            out_encoders = [batch[i][2] for i in range(len(batch))]
            lengths = [batch[i][3] for i in range(len(batch))]
            uuids = [batch[i][4] for i in range(len(batch))]

            global_cache.current_uuid = uuids
            max_len = max(lengths)

            context = profiler() if args.torch_autograd_profiling else nullcontext()

            with context as prof:
                inputs = torch.concat(inputs, dim=0)
                attention_mask = efficient_attention_mask(
                    batch_size=len(lengths),
                    max_len=max_len,
                    lengths=lengths,
                    device=device,
                    dtype=torch_dtype,
                    ones=args.architecture_type == 'encoder-decoder'
                )

                if args.architecture_type == 'encoder-decoder':
                    encoder_attention_mask = [out_encoders[i][0] for i in range(len(out_encoders))]
                    out_encoder = [out_encoders[i][1] for i in range(len(out_encoders))]
                    encoder_attention_mask = pad_attention_mask(encoder_attention_mask)
                    out_encoder = pad_hidden_encoder(out_encoder)
                    out = model(
                        attention_mask=encoder_attention_mask,
                        decoder_attention_mask=attention_mask[:, 0],
                        decoder_input_ids=inputs,
                        encoder_outputs=(out_encoder,),
                        past_key_values=global_cache,
                        use_cache=True,
                        return_dict=False
                    )
                else:
                    position_ids = torch.tensor([[l - 1 for l in lengths]]).T.to(device)
                    out = model(
                        inputs,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=global_cache,
                        use_cache=True,
                        return_dict=False
                    )

                out_logits = out[0]
            
            if args.torch_autograd_profiling:
                print(prof.key_averages().table(sort_by='self_cpu_time_total'))

            for i in range(len(futures)):
                futures[i].set_result((out_logits[i: i + 1, -1:],))

        except Exception as e:
            logging.warning(f"Error in step: {e}")
            futures = [batch[i][0] for i in range(len(batch))]
            for i in range(len(futures)):
                if not futures[i].done():
                    futures[i].set_exception(e)


async def stream(inputs, created, form, request):

    terminators = [tokenizer.eos_token_id]
    replace_tokens = [tokenizer.eos_token]
    for s in form.stop:
        t = tokenizer.convert_tokens_to_ids(s)
        if t is not None:
            terminators.append(t)
        replace_tokens.append(s)

    mask_penalty = torch.ones((len(inputs), model.config.vocab_size)).cuda()

    initial_length = inputs.shape[1]
    if isinstance(request, dict):
        uuid = request['uuid']
    else:
        uuid = request.scope['request']['uuid']
    out_encoder = None

    try:
        for k in range(form.max_tokens):
            if args.architecture_type == 'encoder-decoder':
                if k == 0:
                    l = initial_length
                else:
                    l = k + 1
            else:
                l = k + initial_length

            if k == 0:
                q = prefill_queue
            else:
                q = step_queue

            future = asyncio.Future()
            await q.put((future, inputs, out_encoder, l, uuid))
            out = await future
            logits = out[0]

            if args.architecture_type == 'encoder-decoder' and out_encoder is None:
                out_encoder = out[1]

            idx_next, probs = sample(
                logits,
                mask_penalty,
                temperature=form.temperature,
                top_k=form.top_k,
                top_p=form.top_p
            )

            mask_penalty[0, idx_next[0]] = form.repetition_penalty
            token = decode(tokenizer, idx_next)

            if k == 0 and not isinstance(request, dict):
                request.scope['request']['time_first_token'] = time.time()

            for t in replace_tokens:
                token = token.replace(t, '')

            ids = idx_next[0].tolist()

            if ids == tokenizer.eos_token_id:
                break

            if ids in terminators:
                break

            del logits, probs, inputs
            inputs = idx_next.unsqueeze(0)

            data = {
                'id': uuid,
                'choices': [
                    {'delta': {
                        'content': token,
                        'function_call': None,
                        'role': None,
                        'tool_calls': None
                    },
                        'finish_reason': None,
                        'index': 0,
                        'logprobs': None
                    }
                ],
                'created': created,
                'model': 'model',
                'object': 'chat.completion.chunk',
                'system_fingerprint': None
            }
            yield json.dumps(data)
            await asyncio.sleep(0)

        if not isinstance(request, dict):
            request.scope['request']['time_max_tokens'] = time.time()
            request.scope['request']['total_tokens'] = k + 1

    except asyncio.CancelledError as e:
        logging.warning(f"model step cancelled {uuid}")
        yield ServerSentEvent(**{"data": str(e)})
    
    except Exception as e:
        logging.error(f"model step exception {e} {uuid}")
        yield ServerSentEvent(**{"data": str(e)})

    finally:
        print('purging')
        logging.debug(f'purging {uuid} KV cache')
        for i in range(len(global_cache.key_cache)):
            key_cache = global_cache.key_cache[i].pop(uuid, None)
            value_cache = global_cache.value_cache[i].pop(uuid, None)
            if args.architecture_type == 'encoder-decoder':
                cross_key_cache = global_cache.cross_key_cache[i].pop(uuid, None)
                cross_value_cache = global_cache.cross_value_cache[i].pop(uuid, None)
        torch.cuda.empty_cache()
        gc.collect()


async def chat_completions(
    form: ChatCompletionForm,
    request: Request = None,
):

    if tokenizer.chat_template is not None:
        prompt = tokenizer.apply_chat_template(form.messages, tokenize=False)
    else:
        prompt = form.messages[0].content

    inputs = tokenizer.encode(
        prompt,
        return_tensors='pt',
        add_special_tokens=False,
    ).to(device)

    created = int(time.time())
    if isinstance(request, dict):
        uuid = request['uuid']
    else:
        uuid = request.scope['request']['uuid']

    print(uuid, prompt)
    func = stream(inputs=inputs, created=created, form=form, request=request)

    if form.stream:
        return func
    else:
        tokens = []
        async for data in func:
            if isinstance(data, ServerSentEvent):
                continue
            data = json.loads(data)
            tokens.append(data['choices'][0]['delta']['content'])

        data = {
            'id': uuid,
            'choices': [
                {'finish_reason': 'stop',
                 'index': 0,
                 'logprobs': None,
                 'message': {
                     'content': ''.join(tokens),
                     'role': 'assistant',
                     'function_call': None,
                     'tool_calls': None
                 },
                 'stop_reason': None
                 }
            ],
            'created': created,
            'model': 'model',
            'object': 'chat.completion',
            'system_fingerprint': None,
            'usage': {
                'completion_tokens': len(tokens),
                'prompt_tokens': len(inputs[0]),
                'total_tokens': len(inputs[0]) + len(tokens),
            }
        }
        return data
