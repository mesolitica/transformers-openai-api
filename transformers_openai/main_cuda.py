from transformers_openai.env import args
from transformers_openai.function import (
    sample,
    decode,
    load_hf_tokenizer,
    load_hf_model
)
from transformers_openai.base_model import ChatCompletionForm
from fastapi import Request
from sse_starlette import ServerSentEvent

from datetime import datetime
import torch.nn.functional as F
import threading
import asyncio
import time
import logging
import torch
import json


model = None
tokenizer = None

prefill_queue = asyncio.Queue()
step_queue = asyncio.Queue()
processing = False


async def prefill():
    while True:
        await asyncio.sleep(args.continous_batching_microsleep)
        try:
            batch = []
            while not prefill_queue.empty():
                try:
                    request = await asyncio.wait_for(prefill_queue.get(), timeout=1e-4)
                    batch.append(request)
                except asyncio.TimeoutError:
                    break

            if not len(batch):
                continue

            logging.info(f'{str(datetime.now())} prefill batch size of {len(batch)}')

            futures = [batch[i][0] for i in range(len(batch))]
            inputs = [batch[i][1] for i in range(len(batch))]
            out_encoders = [batch[i][2] for i in range(len(batch))]
            lengths = [batch[i][4] for i in range(len(batch))]

            max_len = max(lengths)

            inputs = [{'input_ids': inputs[i][0]} for i in range(len(inputs))]
            input_ids = tokenizer.pad(inputs, padding=True, return_tensors='pt').to('cuda')

            out = model(
                **input_ids,
                past_key_values=None,
                use_cache=True,
                return_dict=False
            )

            out_logits = out[0]
            out_caches = out[1]
            caches = []
            for i in range(len(batch)):
                cache = []
                for k in range(len(out_caches)):
                    cache_ = []
                    for j in range(len(out_caches[k])):
                        cache_.append(out_caches[k][j][i:i + 1, :, :lengths[i]])
                    cache.append(cache_)
                caches.append(cache)

            for i in range(len(futures)):
                futures[i].set_result((out_logits[i: i + 1, -1:], caches[i]))

            for k in range(len(out_caches)):
                temp = list(out_caches[k])
                for j in range(len(out_caches[k])):
                    del temp[0]

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
                except asyncio.TimeoutError:
                    break

            if not len(batch):
                continue

            logging.info(f'{str(datetime.now())} step batch size of {len(batch)}')

            futures = [batch[i][0] for i in range(len(batch))]
            inputs = [batch[i][1] for i in range(len(batch))]
            out_encoders = [batch[i][2] for i in range(len(batch))]
            caches = [batch[i][3] for i in range(len(batch))]
            lengths = [batch[i][4] for i in range(len(batch))]

            max_len = max([caches[i][0][0].shape[2] for i in range(len(batch))])
            max_len_lengths = max(lengths)
            max_len = min(max_len, max_len_lengths)

            cache_shape = caches[0][0][0].shape
            cache_dtype = caches[0][0][0].dtype
            cache_device = caches[0][0][0].device
            len_cache = len(caches[0])
            len_kv = len(caches[0][0])

            temp_caches = []
            for n in range(len_cache):
                cache = []
                for k in range(len_kv):
                    c = []
                    for i in range(len(batch)):
                        c.append(caches[i][n][k])

                    c = [F.pad(c[i][:, :, :max_len], (0, 0, 0, max(
                        0, max_len - c[i].shape[2]), 0, 0, 0, 0)) for i in range(len(c))]
                    c = torch.concat(c)
                    cache.append(c)

                temp_caches.append(cache)

            inputs = torch.concat(inputs, dim=0)

            out = model(
                inputs,
                past_key_values=temp_caches,
                use_cache=True,
                return_dict=False
            )

            out_logits = out[0]
            out_caches = out[1]
            caches = []
            for i in range(len(batch)):
                cache = []
                for k in range(len(out_caches)):
                    cache_ = []
                    for j in range(len(out_caches[k])):
                        cache_.append(out_caches[k][j][i:i + 1])
                    cache.append(cache_)
                caches.append(cache)

            for i in range(len(futures)):
                futures[i].set_result((out_logits[i: i + 1, -1:], caches[i]))

            for k in range(len(out_caches)):
                temp = list(out_caches[k])
                for j in range(len(temp)):
                    del temp[0]

        except Exception as e:
            print(f"Error in step: {e}")
            futures = [batch[i][0] for i in range(len(batch))]
            for i in range(len(futures)):
                if not futures[i].done():
                    futures[i].set_exception(e)


def load_model():
    global model, tokenizer
    model = load_hf_model()
    tokenizer = load_hf_tokenizer()


async def stream(inputs, id, created, form, request):

    terminators = [tokenizer.eos_token_id]
    replace_tokens = [tokenizer.eos_token]
    for s in form.stop:
        t = tokenizer.convert_tokens_to_ids(s)
        if t is not None:
            terminators.append(t)
        replace_tokens.append(s)

    cache = request.scope['cache']
    cache_is_none = cache is None

    mask_penalty = torch.ones((len(inputs), model.config.vocab_size)).cuda()

    before = time.time()

    initial_length = inputs.shape[1]

    with torch.no_grad():

        if args.architecture_type == 'encoder-decoder':
            out_encoder = model.encoder(inputs, return_dict=False)
            out_encoder = out_encoder
            inputs = torch.tensor([[model.config.decoder_start_token_id]], device='cuda')

        try:

            for k in range(form.max_tokens):

                if args.architecture_type == 'encoder-decoder':
                    if args.continous_batching:
                        if k == 0:
                            q = prefill_queue
                        else:
                            q = step_queue

                        future = asyncio.Future()
                        await q.put((future, inputs, out_encoder, cache, initial_length + k))
                        out = await future
                    else:
                        out = model(
                            decoder_input_ids=inputs,
                            encoder_outputs=out_encoder,
                            past_key_values=cache,
                            use_cache=True,
                            return_dict=False
                        )

                else:
                    if args.continous_batching:
                        if k == 0:
                            q = prefill_queue
                        else:
                            q = step_queue

                        future = asyncio.Future()
                        await q.put((future, inputs, None, cache, initial_length + k))
                        out = await future
                    else:
                        out = model(
                            inputs,
                            past_key_values=cache,
                            use_cache=True,
                            return_dict=False
                        )

                logits = out[0]
                if cache_is_none:
                    cache = out[1]
                    request.scope['cache'] = cache

                idx_next, probs = sample(
                    logits,
                    mask_penalty,
                    top_k=form.top_k,
                    top_p=form.top_p
                )

                mask_penalty[0, idx_next[0]] = form.repetition_penalty
                token = decode(tokenizer, idx_next)

                if k == 0:
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
                    'id': id,
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

            request.scope['request']['time_max_tokens'] = time.time()
            request.scope['request']['total_tokens'] = k

        except asyncio.CancelledError as e:
            logging.warning(f"model step cancelled {request.scope['request']['uuid']}")
            yield ServerSentEvent(**{"data": str(e)})


async def chat_completions(
    form: ChatCompletionForm,
    request: Request = None,
):
    if model is None:
        load_model()

    if tokenizer.chat_template is not None:
        prompt = tokenizer.apply_chat_template(form.messages, tokenize=False)
    else:
        prompt = form.messages[0].content

    inputs = tokenizer.encode(
        prompt,
        return_tensors='pt',
        add_special_tokens=False,
    ).to('cuda')

    id = request.scope['request']['uuid']
    created = int(time.time())

    func = stream(inputs=inputs, id=id, created=created, form=form, request=request)

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
            'id': id,
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
