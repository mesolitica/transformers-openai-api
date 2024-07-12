from transformers_openai.env import args
from transformers_openai.function import (
    sample,
    decode,
    load_hf_tokenizer,
    load_hf_model,
    pad_attention_mask,
    pad_hidden_encoder,
)
from transformers_openai.base_model import ChatCompletionForm
from transformers_openai.cache import (
    DynamicLengthDecoderCache,
    DynamicLengthEncoderDecoderCache,
)
from fastapi import Request
from sse_starlette import ServerSentEvent

from datetime import datetime
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
                    if len(batch) > args.continous_batching_batch_size:
                        break
                except asyncio.TimeoutError:
                    break

            if not len(batch):
                continue

            logging.debug(f'{str(datetime.now())} prefill batch size of {len(batch)}')

            futures = [batch[i][0] for i in range(len(batch))]
            inputs = [batch[i][1] for i in range(len(batch))]
            lengths = [batch[i][4] for i in range(len(batch))]

            max_len = max(lengths)

            inputs = [{'input_ids': inputs[i][0]} for i in range(len(inputs))]
            input_ids = tokenizer.pad(inputs, padding=True, return_tensors='pt').to('cuda')

            if args.architecture_type == 'encoder-decoder':
                out_encoder = model.encoder(**input_ids, return_dict=False)
                inputs = torch.tensor([[model.config.decoder_start_token_id]]
                                      * len(batch), device='cuda')
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
                    cache_ = [
                        out_caches[k][0][i:i + 1],
                        out_caches[k][1][i:i + 1],
                    ]
                    if args.architecture_type == 'encoder-decoder':
                        cache_.extend([
                            out_caches[k][2][i:i + 1, :, :lengths[i]],
                            out_caches[k][3][i:i + 1, :, :lengths[i]]
                        ])
                    cache.append(cache_)
                caches.append(cache)

            if args.architecture_type == 'encoder-decoder':
                last = []
                for i in range(len(batch)):
                    last.append((input_ids['attention_mask'][i:i + 1,
                                :lengths[i]], out_encoder[i:i + 1, :lengths[i]]))
            else:
                last = [None] * len(batch)

            for i in range(len(futures)):
                futures[i].set_result((out_logits[i: i + 1, -1:], caches[i], last[i]))

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
                    if len(batch) > args.continous_batching_batch_size:
                        break
                except asyncio.TimeoutError:
                    break

            if not len(batch):
                continue

            logging.debug(f'{str(datetime.now())} step batch size of {len(batch)}')

            futures = [batch[i][0] for i in range(len(batch))]
            inputs = [batch[i][1] for i in range(len(batch))]
            out_encoders = [batch[i][2] for i in range(len(batch))]
            caches = [batch[i][3] for i in range(len(batch))]
            lengths = [batch[i][4] for i in range(len(batch))]

            cache_device = caches[0][0][0].device

            kv_len = [caches[i][0][0].shape[2] for i in range(len(batch))]
            max_len = max(kv_len)
            max_len_lengths = max(lengths)

            if args.architecture_type == 'encoder-decoder':
                cache = DynamicLengthEncoderDecoderCache
            else:
                cache = DynamicLengthDecoderCache
            cache = cache(lengths=lengths)

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

                if args.architecture_type == 'encoder-decoder':
                    key_cache = []
                    value_cache = []
                    for i in range(len(batch)):
                        key_cache.append(caches[i][n][2])
                        value_cache.append(caches[i][n][3])

                    cache.cross_key_cache.append(key_cache)
                    cache.cross_value_cache.append(value_cache)

            inputs = torch.concat(inputs, dim=0)

            if args.architecture_type == 'encoder-decoder':

                attention_mask = [out_encoders[i][0] for i in range(len(out_encoders))]
                out_encoder = [out_encoders[i][1] for i in range(len(out_encoders))]

                attention_mask = pad_attention_mask(attention_mask)
                out_encoder = pad_hidden_encoder(out_encoder)

                print(attention_mask.shape, out_encoder.shape)

                out = model(
                    attention_mask=attention_mask,
                    decoder_input_ids=inputs,
                    encoder_outputs=(out_encoder,),
                    past_key_values=cache,
                    use_cache=True,
                    return_dict=False
                )
            else:
                position_ids = [torch.tensor([[lengths[i] - 1]]) for i in range(len(lengths))]
                position_ids = torch.concat(position_ids).to(cache_device)
                out = model(
                    inputs,
                    position_ids=position_ids,
                    past_key_values=cache,
                    use_cache=True,
                    return_dict=False
                )

            out_logits = out[0]

            caches = []
            for i in range(len(batch)):
                new_cache = []
                for k in range(len(cache)):
                    keys = cache.key_cache[k]
                    values = cache.value_cache[k]
                    v = [keys[i], values[i]]
                    if args.architecture_type == 'encoder-decoder':
                        keys = cache.cross_key_cache[k]
                        values = cache.cross_value_cache[k]
                        v.extend([keys[i], values[i]])
                    new_cache.append(v)
                caches.append(new_cache)

            for i in range(len(futures)):
                futures[i].set_result((out_logits[i: i + 1, -1:], caches[i]))

            for k in range(len(cache)):
                temp = list(cache[k])
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

        if args.architecture_type == 'encoder-decoder' and not args.continous_batching:
            out_encoder = model.encoder(inputs, return_dict=False)
            out_encoder = out_encoder
            inputs = torch.tensor([[model.config.decoder_start_token_id]], device='cuda')
        else:
            out_encoder = None

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

                if args.architecture_type == 'encoder-decoder' and out_encoder is None:
                    out_encoder = out[2]

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
