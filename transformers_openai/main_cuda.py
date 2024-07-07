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

import asyncio
import time
import logging
import torch
import json


model = None
tokenizer = None


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

    with torch.no_grad():

        if args.architecture_type == 'encoder-decoder':
            out_encoder = model.encoder(inputs, return_dict=False)
            out_encoder = out_encoder
            inputs = torch.tensor([[model.config.decoder_start_token_id]], device='cuda')

        try:

            for k in range(form.max_tokens):

                if args.architecture_type == 'encoder-decoder':
                    out = model(
                        decoder_input_ids=inputs,
                        encoder_outputs=out_encoder,
                        past_key_values=cache,
                        use_cache=True,
                        return_dict=False
                    )

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

    # inputs = torch.concat([inputs] * 3)

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
