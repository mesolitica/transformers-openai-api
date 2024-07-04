from env import HF_MODEL, ATTN_IMPLEMENTATION, TORCH_DTYPE
from app.function import sample, decode
from app.base_model import ChatCompletionForm
from fastapi import Request
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
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
    logging.info(f'loading {HF_MODEL}')
    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL,
        attn_implementation=ATTN_IMPLEMENTATION,
        torch_dtype=getattr(torch, TORCH_DTYPE),
    ).cuda()
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)


async def stream(inputs, id, created, form, request):

    terminators = [tokenizer.eos_token_id]
    replace_tokens = [tokenizer.eos_token]
    for s in form.stop:
        t = tokenizer.convert_tokens_to_ids(s)
        if t is not None:
            terminators.append(t)
        replace_tokens.append(s)

    cache = request.scope['cache']

    mask_penalty = torch.ones((1, model.config.vocab_size)).cuda()

    try:
        with torch.no_grad():
            for _ in range(form.max_tokens):
                logits = model.forward(
                    inputs,
                    past_key_values=cache,
                    use_cache=True,
                    return_dict=False
                )[0]
                idx_next, probs = sample(
                    logits,
                    mask_penalty,
                    top_k=form.top_k,
                    top_p=form.top_p
                )

                mask_penalty[0, idx_next[0]] = form.repetition_penalty
                token = decode(tokenizer, idx_next)

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

    except asyncio.CancelledError as e:

        yield ServerSentEvent(**{"data": str(e)})


async def chat_completions(
    form: ChatCompletionForm,
    request: Request = None,
):
    if model is None:
        load_model()

    prompt = tokenizer.apply_chat_template(form.messages, tokenize=False)
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
