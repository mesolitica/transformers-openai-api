from pydantic import BaseModel
from fastapi import FastAPI, Request
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from transformers import TextStreamer, TextIteratorStreamer
from sse_starlette import EventSourceResponse, ServerSentEvent
from starlette.requests import ClientDisconnect
from threading import Thread
from typing import Union, List, Optional
import asyncio
import time
import logging
import threading
import torch
import json
import os
import uuid

logger = logging.getLogger()
logger.setLevel(os.environ.get('LOGLEVEL', 'INFO').upper())

HF_MODEL = os.environ.get('HF_MODEL', 'mesolitica/malaysian-llama2-7b-32k-instructions')
USE_FLASH_ATTENTION_2 = os.environ.get('USE_FLASH_ATTENTION_2', 'false').lower() == 'true'
HOTLOAD = os.environ.get('HOTLOAD', 'false').lower() == 'true'
TORCH_DTYPE = os.environ.get('TORCH_DTYPE', 'bfloat16')

if HF_MODEL is None:
    raise ValueError('must set `HF_MODEL` in OS environment.')

headers = {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive'
}


class T:
    def __init__(self, **kwargs):
        self.t = Thread(**kwargs)

    def __enter__(self):
        self.t.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.t._delete()


app = FastAPI()


model = None
tokenizer = None


def load_model():
    global model, tokenizer
    if 'AWQ' in HF_MODEL:
        model = AutoModelForCausalLM.from_pretrained(
            HF_MODEL,
            use_flash_attention_2=USE_FLASH_ATTENTION_2,
            device_map='cuda:0',
        )
    else:
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=getattr(torch, TORCH_DTYPE)
        )
        model = AutoModelForCausalLM.from_pretrained(
            HF_MODEL,
            use_flash_attention_2=USE_FLASH_ATTENTION_2,
            quantization_config=nf4_config
        )
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)


class Parameters(BaseModel):
    model: str = 'model'
    temperature: float = 0.9
    top_p: float = 0.95
    top_k: int = 50
    max_tokens: int = 256
    stop: List[str] = []


class ChatMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None

    def __str__(self) -> str:
        if self.role == 'system':
            return f'system:\n{self.content}\n'

        elif self.role == 'user':
            if self.content is None:
                return 'user:\n</s>'
            else:
                return f'user:\n</s>{self.content}\n'

        elif self.role == 'assistant':

            if self.content is None:
                return 'assistant'

            else:
                return f'assistant:\n{self.content}\n'

        else:
            raise ValueError(f'Unsupported role: {self.role}')


class ChatCompletionForm(Parameters):
    messages: List[ChatMessage]
    stream: bool = False


async def stream(generate_kwargs, replace_tokens, id, created):

    try:
        with T(target=model.generate, kwargs=generate_kwargs) as t:
            for text in generate_kwargs['streamer']:
                for t in replace_tokens:
                    text = text.replace(t, '')

                data = {
                    'id': id,
                    'choices': [
                        {'delta': {
                            'content': text,
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


@app.post('/chat/completions')
async def chat_completions(
    form: ChatCompletionForm,
    request: Request = None,
):

    if model is None:
        load_model()

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)

    terminators = [tokenizer.eos_token_id]
    replace_tokens = [tokenizer.eos_token]
    for s in form.stop:
        terminators.append(tokenizer.convert_tokens_to_ids(s))
        replace_tokens.append(s)

    logging.debug(f'input: {form.messages}')
    prompt = tokenizer.apply_chat_template(form.messages, tokenize=False)
    inputs = tokenizer([prompt], return_tensors='pt', add_special_tokens=False).to('cuda')
    generate_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=form.max_tokens,
        top_p=form.top_p,
        top_k=form.top_k,
        temperature=form.temperature,
        eos_token_id=terminators,
        do_sample=True,
        num_beams=1,
        repetition_penalty=1.15,
    )

    id = str(uuid.uuid4())
    created = int(time.time())

    s = stream(generate_kwargs, replace_tokens=replace_tokens, id=id, created=created)

    if form.stream:
        return EventSourceResponse(s, headers=headers)
    else:
        tokens = []
        async for data in s:
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
            'usage': {'completion_tokens': 0, 'prompt_tokens': 0, 'total_tokens': 0}
        }
        return data


@app.get('/')
async def index():
    return {'message': f'serving {HF_MODEL}'}


@app.get('/load')
async def load():
    load_model()
    return {'message': True}

if HOTLOAD:
    logger.info(f'HOTLOAD {HF_MODEL}')
    load_model()
