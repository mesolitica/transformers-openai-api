from typing import Union
from pydantic import BaseModel
from fastapi import FastAPI
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer
)
from threading import Thread
from sse_starlette.sse import EventSourceResponse
import logging
import threading
import torch
import json
import os

HF_MODEL = os.environ.get('HF_MODEL', 'mesolitica/malaysian-llama2-7b-32k-instructions')
USE_FLASH_ATTENTION_2 = os.environ.get('USE_FLASH_ATTENTION_2', 'true').lower() == 'true'
HOTLOAD = os.environ.get('HOTLOAD', 'false').lower() == 'true'
TORCH_DTYPE = os.environ.get('TORCH_DTYPE', 'bfloat16')

if HF_MODEL is None:
    raise ValueError('must set `HF_MODEL` in OS environment.')

headers = {
    'Content-Type': 'text/event-stream',
}

threadLock = threading.Lock()


class T:
    def __init__(self, **kwargs):
        self.t = Thread(**kwargs)

    def __enter__(self):
        self.t.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.t._delete()


app = FastAPI()

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=getattr(torch, TORCH_DTYPE)
)


model = None
tokenizer = None
streamer = None


def load_model():
    global model, tokenizer, streamer
    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL,
        use_flash_attention_2=USE_FLASH_ATTENTION_2,
        quantization_config=nf4_config
    )
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
    streamer = TextIteratorStreamer(
        tokenizer,
        timeout=10.,
        skip_prompt=True,
        skip_special_tokens=True
    )


class InputForm(BaseModel):
    inputs: str
    parameters: dict


def stream(generate_kwargs):
    threadLock.acquire()
    with T(target=model.generate, kwargs=generate_kwargs) as t:

        outputs = []
        for text in streamer:
            data = {
                'token': {
                    'text': text,
                }
            }
            yield f'{json.dumps(data)}'
            outputs.append(text)

        outputs = ''.join(outputs)
        data = {
            'generated_text': outputs
        }
        yield f'{json.dumps(data)}'
    threadLock.release()


@app.post('/chatui')
async def send_message(input: InputForm):
    global model, tokenizer, streamer

    if model is None:
        load_model()

    logging.debug(f'input: {input}')
    inputs = tokenizer([input.inputs], return_tensors='pt', add_special_tokens=False).to('cuda')
    generate_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=input.parameters.get('max_new_tokens', 1024),
        top_p=input.parameters.get('top_p', 0.95),
        top_k=input.parameters.get('top_k', 50),
        temperature=input.parameters.get('temperature', 0.9),
        do_sample=True,
        num_beams=1,
    )
    generate_kwargs = {**generate_kwargs, **input.parameters}
    generate_kwargs.pop('truncate', None)
    generate_kwargs.pop('return_full_text', None)
    generate_kwargs.pop('convId', None)

    return EventSourceResponse(stream(generate_kwargs), headers=headers)


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
