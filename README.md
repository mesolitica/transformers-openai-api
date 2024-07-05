# transformers-openai-api

OpenAI compatibility using FastAPI HuggingFace Transformers, the models wrapped properly with EventSource so it can serve better concurrency.

1. Streaming token.
2. Can serve user defined max concurrency.
3. Each request got it's own KV Cache using Transformers Dynamic Cache.
4. Disconnected signal, so this is to ensure early stop.
5. Properly cleanup KV Cache after each requests.
6. Support Encoder-Decoder like T5.

## how-to local API

1. Make sure you already installed Docker and Docker Compose that has Nvidia GPU access, https://docs.docker.com/config/containers/resource_constraints/#gpu

2. Run Docker-compose,

### Run Decoder

```bash
ATTN_IMPLEMENTATION=flash_attention_2 \
HF_MODEL=mesolitica/malaysian-tinyllama-1.1b-16k-instructions-v4 \
MODEL_TYPE=AutoModelForCausalLM \
TOKENIZER_TYPE=AutoTokenizer \
TOKENIZER_USE_FAST=true \
ARCHITECTURE_TYPE=decoder \
CACHE_TYPE=DynamicCache \
TORCH_DTYPE=bfloat16 \
HOTLOAD=true \
docker-compose up --build
```

```bash
ATTN_IMPLEMENTATION=flash_attention_2 \
HF_MODEL=mesolitica/malaysian-mistral-7b-32k-instructions-v4 \
MODEL_TYPE=AutoModelForCausalLM \
TOKENIZER_TYPE=AutoTokenizer \
TOKENIZER_USE_FAST=true \
ARCHITECTURE_TYPE=decoder \
CACHE_TYPE=DynamicCache \
TORCH_DTYPE=bfloat16 \
HOTLOAD=true \
docker-compose up --build
```

```bash
ATTN_IMPLEMENTATION=flash_attention_2 \
HF_MODEL=mesolitica/malaysian-llama-3-8b-instruct-16k \
MODEL_TYPE=AutoModelForCausalLM \
TOKENIZER_TYPE=AutoTokenizer \
TOKENIZER_USE_FAST=true \
ARCHITECTURE_TYPE=decoder \
CACHE_TYPE=DynamicCache \
TORCH_DTYPE=bfloat16 \
HOTLOAD=true \
docker-compose up --build
```

After that you can test using OpenAI library,

```python
from openai import OpenAI

client = OpenAI(
    api_key='-',
    base_url = 'http://localhost:7088'
)

messages = [
    {'role': 'user', 'content': "hello"}
]
response = client.chat.completions.create(
    model='model',
    messages=messages,
    temperature=0.1,
    max_tokens=1024,
    top_p=0.95,
    stop=['[/INST]', '[INST]', '<s>'],
)
```

Output,

```
ChatCompletion(id='c8695dd2-5ab8-4064-9bd5-c5d666324aa3', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='helo! Bagaimana saya boleh membantu anda hari ini?', role='assistant', function_call=None, tool_calls=None), stop_reason=None)], created=1714235932, model='model', object='chat.completion', system_fingerprint=None, usage=CompletionUsage(completion_tokens=0, prompt_tokens=0, total_tokens=0))
```

#### Streaming

https://github.com/mesolitica/transformers-openai-api/assets/19810909/5a8c873b-2a80-4c87-92d8-f50a64bc5adf

### Run Encoder-Decoder

```bash
ATTN_IMPLEMENTATION=eager \
HF_MODEL=google/flan-t5-base \
MODEL_TYPE=T5ForConditionalGeneration \
TOKENIZER_TYPE=AutoTokenizer \
TOKENIZER_USE_FAST=false \
ARCHITECTURE_TYPE=encoder-decoder \
TORCH_DTYPE=bfloat16 \
CACHE_TYPE=none \
HOTLOAD=true \
docker-compose up --build
```

After that you can test using OpenAI library,

```python
from openai import OpenAI

client = OpenAI(
    api_key='-',
    base_url = 'http://localhost:7088'
)

messages = [
    {'role': 'user', 'content': "Q: Can Geoffrey Hinton have a conversation with George Washington? Give the rationale before answering.</s>"}
]
response = client.chat.completions.create(
    model='model',
    messages=messages,
    temperature=0.1,
    max_tokens=1024,
    top_p=0.95,
)
response
```

Output,

```
ChatCompletion(id='026bb93b-095f-4bfb-8540-b9b26ce41259', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content=' Geoffrey Hinton was born in Virginia in 1862. George Washington was born in 1859. The final answer: yes.', role='assistant', function_call=None, tool_calls=None), stop_reason=None)], created=1720149843, model='model', object='chat.completion', system_fingerprint=None, usage=CompletionUsage(completion_tokens=27, prompt_tokens=24, total_tokens=51))
```

#### Streaming

https://github.com/mesolitica/transformers-openai-api/assets/19810909/0ec43628-30da-4d99-b483-a5d166bc335c

Output,

```
data: {"id": "20e9d233-6f6c-4dc4-95a9-7dcf077e9b57", "choices": [{"delta": {"content": " George", "function_call": null, "role": null, "tool_calls": null}, "finish_reason": null, "index": 0, "logprobs": null}], "created": 1720157833, "model": "model", "object": "chat.completion.chunk", "system_fingerprint": null}

data: {"id": "20e9d233-6f6c-4dc4-95a9-7dcf077e9b57", "choices": [{"delta": {"content": " Washington", "function_call": null, "role": null, "tool_calls": null}, "finish_reason": null, "index": 0, "logprobs": null}], "created": 1720157833, "model": "model", "object": "chat.completion.chunk", "system_fingerprint": null}

data: {"id": "20e9d233-6f6c-4dc4-95a9-7dcf077e9b57", "choices": [{"delta": {"content": " died", "function_call": null, "role": null, "tool_calls": null}, "finish_reason": null, "index": 0, "logprobs": null}], "created": 1720157833, "model": "model", "object": "chat.completion.chunk", "system_fingerprint": null}

data: {"id": "20e9d233-6f6c-4dc4-95a9-7dcf077e9b57", "choices": [{"delta": {"content": " on", "function_call": null, "role": null, "tool_calls": null}, "finish_reason": null, "index": 0, "logprobs": null}], "created": 1720157833, "model": "model", "object": "chat.completion.chunk", "system_fingerprint": null}

data: {"id": "20e9d233-6f6c-4dc4-95a9-7dcf077e9b57", "choices": [{"delta": {"content": " June", "function_call": null, "role": null, "tool_calls": null}, "finish_reason": null, "index": 0, "logprobs": null}], "created": 1720157833, "model": "model", "object": "chat.completion.chunk", "system_fingerprint": null}

data: {"id": "20e9d233-6f6c-4dc4-95a9-7dcf077e9b57", "choices": [{"delta": {"content": " 6,", "function_call": null, "role": null, "tool_calls": null}, "finish_reason": null, "index": 0, "logprobs": null}], "created": 1720157833, "model": "model", "object": "chat.completion.chunk", "system_fingerprint": null}

data: {"id": "20e9d233-6f6c-4dc4-95a9-7dcf077e9b57", "choices": [{"delta": {"content": " 17", "function_call": null, "role": null, "tool_calls": null}, "finish_reason": null, "index": 0, "logprobs": null}], "created": 1720157833, "model": "model", "object": "chat.completion.chunk", "system_fingerprint": null}

data: {"id": "20e9d233-6f6c-4dc4-95a9-7dcf077e9b57", "choices": [{"delta": {"content": "65", "function_call": null, "role": null, "tool_calls": null}, "finish_reason": null, "index": 0, "logprobs": null}], "created": 1720157833, "model": "model", "object": "chat.completion.chunk", "system_fingerprint": null}

data: {"id": "20e9d233-6f6c-4dc4-95a9-7dcf077e9b57", "choices": [{"delta": {"content": ".", "function_call": null, "role": null, "tool_calls": null}, "finish_reason": null, "index": 0, "logprobs": null}], "created": 1720157833, "model": "model", "object": "chat.completion.chunk", "system_fingerprint": null}

data: {"id": "20e9d233-6f6c-4dc4-95a9-7dcf077e9b57", "choices": [{"delta": {"content": " George", "function_call": null, "role": null, "tool_calls": null}, "finish_reason": null, "index": 0, "logprobs": null}], "created": 1720157833, "model": "model", "object": "chat.completion.chunk", "system_fingerprint": null}

data: {"id": "20e9d233-6f6c-4dc4-95a9-7dcf077e9b57", "choices": [{"delta": {"content": " Washington", "function_call": null, "role": null, "tool_calls": null}, "finish_reason": null, "index": 0, "logprobs": null}], "created": 1720157833, "model": "model", "object": "chat.completion.chunk", "system_fingerprint": null}

data: {"id": "20e9d233-6f6c-4dc4-95a9-7dcf077e9b57", "choices": [{"delta": {"content": " was", "function_call": null, "role": null, "tool_calls": null}, "finish_reason": null, "index": 0, "logprobs": null}], "created": 1720157833, "model": "model", "object": "chat.completion.chunk", "system_fingerprint": null}

data: {"id": "20e9d233-6f6c-4dc4-95a9-7dcf077e9b57", "choices": [{"delta": {"content": " born", "function_call": null, "role": null, "tool_calls": null}, "finish_reason": null, "index": 0, "logprobs": null}], "created": 1720157833, "model": "model", "object": "chat.completion.chunk", "system_fingerprint": null}

data: {"id": "20e9d233-6f6c-4dc4-95a9-7dcf077e9b57", "choices": [{"delta": {"content": " in", "function_call": null, "role": null, "tool_calls": null}, "finish_reason": null, "index": 0, "logprobs": null}], "created": 1720157833, "model": "model", "object": "chat.completion.chunk", "system_fingerprint": null}

data: {"id": "20e9d233-6f6c-4dc4-95a9-7dcf077e9b57", "choices": [{"delta": {"content": " Washington", "function_call": null, "role": null, "tool_calls": null}, "finish_reason": null, "index": 0, "logprobs": null}], "created": 1720157833, "model": "model", "object": "chat.completion.chunk", "system_fingerprint": null}

data: {"id": "20e9d233-6f6c-4dc4-95a9-7dcf077e9b57", "choices": [{"delta": {"content": ",", "function_call": null, "role": null, "tool_calls": null}, "finish_reason": null, "index": 0, "logprobs": null}], "created": 1720157833, "model": "model", "object": "chat.completion.chunk", "system_fingerprint": null}

data: {"id": "20e9d233-6f6c-4dc4-95a9-7dcf077e9b57", "choices": [{"delta": {"content": " D", "function_call": null, "role": null, "tool_calls": null}, "finish_reason": null, "index": 0, "logprobs": null}], "created": 1720157833, "model": "model", "object": "chat.completion.chunk", "system_fingerprint": null}

data: {"id": "20e9d233-6f6c-4dc4-95a9-7dcf077e9b57", "choices": [{"delta": {"content": ".", "function_call": null, "role": null, "tool_calls": null}, "finish_reason": null, "index": 0, "logprobs": null}], "created": 1720157833, "model": "model", "object": "chat.completion.chunk", "system_fingerprint": null}

data: {"id": "20e9d233-6f6c-4dc4-95a9-7dcf077e9b57", "choices": [{"delta": {"content": "C", "function_call": null, "role": null, "tool_calls": null}, "finish_reason": null, "index": 0, "logprobs": null}], "created": 1720157833, "model": "model", "object": "chat.completion.chunk", "system_fingerprint": null}

data: {"id": "20e9d233-6f6c-4dc4-95a9-7dcf077e9b57", "choices": [{"delta": {"content": ".", "function_call": null, "role": null, "tool_calls": null}, "finish_reason": null, "index": 0, "logprobs": null}], "created": 1720157833, "model": "model", "object": "chat.completion.chunk", "system_fingerprint": null}

data: {"id": "20e9d233-6f6c-4dc4-95a9-7dcf077e9b57", "choices": [{"delta": {"content": " So", "function_call": null, "role": null, "tool_calls": null}, "finish_reason": null, "index": 0, "logprobs": null}], "created": 1720157833, "model": "model", "object": "chat.completion.chunk", "system_fingerprint": null}

data: {"id": "20e9d233-6f6c-4dc4-95a9-7dcf077e9b57", "choices": [{"delta": {"content": " the", "function_call": null, "role": null, "tool_calls": null}, "finish_reason": null, "index": 0, "logprobs": null}], "created": 1720157833, "model": "model", "object": "chat.completion.chunk", "system_fingerprint": null}

data: {"id": "20e9d233-6f6c-4dc4-95a9-7dcf077e9b57", "choices": [{"delta": {"content": " final", "function_call": null, "role": null, "tool_calls": null}, "finish_reason": null, "index": 0, "logprobs": null}], "created": 1720157833, "model": "model", "object": "chat.completion.chunk", "system_fingerprint": null}

data: {"id": "20e9d233-6f6c-4dc4-95a9-7dcf077e9b57", "choices": [{"delta": {"content": " answer", "function_call": null, "role": null, "tool_calls": null}, "finish_reason": null, "index": 0, "logprobs": null}], "created": 1720157833, "model": "model", "object": "chat.completion.chunk", "system_fingerprint": null}

data: {"id": "20e9d233-6f6c-4dc4-95a9-7dcf077e9b57", "choices": [{"delta": {"content": " is", "function_call": null, "role": null, "tool_calls": null}, "finish_reason": null, "index": 0, "logprobs": null}], "created": 1720157833, "model": "model", "object": "chat.completion.chunk", "system_fingerprint": null}

data: {"id": "20e9d233-6f6c-4dc4-95a9-7dcf077e9b57", "choices": [{"delta": {"content": " no", "function_call": null, "role": null, "tool_calls": null}, "finish_reason": null, "index": 0, "logprobs": null}], "created": 1720157833, "model": "model", "object": "chat.completion.chunk", "system_fingerprint": null}

data: {"id": "20e9d233-6f6c-4dc4-95a9-7dcf077e9b57", "choices": [{"delta": {"content": ".", "function_call": null, "role": null, "tool_calls": null}, "finish_reason": null, "index": 0, "logprobs": null}], "created": 1720157833, "model": "model", "object": "chat.completion.chunk", "system_fingerprint": null}
```

## OS Environment

```python
LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()

MODEL_TYPE = os.environ.get('MODEL_TYPE', 'AutoModelForCausalLM')

TOKENIZER_TYPE = os.environ.get('TOKENIZER_TYPE', 'AutoTokenizer')
TOKENIZER_USE_FAST = os.environ.get('TOKENIZER_USE_FAST', 'true').lower() == 'true'

HF_MODEL = os.environ.get('HF_MODEL', 'mesolitica/malaysian-llama2-7b-32k-instructions')
HOTLOAD = os.environ.get('HOTLOAD', 'false').lower() == 'true'

ATTN_IMPLEMENTATION = os.environ.get('ATTN_IMPLEMENTATION', 'sdpa').lower()
TORCH_DTYPE = os.environ.get('TORCH_DTYPE', 'bfloat16')

ARCHITECTURE_TYPE = os.environ.get('ARCHITECTURE_TYPE', 'decoder')

CACHE_TYPE = os.environ.get('CACHE_TYPE', 'DynamicCache')

N_POSITIONS = int(os.environ.get('N_POSITIONS', '2048'))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '1'))

ACCELERATOR_TYPE = os.environ.get('ACCELERATOR_TYPE', 'cuda')

MAX_CONCURRENT = int(os.environ.get('MAX_CONCURRENT', '50'))
```

## How to simulate disconnected?

Simple,

```python
import aiohttp
import asyncio
import json
import time

url = 'http://100.93.25.29:7088/chat/completions'
headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json'
}
payload = {
    "model": "model",
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 50,
    "max_tokens": 256,
    "truncate": 2048,
    "repetition_penalty": 1,
    "stop": [],
    "messages": [
        {
            "role": "user",
            "content": "hello, what is good about malaysia"
        }
    ],
    "stream": True
}

count = 0

async with aiohttp.ClientSession() as session:
    async with session.post(url, headers=headers, json=payload) as response:
        async for line in response.content:
            
            if count > 5:
                break
                
            count += 1
```

You should see warning logs,

```
transformers-openai-api    | WARNING:root:Cancelled by cancel scope 7f686d03b010
transformers-openai-api    | WARNING:root:Cancelling f1bc7ca5-f4a0-4f4d-acd2-d6c6bbdee98c due to disconnect
transformers-openai-api    | INFO:     100.93.25.29:49762 - "POST /chat/completions HTTP/1.1" 200 OK
transformers-openai-api    | WARNING:root:Cancelled by cancel scope 7f686cf93970
transformers-openai-api    | WARNING:root:Cancelling 8537d90d-65c0-410e-acf0-7176accf8f37 due to disconnect
```

## [Stress test](stress-test)

### [T5](stress-test/t5.py)

Rate of 5 users per second, total requests up to 50 users for 30 seconds on shared RTX 3090 Ti FlanT5 Base,

![alt text](stress-test/graph-t5.png)
