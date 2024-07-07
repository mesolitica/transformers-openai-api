# transformers-openai-api

OpenAI compatibility using FastAPI HuggingFace Transformers, the models wrapped properly with EventSource so it can serve better concurrency.

1. Streaming token.
2. Can serve user defined max concurrency.
3. Each request got it's own KV Cache using Transformers Dynamic Cache.
4. Disconnected signal, so this is to ensure early stop.
5. Properly cleanup KV Cache after each requests.
6. Support Encoder-Decoder like T5.

## how to install

Using PIP with git,

```bash
pip3 install git+https://github.com/mesolitica/transformers-openai-api
```

Or you can git clone,

```bash
git clone https://github.com/mesolitica/transformers-openai-api && cd transformers-openai-api
```

## how to local

For docker, Make sure you already installed Docker and Docker Compose that has Nvidia GPU access, https://docs.docker.com/config/containers/resource_constraints/#gpu

### Supported parameters

```bash
python3 -m transformers_openai.main --help
```

```
usage: main.py [-h] [--host HOST] [--port PORT] [--loglevel LOGLEVEL] [--model-type MODEL_TYPE] [--tokenizer-type TOKENIZER_TYPE]
               [--tokenizer-use-fast TOKENIZER_USE_FAST] [--hf-model HF_MODEL] [--hotload HOTLOAD]
               [--attn-implementation ATTN_IMPLEMENTATION] [--torch-dtype TORCH_DTYPE]
               [--architecture-type {decoder,encoder-decoder}] [--cache-type CACHE_TYPE] [--n-positions N_POSITIONS]
               [--batch-size BATCH_SIZE] [--accelerator-type ACCELERATOR_TYPE] [--max-concurrent MAX_CONCURRENT]

Configuration parser

options:
  -h, --help            show this help message and exit
  --host HOST           host name to host the app (default: 0.0.0.0, env: HOSTNAME)
  --port PORT           port to host the app (default: 7088, env: PORT)
  --loglevel LOGLEVEL   Logging level (default: INFO, env: LOGLEVEL)
  --model-type MODEL_TYPE
                        Model type (default: AutoModelForCausalLM, env: MODEL_TYPE)
  --tokenizer-type TOKENIZER_TYPE
                        Tokenizer type (default: AutoTokenizer, env: TOKENIZER_TYPE)
  --tokenizer-use-fast TOKENIZER_USE_FAST
                        Use fast tokenizer (default: True, env: TOKENIZER_USE_FAST)
  --hf-model HF_MODEL   Hugging Face model (default: mesolitica/malaysian-llama2-7b-32k-instructions, env: HF_MODEL)
  --hotload HOTLOAD     Enable hot loading (default: True, env: HOTLOAD)
  --attn-implementation ATTN_IMPLEMENTATION
                        Attention implementation (default: sdpa, env: ATTN_IMPLEMENTATION)
  --torch-dtype TORCH_DTYPE
                        Torch data type (default: bfloat16, env: TORCH_DTYPE)
  --architecture-type {decoder,encoder-decoder}
                        Architecture type (default: decoder, env: ARCHITECTURE_TYPE)
  --cache-type CACHE_TYPE
                        Cache type (default: DynamicCache, env: CACHE_TYPE)
  --n-positions N_POSITIONS
                        Number of positions (default: 2048, env: N_POSITIONS)
  --batch-size BATCH_SIZE
                        Batch size (default: 1, env: BATCH_SIZE)
  --accelerator-type ACCELERATOR_TYPE
                        Accelerator type (default: cuda, env: ACCELERATOR_TYPE)
  --max-concurrent MAX_CONCURRENT
                        Maximum concurrent requests (default: 100, env: MAX_CONCURRENT)
```

**We support both args and OS environment**.

### Run Decoder

#### Using CLI

```bash
python3 -m transformers_openai.main \
--host 0.0.0.0 --port 7088 --hf-model mesolitica/malaysian-tinyllama-1.1b-16k-instructions-v4
```

#### Using Docker

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

#### Example OpenAI library

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

Recorded streaming,

https://github.com/mesolitica/transformers-openai-api/assets/19810909/5a8c873b-2a80-4c87-92d8-f50a64bc5adf

### Run Encoder-Decoder

#### Using CLI

```bash
python3 -m transformers_openai.main \
--host 0.0.0.0 --port 7088 \
--attn-implementation eager \
--model-type T5ForConditionalGeneration \
--tokenizer-type AutoTokenizer \
--tokenizer-use-fast false \
--architecture-type encoder-decoder \
--torch-dtype bfloat16 \
--cache-type none \
--hf-model google/flan-t5-base
```

#### Using Docker

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

#### Example OpenAI library

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

Recorded streaming,

https://github.com/mesolitica/transformers-openai-api/assets/19810909/0ec43628-30da-4d99-b483-a5d166bc335c

Output streaming,

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

## How to simulate disconnected?

Simple,

```python
import aiohttp
import asyncio
import json
import time

url = 'http://localhost:7088/chat/completions'
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
            
            if count > 3:
                break
                
            count += 1
```

You should see warning logs,

```
INFO:root:Received request ae6af2a2-c1a3-4e5f-a9cf-eb1cf645870e in queue 1.9073486328125e-06
INFO:     127.0.0.1:60416 - "POST /chat/completions HTTP/1.1" 200 OK
WARNING:root:
WARNING:root:Cancelling ae6af2a2-c1a3-4e5f-a9cf-eb1cf645870e due to disconnect
```

## [Stress test](stress-test)

### [FlanT5 Base](stress-test/t5.py)

Rate of 5 users per second, total requests up to 50 users for 30 seconds on shared RTX 3090 Ti,

![alt text](stress-test/graph-t5.png)
