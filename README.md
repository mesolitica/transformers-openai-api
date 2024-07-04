# transformers-openai-api

OpenAI compatibility using FastAPI HuggingFace Transformers, the models wrapped properly with EventSource so it can serve better concurrency.

1. Can serve user defined max concurrency.
2. Each request got it's own KV Cache using Transformers Dynamic Cache.
3. Disconnected signal, so this is to ensure early cutdown to prevent unnecessary steps.
4. Properly cleanup KV Cache after each requests.

## how-to local API

1. Make sure you already installed Docker and Docker Compose that has Nvidia GPU access, https://docs.docker.com/config/containers/resource_constraints/#gpu

2. Run Docker-compose,

```bash
ATTN_IMPLEMENTATION=flash_attention_2 \
HF_MODEL=mesolitica/malaysian-tinyllama-1.1b-16k-instructions-v4 \
TORCH_DTYPE=bfloat16 \
HOTLOAD=true \
docker-compose up --build
```

```bash
ATTN_IMPLEMENTATION=flash_attention_2 \
HF_MODEL=mesolitica/malaysian-mistral-7b-32k-instructions-v4 \
TORCH_DTYPE=bfloat16 \
HOTLOAD=true \
docker-compose up --build
```

```bash
ATTN_IMPLEMENTATION=flash_attention_2 \
HF_MODEL=mesolitica/malaysian-llama-3-8b-instruct-16k \
TORCH_DTYPE=bfloat16 \
HOTLOAD=true \
docker-compose up --build
```

List of OS environment,

- `HF_MODEL`, huggingface model, default is `mesolitica/malaysian-llama2-7b-32k-instructions`.
- `HOTLOAD`, will hotload the model during API start, if false, will load the model during first API request. default is `false`.
- `TORCH_DTYPE`, torch datatype for bitsandbytes `bnb_4bit_compute_dtype`, default is `bfloat16`.

**bloat16 required 8.0 compute capability**.

3. Access API at http://localhost:7088, or you can use OpenAI,

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

```
ChatCompletion(id='c8695dd2-5ab8-4064-9bd5-c5d666324aa3', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='helo! Bagaimana saya boleh membantu anda hari ini?', role='assistant', function_call=None, tool_calls=None), stop_reason=None)], created=1714235932, model='model', object='chat.completion', system_fingerprint=None, usage=CompletionUsage(completion_tokens=0, prompt_tokens=0, total_tokens=0))
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