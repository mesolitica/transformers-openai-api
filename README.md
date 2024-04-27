# huggingface-openai-api

OpenAI compatibility using FastAPI and Vanilla bitsandbytes HuggingFace Transformers, because you just don't care about speed.

## how-to local API

1. Make sure you already installed Docker and Docker Compose that has Nvidia GPU access, https://docs.docker.com/config/containers/resource_constraints/#gpu

2. Run Docker-compose,

```bash
USE_FLASH_ATTENTION_2=true \
HF_MODEL=mesolitica/malaysian-mistral-7b-32k-instructions-v4 \
TORCH_DTYPE=bfloat16 \
HOTLOAD=true \
docker-compose up --build
```

```bash
USE_FLASH_ATTENTION_2=true \
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

3. Access API at http://localhost:7075, or you can use OpenAI,

```python
from openai import OpenAI

client = OpenAI(
    api_key='-',
    base_url = 'http://localhost:7075'
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