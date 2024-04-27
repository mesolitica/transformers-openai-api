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

List of OS environment,

- `HF_MODEL`, huggingface model, default is `mesolitica/malaysian-llama2-7b-32k-instructions`.
- `HOTLOAD`, will hotload the model during API start, if false, will load the model during first API request. default is `false`.
- `TORCH_DTYPE`, torch datatype for bitsandbytes `bnb_4bit_compute_dtype`, default is `bfloat16`.

**bloat16 required 8.0 compute capability**.

3. Access API at http://localhost:7075