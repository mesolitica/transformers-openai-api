import logging
import os

HEADERS = {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive'
}

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

MAX_CONCURRENT = int(os.environ.get('MAX_CONCURRENT', '100'))

if HF_MODEL is None:
    raise ValueError('must set `HF_MODEL` in OS environment.')

ACCEPTABLE_ARCHITECTURE = {'decoder', 'encoder-decoder'}
if ARCHITECTURE_TYPE not in ACCEPTABLE_ARCHITECTURE:
    raise ValueError(f'{ARCHITECTURE_TYPE} only accept one of {ACCEPTABLE_ARCHITECTURE}.')

logging.basicConfig(level=LOGLEVEL)
