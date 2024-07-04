import logging
import os

logging.basicConfig(level=os.environ.get('LOGLEVEL', 'INFO').upper())

HEADERS = {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive'
}

HF_MODEL = os.environ.get('HF_MODEL', 'mesolitica/malaysian-llama2-7b-32k-instructions')
HOTLOAD = os.environ.get('HOTLOAD', 'false').lower() == 'true'

ATTN_IMPLEMENTATION = os.environ.get('ATTN_IMPLEMENTATION', 'sdpa').lower()
TORCH_DTYPE = os.environ.get('TORCH_DTYPE', 'bfloat16')

MODEL_TYPE = os.environ.get('MODEL_TYPE', 'decoder')

N_POSITIONS = int(os.environ.get('N_POSITIONS', '2048'))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '1'))

ACCELERATOR_TYPE = os.environ.get('ACCELERATOR_TYPE', 'cuda')

MAX_PROCESS = int(os.environ.get('MAX_PROCESS', '50'))

if HF_MODEL is None:
    raise ValueError('must set `HF_MODEL` in OS environment.')
