import argparse
import logging
import os


def parse_arguments():
    parser = argparse.ArgumentParser(description='Configuration parser')

    parser.add_argument(
        '--host', type=str, default=os.environ.get('HOSTNAME', '0.0.0.0'),
        help='host name to host the app (default: %(default)s, env: HOSTNAME)'
    )
    parser.add_argument(
        '--port', type=int, default=int(os.environ.get('PORT', '7088')),
        help='port to host the app (default: %(default)s, env: PORT)'
    )
    parser.add_argument(
        '--loglevel', default=os.environ.get('LOGLEVEL', 'INFO').upper(),
        help='Logging level (default: %(default)s, env: LOGLEVEL)'
    )
    parser.add_argument(
        '--model-type',
        default=os.environ.get(
            'MODEL_TYPE',
            'AutoModelForCausalLM'),
        help='Model type (default: %(default)s, env: MODEL_TYPE)'
    )
    parser.add_argument(
        '--tokenizer-type',
        default=os.environ.get(
            'TOKENIZER_TYPE',
            'AutoTokenizer'),
        help='Tokenizer type (default: %(default)s, env: TOKENIZER_TYPE)'
    )
    parser.add_argument(
        '--tokenizer-use-fast', type=lambda x: x.lower() == 'true',
        default=os.environ.get('TOKENIZER_USE_FAST', 'true').lower() == 'true',
        help='Use fast tokenizer (default: %(default)s, env: TOKENIZER_USE_FAST)'
    )
    parser.add_argument(
        '--hf-model',
        default=os.environ.get(
            'HF_MODEL',
            'mesolitica/malaysian-llama2-7b-32k-instructions'),
        help='Hugging Face model (default: %(default)s, env: HF_MODEL)'
    )
    parser.add_argument(
        '--hotload', type=lambda x: x.lower() == 'true',
        default=os.environ.get('HOTLOAD', 'true').lower() == 'true',
        help='Enable hot loading (default: %(default)s, env: HOTLOAD)'
    )
    parser.add_argument(
        '--attn-implementation',
        default=os.environ.get(
            'ATTN_IMPLEMENTATION',
            'sdpa').lower(),
        help='Attention implementation (default: %(default)s, env: ATTN_IMPLEMENTATION)'
    )
    parser.add_argument(
        '--torch-dtype', default=os.environ.get('TORCH_DTYPE', 'bfloat16'),
        help='Torch data type (default: %(default)s, env: TORCH_DTYPE)'
    )
    parser.add_argument(
        '--architecture-type',
        default=os.environ.get(
            'ARCHITECTURE_TYPE',
            'decoder'
        ),
        choices=['decoder', 'encoder-decoder'],
        help='Architecture type (default: %(default)s, env: ARCHITECTURE_TYPE)'
    )
    parser.add_argument(
        '--cache-type', default=os.environ.get('CACHE_TYPE', 'none'),
        help='Cache type (default: %(default)s, env: CACHE_TYPE)'
    )
    parser.add_argument(
        '--continous-batching', type=lambda x: x.lower() == 'true',
        default=os.environ.get('CONTINOUS_BATCHING', 'false').lower() == 'true',
        help='Enable continous batching (default: %(default)s, env: CONTINOUS_BATCHING)'
    )
    parser.add_argument(
        '--continous-batching-microsleep', type=float,
        default=float(os.environ.get('CONTINOUS_BATCHING_MICROSLEEP', '1e-4')),
        help='microsleep to group continous batching, 1 / 1e-3 = 1k steps for second (default: %(default)s, env: CONTINOUS_BATCHING_MICROSLEEP)'
    )

    parser.add_argument(
        '--n-positions',
        type=int,
        default=int(
            os.environ.get(
                'N_POSITIONS',
                '2048')),
        help='Number of positions (default: %(default)s, env: N_POSITIONS)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=int(os.environ.get('BATCH_SIZE', '1')),
        help='Batch size (default: %(default)s, env: BATCH_SIZE)'
    )
    parser.add_argument(
        '--accelerator-type', default=os.environ.get('ACCELERATOR_TYPE', 'cuda'),
        help='Accelerator type (default: %(default)s, env: ACCELERATOR_TYPE)'
    )
    parser.add_argument(
        '--max-concurrent',
        type=int,
        default=int(
            os.environ.get(
                'MAX_CONCURRENT',
                '100')),
        help='Maximum concurrent requests (default: %(default)s, env: MAX_CONCURRENT)'
    )

    args = parser.parse_args()

    if args.hf_model is None:
        raise ValueError('must set `--hf-model` or `HF_MODEL` environment variable.')

    return args


args = parse_arguments()

logging.basicConfig(level=args.loglevel)

logging.info(f'Serving app using {args}')
