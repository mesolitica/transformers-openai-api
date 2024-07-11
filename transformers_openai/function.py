from transformers_openai.env import args
import torch
import logging
import importlib
from typing import Optional
import transformers

SPIECE_UNDERLINE = "▁"


def load_hf_model():
    if 't5' in args.model_type.lower() and 'sdpa' not in args.attn_implementation.lower():
        logging.warning(
            'you are using T5 without SDPA, might want to set `--attn-implementation sdpa --model-type transformers_openai.models.T5ForConditionalGeneration`')

    if '.' in args.model_type:

        module_name, class_name = args.model_type.rsplit('.', 1)
        module = importlib.import_module(module_name)
        class_ = getattr(module, class_name)

    else:
        class_ = getattr(transformers, args.model_type)

    logging.info(f'loading {class_} {args.hf_model}')

    if 'auto_gptq' in module_name:
        return class_.from_quantized(args.hf_model)

    else:
        return class_.from_pretrained(
            args.hf_model,
            attn_implementation=args.attn_implementation,
            torch_dtype=getattr(torch, args.torch_dtype),
        ).cuda()


def load_hf_tokenizer():
    tokenizer = getattr(transformers, args.tokenizer_type).from_pretrained(
        args.hf_model,
        use_fast=args.tokenizer_use_fast,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0
    return tokenizer


def decode(tokenizer, ids):
    tokens = tokenizer.convert_ids_to_tokens(ids)
    if SPIECE_UNDERLINE in tokens[0]:
        prefix = ' '
    else:
        prefix = ''
    return prefix + tokenizer.convert_tokens_to_string(tokens)


def multinomial_sample_one_no_sync(probs_sort):
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def logits_to_probs(
    logits,
    mask_penalty,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
):

    logits = logits / mask_penalty
    if temperature > 0:
        logits = logits / max(temperature, 1e-5)

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            pivot = v.select(-1, -1).unsqueeze(-1)
            logits = torch.where(logits < pivot, -float("Inf"), logits)

        probs = torch.nn.functional.softmax(logits, dim=-1)

        if top_p is not None:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            indices_to_remove = cumulative_probs > top_p
            indices_to_remove[..., 1:] = indices_to_remove[..., :-1].clone()
            indices_to_remove[..., 0] = 0
            probs[sorted_indices[indices_to_remove]] = 0.0
            probs = probs / probs.sum(dim=-1, keepdim=True)  # renormalize

        idx_next = multinomial_sample_one_no_sync(probs)
    else:
        probs = logits
        idx_next = logits.argmax(-1)

    return idx_next, probs


def sample(
    logits,
    mask_penalty,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None
):
    return logits_to_probs(logits[0, -1], mask_penalty[0], temperature, top_k, top_p)
