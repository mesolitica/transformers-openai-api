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
            'you are using T5 without SDPA, might want to use this fork https://github.com/mesolitica/t5-sdpa')

    logging.info(f'loading model {args.hf_model}')

    if '.' in args.model_type:
        module_name, class_name = args.model_type.split('.')
        module = importlib.import_module(module_name)
        class_ = getattr(module, class_name)
        if module_name == 'auto_gptq':
            return class_.from_quantized(args.hf_model)
        else:
            return class_.from_pretrained(args.hf_model)

    else:
        return getattr(transformers, args.model_type).from_pretrained(
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
    tokenizer.padding_side = 'left'
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

    return probs


def sample(
    logits,
    mask_penalty,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None
):
    probs = logits_to_probs(logits[0, -1], mask_penalty[0], temperature, top_k, top_p)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs
