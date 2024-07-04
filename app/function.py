import torch
import logging
from typing import Optional

SPIECE_UNDERLINE = "▁"


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
