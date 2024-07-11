from typing import List, Tuple, Optional, Dict, Any
from transformers.cache_utils import Cache
from collections import defaultdict
import torch
import torch.nn.functional as F


class DynamicLengthDecoderCache(Cache):

    def __init__(self, lengths) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self._seen_tokens = max(lengths)

    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return self.key_cache[layer_idx], self.value_cache[layer_idx]
        else:
            raise KeyError(
                f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_cache)

    def pad_kv(self, caches):
        """
        List[head, seq, dims]
        """
        shapes = [caches[i].shape[2] for i in range(len(caches))]
        maxlen = max(shapes)
        new_caches = []
        for i in range(len(caches)):
            pad_val = (0, 0, 0, maxlen - caches[i].shape[2], 0, 0, 0, 0)
            pad = F.pad(caches[i], pad_val, value=0.0)
            new_caches.append(pad)
        return torch.concat(new_caches)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        for i in range(len(key_states)):
            self.key_cache[layer_idx][i] = torch.cat(
                [self.key_cache[layer_idx][i], key_states[i: i + 1]], dim=-2)
            self.value_cache[layer_idx][i] = torch.cat(
                [self.value_cache[layer_idx][i], value_states[i: i + 1]], dim=-2)

        k = self.pad_kv(self.key_cache[layer_idx])
        v = self.pad_kv(self.value_cache[layer_idx])
        return k, v

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`
        if len(self.key_cache) <= layer_idx:
            return 0
        return self._seen_tokens

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. DynamicCache does not have a maximum length."""
        return None
