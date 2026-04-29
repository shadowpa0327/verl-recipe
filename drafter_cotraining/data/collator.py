"""Right-pad variable-length samples to a uniform bucketed length.

Generic refactor of TorchSpec ``DataCollatorWithPadding``. Pads any per-sample
dict by tensor rank (2D ``[1,T]`` or 3D ``[1,T,D]``); only keys present in
*every* sample are emitted (so optional features like Eagle3's
``last_hidden_states`` are dropped automatically when missing).

The 256-token bucket bounds the set of distinct ``T_pad`` shapes the draft
model sees, so torch.compile / FlexAttention recompiles are bounded.

Per-sample input dict expected (Eagle3 example, but any keys with rank 2 or 3
work):
    input_ids:          [1, T]      int   ← used to seed attention_mask
    loss_mask:          [1, T]      int
    hidden_states:      [1, T, 3D]  bf16
    last_hidden_states: [1, T, D]   bf16  (optional)

Output:
    input_ids:          [B, T_pad]
    attention_mask:     [B, T_pad]   1 on real tokens, 0 on padding
    loss_mask:          [B, T_pad]
    hidden_states:      [B, T_pad, 3D]
    last_hidden_states: [B, T_pad, D]   (only when present in all samples)
"""

from typing import Any, Dict, List, Optional

import torch

_DEFAULT_BUCKET = 256


def _pad_2d(t: torch.Tensor, n_target: int) -> torch.Tensor:
    """Right-pad or truncate a [B, T] tensor to [B, n_target]."""
    _, n = t.shape
    if n > n_target:
        return t[:, :n_target]
    if n == n_target:
        return t
    pad = torch.zeros(t.shape[0], n_target - n, dtype=t.dtype, device=t.device)
    return torch.cat((t, pad), dim=1)


def _pad_3d(t: torch.Tensor, n_target: int) -> torch.Tensor:
    """Right-pad or truncate a [B, T, D] tensor to [B, n_target, D]."""
    _, n, d = t.shape
    if n > n_target:
        return t[:, :n_target, :]
    if n == n_target:
        return t
    pad = torch.zeros(t.shape[0], n_target - n, d, dtype=t.dtype, device=t.device)
    return torch.cat((t, pad), dim=1)


class DataCollatorWithPadding:
    """Pad variable-seq-len samples into a rectangular batch.

    ``length_key`` is the field used to determine ``T`` per sample and to seed
    the auto-generated ``attention_mask`` (skipped if the caller already
    supplies ``attention_mask`` in the features).
    """

    def __init__(self, length_key: str = "input_ids", bucket: int = _DEFAULT_BUCKET):
        self.length_key = length_key
        self.bucket = bucket

    def __call__(
        self,
        features: List[Dict[str, Any]],
        bucket_size_override: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Pad samples to a uniform [B, T_pad].

        bucket_size_override: when provided, T_pad is at least this value
            (still rounded up to a bucket boundary). Used by the micro-batch
            loop in update_drafter to ensure all micro-batches in one macro-step
            share T_pad — torch.compile won't recompile across micro-batches.
        """
        max_length = max(item[self.length_key].shape[1] for item in features)
        if bucket_size_override is not None:
            max_length = max(max_length, int(bucket_size_override))
        max_length = ((max_length + self.bucket - 1) // self.bucket) * self.bucket

        common_keys = set(features[0].keys())
        for item in features[1:]:
            common_keys &= set(item.keys())
        # Drop keys whose value is None in any sample (matches the original
        # ``all(item.get("last_hidden_states") is not None ...)`` semantics).
        common_keys = {k for k in common_keys if all(item[k] is not None for item in features)}

        batch: Dict[str, torch.Tensor] = {}
        for key in common_keys:
            tensors = [item[key] for item in features]
            ndim = tensors[0].ndim
            if ndim == 2:
                batch[key] = torch.cat([_pad_2d(t, max_length) for t in tensors])
            elif ndim == 3:
                batch[key] = torch.cat([_pad_3d(t, max_length) for t in tensors])
            else:
                raise ValueError(
                    f"Cannot pad key {key!r}: expected ndim 2 or 3, got {ndim}"
                )

        if "attention_mask" not in batch:
            batch["attention_mask"] = torch.cat([
                _pad_2d(torch.ones_like(item[self.length_key]).long(), max_length)
                for item in features
            ])

        return batch
