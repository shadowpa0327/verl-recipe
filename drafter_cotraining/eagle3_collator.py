"""Collator that pads variable-seq-len Eagle3 samples into a rectangular batch.

Ported from /root/TorchSpec/torchspec/data/utils.py (DataCollatorWithPadding).
Drops sp_degree (pinned to 1 — no sequence parallel on the drafter today).
Keeps the 256-token bucket so torch.compile / FlexAttention recompiles are
bounded if the draft model is ever compiled.

Per-sample input dict (built in update_drafter from EagleMooncakeStore.get):
    input_ids:          [1, T]      int
    hidden_states:      [1, T, 3D]  bf16
    last_hidden_states: [1, T, D]   bf16   (optional)
    loss_mask:          [1, T]      int

Output (matches FSDPDrafterEngine.prepare_model_inputs's contract):
    input_ids:          [B, T_pad]
    attention_mask:     [B, T_pad]   1 on real tokens, 0 on padding
    loss_mask:          [B, T_pad]
    hidden_states:      [B, T_pad, 3D]
    last_hidden_states: [B, T_pad, D]
"""

from typing import Any, Dict, List

import torch

_BUCKET = 256


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


class Eagle3Collator:
    """Pad variable-seq-len Eagle3 samples into a rectangular batch."""

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_length = max(item["input_ids"].shape[1] for item in features)
        max_length = ((max_length + _BUCKET - 1) // _BUCKET) * _BUCKET

        attention_masks = [torch.ones_like(item["input_ids"]).long() for item in features]

        batch: Dict[str, torch.Tensor] = {
            "input_ids": torch.cat(
                [_pad_2d(item["input_ids"], max_length) for item in features]
            ),
            "attention_mask": torch.cat(
                [_pad_2d(m, max_length) for m in attention_masks]
            ),
            "loss_mask": torch.cat(
                [_pad_2d(item["loss_mask"], max_length) for item in features]
            ),
            "hidden_states": torch.cat(
                [_pad_3d(item["hidden_states"], max_length) for item in features]
            ),
        }

        if all(item.get("last_hidden_states") is not None for item in features):
            batch["last_hidden_states"] = torch.cat(
                [_pad_3d(item["last_hidden_states"], max_length) for item in features]
            )

        return batch
