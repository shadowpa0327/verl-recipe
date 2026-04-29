"""Draft-vocabulary pruning support for the EAGLE3 drafter.

Ported from SpecForge ``specforge/data/preprocessing.py`` (functions
``generate_vocab_mapping_file`` and ``process_token_dict_to_mappings``).
The on-disk format and the d2t/t2d shape contract match SpecForge, so a
mapping built here loads via :meth:`Eagle3DraftModel.set_vocab_buffers`
without modification.

Workflow:

1. Iterate the training corpus (already tokenized + assistant-loss-masked)
   and count token frequencies on supervised positions only.
2. Take the top ``draft_vocab_size`` tokens by frequency. Pad with unused
   ids if the corpus has fewer distinct tokens than ``draft_vocab_size``.
3. Emit:
     ``t2d``: bool tensor of shape ``[V_target]`` — True iff token id is
              in the pruned draft vocabulary.
     ``d2t``: int64 tensor of shape ``[V_draft]`` — for sorted draft
              index ``i``, ``d2t[i] + i`` is the corresponding target id.

The training loss kernel slices target logits by ``t2d`` (eagle3_model.py
``compute_target_p_padded``) and zeroes positions whose verifier-argmax
falls outside the draft vocabulary.
"""

from __future__ import annotations

import os
from collections import Counter
from typing import Iterable, Tuple

import torch
from tqdm import tqdm


def process_token_dict_to_mappings(
    token_dict: Counter,
    draft_vocab_size: int,
    target_vocab_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build (d2t, t2d) tensors from a token-frequency Counter.

    Args:
        token_dict: Counter mapping target-vocab token id → frequency.
        draft_vocab_size: Size of the pruned draft vocabulary.
        target_vocab_size: Size of the target (full) vocabulary.

    Returns:
        d2t: int64 tensor [V_draft] — d2t[i] = used_tokens[i] - i, where
            used_tokens is the sorted-ascending list of selected ids.
        t2d: bool tensor [V_target] — t2d[id] = (id in used_tokens).
    """
    if len(token_dict) < draft_vocab_size:
        existing_tokens = set(token_dict.keys())
        for token in range(target_vocab_size):
            if token in existing_tokens:
                continue
            token_dict[token] = 0
            if len(token_dict) >= draft_vocab_size:
                break

    total_frequency = sum(token_dict.values())
    top_n = token_dict.most_common(draft_vocab_size)
    top_n_frequency_sum = sum(freq for _, freq in top_n)

    if total_frequency == 0:
        top_n_ratio = 0.0
    else:
        top_n_ratio = top_n_frequency_sum / total_frequency
    print(f"top {draft_vocab_size} token frequency ratio: {top_n_ratio:.2%}")

    used_tokens = sorted(key for key, _ in top_n)
    used_set = set(used_tokens)

    d2t = torch.tensor(
        [used_tokens[i] - i for i in range(len(used_tokens))], dtype=torch.int64
    )
    t2d = torch.tensor(
        [i in used_set for i in range(target_vocab_size)], dtype=torch.bool
    )
    return d2t, t2d


def generate_vocab_mapping_file(
    samples: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    target_vocab_size: int,
    draft_vocab_size: int,
    output_path: str,
    total: int | None = None,
) -> str:
    """Build a t2d/d2t mapping from supervised tokens and save to ``output_path``.

    Args:
        samples: iterable yielding ``(input_ids, loss_mask)`` int tensors of
            shape ``[T]``. Only positions where ``loss_mask == 1`` count.
        target_vocab_size: target model's full vocabulary size.
        draft_vocab_size: pruned draft vocabulary size (< target_vocab_size).
        output_path: path to a ``.pt`` file to write. The same file path is
            consumed by ``FSDPDrafterEngine`` at training time.
        total: optional progress-bar hint (number of samples).

    Returns:
        ``output_path`` (for chaining).
    """
    if draft_vocab_size > target_vocab_size:
        raise ValueError(
            f"draft_vocab_size ({draft_vocab_size}) must be <= "
            f"target_vocab_size ({target_vocab_size})"
        )

    token_dict: Counter = Counter()
    for input_ids, loss_mask in tqdm(
        samples, total=total, desc="Counting tokens for vocab mapping"
    ):
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.as_tensor(input_ids)
        if not isinstance(loss_mask, torch.Tensor):
            loss_mask = torch.as_tensor(loss_mask)
        masked_ids = input_ids[loss_mask == 1]
        if masked_ids.numel() == 0:
            continue
        unique_ids, counts = masked_ids.unique(return_counts=True)
        token_dict.update(dict(zip(unique_ids.tolist(), counts.tolist())))

    d2t, t2d = process_token_dict_to_mappings(
        token_dict, draft_vocab_size, target_vocab_size
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save({"d2t": d2t, "t2d": t2d}, output_path)
    print(f"Saved vocab mapping to: {output_path}")
    return output_path


def load_vocab_mapping(path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load (d2t, t2d) tensors from a ``.pt`` file written by this module."""
    mapping = torch.load(path, map_location="cpu", weights_only=True)
    return mapping["d2t"], mapping["t2d"]
