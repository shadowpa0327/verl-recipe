# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Build a draft-vocab pruning mapping from a canonical pretrain parquet.

Reads parquet files written by ``jsonl_to_parquet.py`` (schema: ``id`` +
``conversations`` list<{role, content}>), tokenizes each row with the same
chat template and assistant-loss-mask logic the trainer uses, counts
supervised tokens, and writes a ``.pt`` file with ``{t2d, d2t}`` tensors
ready for ``FSDPDrafterEngine`` to consume via
``model_config.vocab_mapping_path``.

Example::

    python -m recipe.drafter_cotraining.scripts.data_preprocess.build_vocab_mapping \
        --data_files ~/data/qwen3_8b_eagle3/train.parquet \
        --tokenizer Qwen/Qwen3-8B \
        --chat_template qwen \
        --max_length 4096 \
        --target_vocab_size 151936 \
        --draft_vocab_size 32000 \
        --output ~/cache/vocab_mapping/qwen3_8b_32k.pt

The output file is the value to set as
``actor_rollout_ref.drafter.model_config.vocab_mapping_path`` in the recipe
yaml. Make sure the corresponding draft architecture template (
``model_config.local_path``) declares ``draft_vocab_size`` matching
``--draft_vocab_size`` here, or set it directly on the auto-derived config.
"""

from __future__ import annotations

import argparse
import os
from typing import Iterable

import pandas as pd

from recipe.drafter_cotraining.data_preprocessing import build_input_ids_and_loss_mask
from recipe.drafter_cotraining.vocab_mapping import generate_vocab_mapping_file
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.py_functional import convert_nested_value_to_list_recursive


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_files",
        nargs="+",
        required=True,
        help="One or more canonical pretrain parquet files (id + conversations).",
    )
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="HF tokenizer path or repo (must match the target model).",
    )
    parser.add_argument(
        "--chat_template",
        required=True,
        help="Short chat-template name registered in data_preprocessing.get_anchors (e.g. qwen, llama3).",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=4096,
        help="Truncation length used during tokenization. Should match training max_length.",
    )
    parser.add_argument(
        "--target_vocab_size",
        type=int,
        default=None,
        help="Target model vocab size. Defaults to tokenizer.vocab_size, but pass explicitly when the model config's vocab_size differs (e.g. padded vocab).",
    )
    parser.add_argument(
        "--draft_vocab_size",
        type=int,
        required=True,
        help="Draft (pruned) vocab size. Must be < target_vocab_size to actually prune.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output .pt path. Created (or overwritten); parent dirs are made automatically.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="Cap rows scanned across all files. -1 means scan everything.",
    )
    parser.add_argument(
        "--conversations_key",
        default="conversations",
        help="Column name holding the canonical conversation list.",
    )
    return parser.parse_args()


def iter_conversations(
    data_files: list[str],
    conversations_key: str,
    max_samples: int,
) -> Iterable[list[dict[str, str]]]:
    seen = 0
    for data_file in data_files:
        local_path = copy_local_path_from_hdfs(data_file, verbose=True)
        df = pd.read_parquet(local_path, dtype_backend="pyarrow")
        if conversations_key not in df.columns:
            raise ValueError(
                f"{local_path} is missing required column {conversations_key!r}"
            )
        for _, row in df.iterrows():
            conv = convert_nested_value_to_list_recursive(row[conversations_key])
            if not isinstance(conv, list) or not conv:
                continue
            yield conv
            seen += 1
            if max_samples > 0 and seen >= max_samples:
                return


def main() -> None:
    args = parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    target_vocab_size = args.target_vocab_size or int(tokenizer.vocab_size)
    if args.draft_vocab_size > target_vocab_size:
        raise ValueError(
            f"--draft_vocab_size ({args.draft_vocab_size}) must be <= "
            f"target_vocab_size ({target_vocab_size})"
        )

    def _samples():
        for conv in iter_conversations(
            args.data_files, args.conversations_key, args.max_samples
        ):
            try:
                ids, mask = build_input_ids_and_loss_mask(
                    tokenizer, conv, args.chat_template, args.max_length
                )
            except Exception:  # noqa: BLE001
                continue
            if ids.numel() == 0 or int(mask.sum()) == 0:
                continue
            yield ids, mask

    output_path = os.path.expanduser(args.output)
    generate_vocab_mapping_file(
        samples=_samples(),
        target_vocab_size=target_vocab_size,
        draft_vocab_size=args.draft_vocab_size,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
