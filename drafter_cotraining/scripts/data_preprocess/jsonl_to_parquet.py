# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Convert canonical conversation .jsonl to parquet for drafter pretraining.

Canonical raw row (one JSON object per line):

    {"id": "any_string", "conversations": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."},
        ...
    ]}

ShareGPT shape ({from, value} with human/gpt) is auto-mapped to canonical
form. The output parquet preserves the full multi-turn conversation, so the
trainer can build a per-turn assistant loss mask. No prompt/response split is
performed — that happens (lossily) at trainer time.

Output schema:
    id:            string
    conversations: list<struct<role: string, content: string>>
    extra_info:    struct<split: string, index: int64>
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

WRITE_BATCH_SIZE = 1024

OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field(
            "conversations",
            pa.list_(
                pa.struct(
                    [pa.field("role", pa.string()), pa.field("content", pa.string())]
                )
            ),
        ),
        pa.field(
            "extra_info",
            pa.struct(
                [pa.field("split", pa.string()), pa.field("index", pa.int64())]
            ),
        ),
    ]
)

_ROLE_MAPPING = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "system": "system",
    "tool": "tool",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="Path to canonical .jsonl (one JSON object per line).",
    )
    parser.add_argument(
        "--local_save_dir",
        required=True,
        help="Output directory; train.parquet / test.parquet are written here.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.1,
        help="If <1, fraction held out for test. If >=1, absolute test row count.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="Absolute cap on number of rows read. -1 means no cap.",
    )
    return parser.parse_args()


def _normalize_conversation(conv: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Map ShareGPT ``{from, value}`` to canonical ``{role, content}``."""
    if not conv:
        raise ValueError("Conversation is empty.")
    out = []
    for turn in conv:
        if not isinstance(turn, dict):
            raise ValueError(f"Conversation turn is not a dict: {turn!r}")
        if "role" in turn and "content" in turn:
            role = str(turn["role"])
            content = turn["content"]
        elif "from" in turn and "value" in turn:
            role = str(turn["from"])
            content = turn["value"]
        else:
            raise ValueError(f"Unrecognized conversation turn keys: {list(turn.keys())}")
        if role not in _ROLE_MAPPING:
            raise ValueError(f"Unsupported role {role!r}")
        if not isinstance(content, str):
            raise ValueError(f"Turn content must be string, got {type(content).__name__}")
        out.append({"role": _ROLE_MAPPING[role], "content": content})
    return out


def _iter_rows(path: str, max_samples: int) -> Iterable[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
            if max_samples > 0 and (i + 1) >= max_samples:
                break


def _load_rows(path: str, max_samples: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    progress_total = max_samples if max_samples > 0 else None
    with tqdm(total=progress_total, desc="Reading", unit="row") as bar:
        for row in _iter_rows(path, max_samples):
            rows.append(row)
            bar.update(1)
    if not rows:
        raise ValueError(f"No rows loaded from {path}")
    return rows


def _split(rows: list[dict[str, Any]], test_size: float, seed: int):
    rows = list(rows)
    rng = random.Random(seed)
    rng.shuffle(rows)
    if test_size <= 0 or len(rows) == 1:
        return rows, []
    if test_size < 1:
        test_count = round(len(rows) * test_size)
    else:
        test_count = int(test_size)
    test_count = max(1, min(test_count, len(rows) - 1))
    return rows[test_count:], rows[:test_count]


def _convert_row(row: dict[str, Any], split: str, idx: int) -> dict[str, Any]:
    raw_id = row.get("id")
    row_id = str(raw_id) if raw_id is not None else f"{split}_{idx}"
    conv = _normalize_conversation(row.get("conversations") or [])
    return {
        "id": row_id,
        "conversations": conv,
        "extra_info": {"split": split, "index": idx},
    }


def _write(rows: list[dict[str, Any]], split: str, path: str) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        pq.write_table(pa.Table.from_pylist([], schema=OUTPUT_SCHEMA), path, compression="zstd")
        return 0
    writer: pq.ParquetWriter | None = None
    batch: list[dict[str, Any]] = []
    written = 0
    try:
        for idx, row in enumerate(tqdm(rows, desc=f"Writing {split}", unit="row")):
            try:
                batch.append(_convert_row(row, split, idx))
            except ValueError as exc:
                tqdm.write(f"[skip {split}#{idx}] {exc}")
                continue
            if len(batch) >= WRITE_BATCH_SIZE:
                table = pa.Table.from_pylist(batch, schema=OUTPUT_SCHEMA)
                if writer is None:
                    writer = pq.ParquetWriter(path, table.schema, compression="zstd")
                writer.write_table(table)
                written += len(batch)
                batch.clear()
        if batch:
            table = pa.Table.from_pylist(batch, schema=OUTPUT_SCHEMA)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema, compression="zstd")
            writer.write_table(table)
            written += len(batch)
    finally:
        if writer is not None:
            writer.close()
    return written


def main() -> None:
    args = parse_args()
    local_save_dir = os.path.expanduser(args.local_save_dir)

    raw_rows = _load_rows(os.path.expanduser(args.input), args.max_samples)
    train_rows, test_rows = _split(raw_rows, args.test_size, args.seed)

    train_path = os.path.join(local_save_dir, "train.parquet")
    test_path = os.path.join(local_save_dir, "test.parquet")
    n_train = _write(train_rows, "train", train_path)
    n_test = _write(test_rows, "test", test_path)

    print(f"Loaded {len(raw_rows)} rows from {args.input}")
    print(f"Wrote {n_train} train rows to {train_path}")
    print(f"Wrote {n_test} test rows to {test_path}")


if __name__ == "__main__":
    main()
