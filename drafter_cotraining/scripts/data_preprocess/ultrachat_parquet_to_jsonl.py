# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Convert sharded ultrachat-style parquet (e.g. shadowpa0327/qwen3_8b_eagle3-parquet,
the sharded form of Tengyunw/qwen3_8b_eagle3) into canonical conversation JSONL.

Upstream rows use the ShareGPT-style ``{from, value}`` schema; this script
maps them to the canonical ``{role, content}`` schema consumed by
``recipe.drafter_cotraining.scripts.data_preprocess.jsonl_to_parquet``.

Input row schema (per shard):
    id:            string
    conversations: list<struct<from: string, value: string>>

Output row schema (one JSON object per line):
    id:            string
    conversations: list<{role: str, content: str}>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Optional

ROLE_MAPPING = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "chatgpt": "assistant",
    "assistant": "assistant",
    "bing": "assistant",
    "system": "system",
}


def import_parquet():
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit(
            "pyarrow is required. `pip install pyarrow`."
        ) from exc
    return pq


def normalize_message(message: dict[str, Any]) -> Optional[dict[str, str]]:
    if not isinstance(message, dict):
        return None
    if "role" in message and "content" in message:
        role, content = message["role"], message["content"]
    elif "from" in message and "value" in message:
        role = ROLE_MAPPING.get(message["from"], message["from"])
        content = message["value"]
    else:
        return None
    if not role:
        return None
    if content is None:
        content = ""
    elif not isinstance(content, str):
        content = str(content)
    return {"role": str(role), "content": content}


def build_record(row: dict[str, Any], fallback_id: str) -> Optional[dict[str, Any]]:
    raw = row.get("conversations")
    if not isinstance(raw, list) or not raw:
        return None
    conversations = []
    for msg in raw:
        norm = normalize_message(msg)
        if norm is None:
            return None
        conversations.append(norm)
    if not any(m["role"] == "assistant" and m["content"] for m in conversations):
        return None
    data_id = row.get("id") or fallback_id
    return {"id": str(data_id), "conversations": conversations}


def iter_shards(input_dir: Path, pattern: str) -> list[Path]:
    shards = sorted(input_dir.glob(pattern))
    if not shards:
        raise FileNotFoundError(f"No parquet shards matched {input_dir}/{pattern}")
    return shards


def iter_rows(shards: Iterable[Path], batch_size: int) -> Iterable[dict[str, Any]]:
    pq = import_parquet()
    columns = ["id", "conversations"]
    for shard in shards:
        pf = pq.ParquetFile(shard)
        for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
            yield from batch.to_pylist()


def convert(
    input_dir: Path,
    output_path: Path,
    *,
    pattern: str,
    limit: Optional[int],
    batch_size: int,
) -> dict[str, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shards = iter_shards(input_dir, pattern)

    seen = written = skipped = 0
    with output_path.open("w", encoding="utf-8") as out:
        for row in iter_rows(shards, batch_size):
            if limit is not None and limit > 0 and written >= limit:
                break
            fallback_id = f"ultrachat_{seen:08d}"
            record = build_record(row, fallback_id)
            seen += 1
            if record is None:
                skipped += 1
                continue
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    return {"seen": seen, "written": written, "skipped": skipped}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing sharded ultrachat parquet files.",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output canonical JSONL path.",
    )
    p.add_argument(
        "--pattern",
        default="train-*.parquet",
        help="Glob for parquet shards inside --input-dir.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Max output rows. -1 means no cap.",
    )
    p.add_argument("--batch-size", type=int, default=1024)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    limit = args.limit if args.limit and args.limit > 0 else None
    stats = convert(
        args.input_dir,
        args.output,
        pattern=args.pattern,
        limit=limit,
        batch_size=args.batch_size,
    )
    print(
        f"read {stats['seen']} rows, wrote {stats['written']} records to {args.output} "
        f"(skipped {stats['skipped']})"
    )


if __name__ == "__main__":
    main()
