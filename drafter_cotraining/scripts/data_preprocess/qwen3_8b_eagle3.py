# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess Tengyunw/qwen3_8b_eagle3 to verl parquet.

The dataset repo contains ShareGPT-style conversation records:
    {"id": "...", "conversations": [{"from": "human", "value": "..."}, ...]}

This script reads the dataset repo with:
    load_dataset("Tengyunw/qwen3_8b_eagle3", split="train", streaming=True)

If the raw Hugging Face file is already cached locally, the script reads that
cache first to avoid repeated streaming setup/downloads.

It writes drafter pretrain parquet with explicit `prompt_messages` and
`response` columns so the trainer does not need to infer targets from a full
conversation.
"""

import argparse
import os
import random
from pathlib import Path
from typing import Any

import datasets
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from verl.utils.hdfs_io import copy, makedirs


DATA_SOURCE = "Tengyunw/qwen3_8b_eagle3"
DATA_SPLIT = "train"
DATA_FILENAME = "regenerated_complete.json"
WRITE_BATCH_SIZE = 1024
OUTPUT_SCHEMA = pa.schema(
    [
        pa.field(
            "prompt_messages",
            pa.list_(pa.struct([pa.field("role", pa.string()), pa.field("content", pa.string())])),
        ),
        pa.field("response", pa.string()),
        pa.field(
            "extra_info",
            pa.struct([pa.field("split", pa.string()), pa.field("index", pa.int64()), pa.field("id", pa.string())]),
        ),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="Deprecated. Use --local_save_dir.")
    parser.add_argument("--local_save_dir", default="~/data/qwen3_8b_eagle3")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.1,
        help="If < 1, fraction used for test. If >= 1, absolute test row count.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=-1, help="Absolute cap. -1 means no extra cap.")
    parser.add_argument(
        "--cache_path",
        default=None,
        help="Optional local regenerated_complete.json path. If omitted, the HF cache is used when present.",
    )
    return parser.parse_args()


def find_cached_data_path(cache_path: str | None) -> str | None:
    if cache_path is not None:
        cache_path = os.path.expanduser(cache_path)
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"--cache_path does not exist: {cache_path}")
        return cache_path

    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None

    cached_path = try_to_load_from_cache(DATA_SOURCE, DATA_FILENAME, repo_type="dataset")
    if isinstance(cached_path, str) and os.path.exists(cached_path):
        return cached_path
    return None


def iter_cached_rows(path: str):
    try:
        import ijson
    except ImportError as exc:
        raise ImportError("Reading cached JSON requires ijson. Install ijson or omit --cache_path.") from exc

    with open(path, "rb") as f:
        yield from ijson.items(f, "item", use_float=True)


def iter_hf_rows():
    yield from datasets.load_dataset(DATA_SOURCE, split=DATA_SPLIT, streaming=True)


def load_rows(max_samples: int, cache_path: str | None) -> tuple[list[dict[str, Any]], str]:
    cached_path = find_cached_data_path(cache_path)
    if cached_path is not None:
        source = iter_cached_rows(cached_path)
        source_name = cached_path
    else:
        source = iter_hf_rows()
        source_name = f"{DATA_SOURCE}:{DATA_SPLIT}"

    progress_total = max_samples if max_samples > 0 else None
    rows = []
    with tqdm(total=progress_total, desc="Loading rows", unit="row") as progress:
        for row in source:
            rows.append(dict(row))
            progress.update(1)
            if max_samples > 0 and len(rows) >= max_samples:
                break
    if not rows:
        raise ValueError(f"No rows loaded from {source_name}")
    return rows, source_name


def split_rows(
    rows: list[dict[str, Any]], test_size: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if test_size < 0:
        raise ValueError(f"--test_size must be non-negative, got {test_size}")

    rows = list(rows)
    rng = random.Random(seed)
    rng.shuffle(rows)

    if test_size == 0 or len(rows) == 1:
        return rows, []

    if test_size < 1:
        test_count = round(len(rows) * test_size)
    else:
        test_count = int(test_size)
    test_count = max(1, min(test_count, len(rows) - 1))

    return rows[test_count:], rows[:test_count]


def role_from_sharegpt(role: str) -> str:
    role_mapping = {
        "human": "user",
        "user": "user",
        "gpt": "assistant",
        "assistant": "assistant",
        "system": "system",
    }
    if role not in role_mapping:
        raise ValueError(f"Unsupported conversation role {role!r}.")
    return role_mapping[role]


def conversations_to_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    conversations = row.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        raise ValueError("Row must contain a non-empty `conversations` list.")

    messages = []
    for turn_index, turn in enumerate(conversations):
        if not isinstance(turn, dict):
            raise ValueError(f"Conversation turn {turn_index} is {type(turn).__name__}, expected object.")
        role = turn.get("from")
        content = turn.get("value")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError(f"Conversation turn {turn_index} must contain string `from` and `value`.")
        messages.append({"role": role_from_sharegpt(role), "content": content})
    return messages


def split_prompt_response(messages: list[dict[str, str]]) -> tuple[list[dict[str, str]], str]:
    last_assistant_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "assistant":
            last_assistant_idx = i
            break
    if last_assistant_idx is None:
        raise ValueError("Conversation has no assistant response.")
    prompt_messages = messages[:last_assistant_idx]
    response = messages[last_assistant_idx]["content"]
    if not prompt_messages:
        raise ValueError("Conversation has no prompt turns before the response.")
    return prompt_messages, response


def convert_row(row: dict[str, Any], split: str, idx: int) -> dict[str, Any]:
    extra_info = {
        "split": split,
        "index": idx,
    }
    if "id" in row:
        extra_info["id"] = row["id"]
    else:
        extra_info["id"] = None
    prompt_messages, response = split_prompt_response(conversations_to_messages(row))
    return {
        "prompt_messages": prompt_messages,
        "response": response,
        "extra_info": extra_info,
    }


def flush_batch(writer: pq.ParquetWriter | None, rows: list[dict[str, Any]], path: str) -> pq.ParquetWriter:
    table = pa.Table.from_pylist(rows, schema=OUTPUT_SCHEMA)
    if writer is None:
        writer = pq.ParquetWriter(path, table.schema, compression="zstd")
    writer.write_table(table)
    rows.clear()
    return writer


def write_split(rows: list[dict[str, Any]], split: str, path: str) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        pq.write_table(pa.Table.from_pylist([], schema=OUTPUT_SCHEMA), path, compression="zstd")
        return 0

    writer = None
    batch = []
    for idx, row in enumerate(tqdm(rows, desc=f"Writing {split}", unit="row")):
        batch.append(convert_row(row, split, idx))
        if len(batch) >= WRITE_BATCH_SIZE:
            writer = flush_batch(writer, batch, path)

    if batch:
        writer = flush_batch(writer, batch, path)
    if writer is not None:
        writer.close()
    return len(rows)


def main() -> None:
    args = parse_args()

    raw_rows, source_name = load_rows(args.max_samples, args.cache_path)
    train_rows, test_rows = split_rows(raw_rows, args.test_size, args.seed)

    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    local_save_dir = os.path.expanduser(local_save_dir)
    train_count = write_split(train_rows, "train", os.path.join(local_save_dir, "train.parquet"))
    test_count = write_split(test_rows, "test", os.path.join(local_save_dir, "test.parquet"))

    print(f"Loaded {len(raw_rows)} rows from {source_name}")
    print(f"Wrote {train_count} train rows to {os.path.join(local_save_dir, 'train.parquet')}")
    print(f"Wrote {test_count} test rows to {os.path.join(local_save_dir, 'test.parquet')}")

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_save_dir, dst=args.hdfs_dir)


if __name__ == "__main__":
    main()
