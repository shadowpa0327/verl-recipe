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
Preprocess ultrachat_200k_tenyun_regenerate data to verl parquet format.

Input format (ShareGPT):
    id: string
    conversations: list<{"from": string, "value": string}>

Output format (canonical):
    id: string
    messages: list<{"role": string, "content": string}>
    extra_info: {"split": string, "index": int, "id": string}

Legacy output format (--output_format=legacy):
    prompt_messages: list<{"role": string, "content": string}>
    response: string
    extra_info: {"split": string, "index": int, "id": string}

The canonical format preserves the full conversation for multi-turn loss mask
computation, which is required for proper drafter pretraining that supervises
all assistant turns, not just the last response.
"""

import argparse
import os
from pathlib import Path
from typing import Any, List

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


WRITE_BATCH_SIZE = 1024

# Canonical schema: full conversation for multi-turn loss mask
OUTPUT_SCHEMA_CANONICAL = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field(
            "messages",
            pa.list_(pa.struct([pa.field("role", pa.string()), pa.field("content", pa.string())])),
        ),
        pa.field(
            "extra_info",
            pa.struct([pa.field("split", pa.string()), pa.field("index", pa.int64()), pa.field("id", pa.string())]),
        ),
    ]
)

# Legacy schema: prompt_messages + response split (for backward compatibility)
OUTPUT_SCHEMA_LEGACY = pa.schema(
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
    parser.add_argument("--input_dir", required=True, help="Directory containing parquet files")
    parser.add_argument("--output_dir", required=True, help="Output directory for verl format parquet files")
    parser.add_argument("--max_train_samples", type=int, default=-1, help="Max train samples. -1 for no limit.")
    parser.add_argument("--max_test_samples", type=int, default=-1, help="Max test samples. -1 for no limit.")
    parser.add_argument("--test_size", type=float, default=0.1, help="Test split ratio (0-1). Used when max_test_samples not set.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    parser.add_argument(
        "--output_format",
        choices=["canonical", "legacy"],
        default="canonical",
        help="Output format: 'canonical' (full messages) or 'legacy' (prompt_messages + response). "
        "Canonical is recommended for multi-turn loss mask support.",
    )
    return parser.parse_args()


def role_from_sharegpt(role: str) -> str:
    """Convert ShareGPT role names to standard role names."""
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


def conversations_to_messages(conversations: list[dict[str, str]]) -> list[dict[str, str]]:
    """Convert ShareGPT conversations to standard messages format."""
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


def split_prompt_response(messages: List[dict[str, str]]) -> tuple[List[dict[str, str]], str]:
    """Split messages into prompt_messages and last assistant response."""
    if not messages:
        raise ValueError("Messages list is empty")

    # Find last assistant message
    last_assistant_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "assistant":
            last_assistant_idx = i
            break

    if last_assistant_idx is None:
        raise ValueError("Conversation has no assistant response")

    prompt_messages = messages[:last_assistant_idx]
    response = messages[last_assistant_idx]["content"]

    if not prompt_messages:
        raise ValueError("Conversation has no prompt turns before the response")

    return prompt_messages, response


def convert_row_canonical(row: dict[str, Any], split: str, idx: int) -> dict[str, Any]:
    """Convert a single row to canonical format (full messages)."""
    conversations = row.get("conversations", [])
    messages = conversations_to_messages(conversations)

    return {
        "id": row.get("id", f"sample_{idx}"),
        "messages": messages,
        "extra_info": {
            "split": split,
            "index": idx,
            "id": row.get("id", f"sample_{idx}"),
        },
    }


def convert_row_legacy(row: dict[str, Any], split: str, idx: int) -> dict[str, Any]:
    """Convert a single row to legacy format (prompt_messages + response)."""
    conversations = row.get("conversations", [])
    messages = conversations_to_messages(conversations)
    prompt_messages, response = split_prompt_response(messages)

    return {
        "prompt_messages": prompt_messages,
        "response": response,
        "extra_info": {
            "split": split,
            "index": idx,
            "id": row.get("id", ""),
        },
    }


def process_files(input_files: list[str], output_path: str, max_samples: int) -> list[dict[str, Any]]:
    """Process all parquet files and return converted rows."""
    all_rows = []
    global_idx = 0

    for input_file in input_files:
        print(f"Reading {input_file}...")
        table = pq.read_table(input_file)

        for row_idx in tqdm(range(table.num_rows), desc="Loading rows", unit="row"):
            row_dict = {col: table[col][row_idx].as_py() for col in table.column_names}

            try:
                converted = convert_row(row_dict, "all", global_idx)
                all_rows.append(converted)
                global_idx += 1

                if max_samples > 0 and len(all_rows) >= max_samples:
                    break

            except ValueError:
                # Skip invalid rows
                continue

        if max_samples > 0 and len(all_rows) >= max_samples:
            break

    return all_rows


def split_rows(rows: list[dict[str, Any]], test_size: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split rows into train and test sets."""
    import random

    if test_size <= 0 or len(rows) == 1:
        return rows, []

    rows = list(rows)
    rng = random.Random(seed)
    rng.shuffle(rows)

    if test_size < 1:
        test_count = round(len(rows) * test_size)
    else:
        test_count = int(test_size)

    test_count = max(1, min(test_count, len(rows) - 1))
    return rows[test_count:], rows[:test_count]


def write_parquet(
    rows: List[dict[str, Any]],
    output_path: str,
    split: str,
    output_format: str = "canonical",
) -> int:
    """Write rows to parquet file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Select schema and converter based on output format
    if output_format == "canonical":
        schema = OUTPUT_SCHEMA_CANONICAL
        convert_fn = convert_row_canonical
    else:
        schema = OUTPUT_SCHEMA_LEGACY
        convert_fn = convert_row_legacy

    writer = None
    batch = []

    for idx, row in enumerate(tqdm(rows, desc=f"Writing {split}", unit="row")):
        try:
            converted = convert_fn(row, split, idx)
        except ValueError:
            # Skip invalid rows
            continue
        batch.append(converted)
        if len(batch) >= WRITE_BATCH_SIZE:
            table_batch = pa.Table.from_pylist(batch, schema=schema)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table_batch.schema, compression="zstd")
            writer.write_table(table_batch)
            batch.clear()

    if batch:
        table_batch = pa.Table.from_pylist(batch, schema=schema)
        if writer is None:
            writer = pq.ParquetWriter(output_path, table_batch.schema, compression="zstd")
        writer.write_table(table_batch)

    if writer is not None:
        writer.close()

    return len(rows)


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all parquet files
    input_files = sorted(input_dir.glob("*.parquet"))

    if not input_files:
        raise ValueError(f"No parquet files found in {input_dir}")

    print(f"Found {len(input_files)} parquet files")
    print(f"Output format: {args.output_format}")

    # Determine how many samples to load
    if args.max_train_samples > 0 and args.max_test_samples > 0:
        max_samples = args.max_train_samples + args.max_test_samples
    elif args.max_train_samples > 0:
        max_samples = args.max_train_samples
        if args.test_size > 0:
            max_samples = int(args.max_train_samples / (1 - args.test_size))
    elif args.max_test_samples > 0:
        max_samples = args.max_test_samples
        if args.test_size > 0:
            max_samples = int(args.max_test_samples / args.test_size)
    else:
        max_samples = -1

    # Process all files
    all_rows = process_files([str(f) for f in input_files], str(output_dir / "temp.parquet"), max_samples)
    print(f"Loaded {len(all_rows)} rows")

    # Split into train/test
    if args.max_train_samples > 0 and args.max_test_samples > 0:
        # Use specified counts
        train_rows = all_rows[:args.max_train_samples]
        test_rows = all_rows[args.max_train_samples:args.max_train_samples + args.max_test_samples]
    else:
        # Use ratio-based split
        train_rows, test_rows = split_rows(all_rows, args.test_size, args.seed)

    # Write train split
    train_count = write_parquet(
        train_rows,
        str(output_dir / "train.parquet"),
        "train",
        output_format=args.output_format,
    )
    print(f"Wrote {train_count} train rows to {output_dir / 'train.parquet'}")

    # Write test split
    if test_rows:
        test_count = write_parquet(
            test_rows,
            str(output_dir / "test.parquet"),
            "test",
            output_format=args.output_format,
        )
        print(f"Wrote {test_count} test rows to {output_dir / 'test.parquet'}")


if __name__ == "__main__":
    main()
