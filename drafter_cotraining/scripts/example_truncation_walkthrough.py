"""Walk through how `data_preprocessing.build_input_ids_and_loss_mask`
tokenizes a multi-turn conversation, builds the per-turn assistant loss
mask, and applies right-truncation at varying `max_seq_length`.

Run:
    .venv/bin/python -m recipe.drafter_cotraining.scripts.example_truncation_walkthrough

Optional:
    --row-id regenerated_6262     # pick a specific row from the parquet
    --parquet ~/data/qwen3_8b_eagle3_10k/train.parquet
    --tokenizer Qwen/Qwen3-8B
    --caps 8192 2048 512 256 128 64 32   # max_seq_length values to demo

The output for each cap shows:
* the truncated input_ids length
* the surviving (start, end) assistant token spans
* the total supervised positions (loss_mask.sum())
* the surviving spans rendered as text snippets

This is a teaching script, not a production data-prep tool.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from recipe.drafter_cotraining.data_preprocessing import (  # noqa: E402
    build_input_ids_and_loss_mask,
    get_anchors,
    render_conversation,
    tokenize_with_assistant_mask,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", default="~/data/qwen3_8b_eagle3_10k/train.parquet")
    p.add_argument("--row-id", default=None, help="If set, pick this id from the parquet; else row 0.")
    p.add_argument("--tokenizer", default="Qwen/Qwen3-8B")
    p.add_argument("--chat-template", default="qwen")
    p.add_argument(
        "--caps",
        nargs="+",
        type=int,
        default=[8192, 2048, 512, 256, 128, 64, 32],
        help="max_seq_length values to demonstrate (descending = more interesting).",
    )
    return p.parse_args()


def find_spans(mask: torch.Tensor) -> list[tuple[int, int]]:
    """Return a list of (start, end) contiguous-1 spans in a 1D 0/1 tensor."""
    spans: list[tuple[int, int]] = []
    n = mask.shape[0]
    i = 0
    while i < n:
        if mask[i] == 1:
            j = i
            while j < n and mask[j] == 1:
                j += 1
            spans.append((i, j))
            i = j
        else:
            i += 1
    return spans


def show_synthetic_demo(tokenizer, chat_template: str, caps: Sequence[int]) -> None:
    """Tiny 4-turn synthetic conversation — easy to eyeball."""
    conv = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "It's 4."},
        {"role": "user", "content": "Why?"},
        {"role": "assistant", "content": "Addition."},
    ]
    print("=" * 80)
    print(" Synthetic 4-turn conversation (2 assistant turns)")
    print("=" * 80)

    anchors = get_anchors(chat_template)
    text = render_conversation(tokenizer, conv, anchors)
    print(f"Rendered text ({len(text)} chars):\n{text!r}\n")

    untruncated_ids, untruncated_mask = build_input_ids_and_loss_mask(
        tokenizer, conv, chat_template, max_length=10**9
    )
    full_T = untruncated_ids.shape[0]
    full_spans = find_spans(untruncated_mask)
    print(f"Untruncated: T={full_T}, spans={full_spans}, supervised={int(untruncated_mask.sum())}\n")

    print(f"{'cap':>6} | {'T':>4} | {'sum':>4} | {'#spans':>6} | spans")
    print("-" * 60)
    for cap in caps:
        ids, mask = tokenize_with_assistant_mask(tokenizer, text, anchors, max_length=cap)
        spans = find_spans(mask)
        T = ids.shape[0]
        print(f"{cap:>6} | {T:>4} | {int(mask.sum()):>4} | {len(spans):>6} | {spans}")
    print()


def show_real_row(
    parquet_path: str,
    row_id: str | None,
    tokenizer,
    chat_template: str,
    caps: Sequence[int],
) -> None:
    """Pull one real row from the converted parquet and rerun the whole flow."""
    parquet_path = str(Path(parquet_path).expanduser())
    print("=" * 80)
    print(f" Real row from {parquet_path}")
    print("=" * 80)

    df = pd.read_parquet(parquet_path)
    if row_id is not None:
        sub = df[df["id"].astype(str) == str(row_id)]
        if sub.empty:
            raise SystemExit(f"row id {row_id!r} not found in parquet")
        row = sub.iloc[0]
    else:
        row = df.iloc[0]

    conv = list(row["conversations"])
    print(f"id={row['id']!r}, turns={len(conv)}, "
          f"roles={[t['role'] for t in conv]}\n")

    anchors = get_anchors(chat_template)
    text = render_conversation(tokenizer, conv, anchors)
    full_ids, full_mask = build_input_ids_and_loss_mask(
        tokenizer, conv, chat_template, max_length=10**9
    )
    full_T = full_ids.shape[0]
    full_spans = find_spans(full_mask)
    print(f"Untruncated: T={full_T}, #spans={len(full_spans)}, "
          f"supervised={int(full_mask.sum())}\n")

    print(f"{'cap':>6} | {'T':>5} | {'sum':>5} | {'#spans':>6} | first-3 spans")
    print("-" * 80)
    for cap in caps:
        ids, mask = tokenize_with_assistant_mask(tokenizer, text, anchors, max_length=cap)
        spans = find_spans(mask)
        T = ids.shape[0]
        first_3 = spans[:3]
        suffix = " …" if len(spans) > 3 else ""
        print(
            f"{cap:>6} | {T:>5} | {int(mask.sum()):>5} | {len(spans):>6} | {first_3}{suffix}"
        )
    print()

    # Show one decoded supervised span at the smallest cap that still has spans.
    for cap in sorted(caps, reverse=True):
        ids, mask = tokenize_with_assistant_mask(tokenizer, text, anchors, max_length=cap)
        spans = find_spans(mask)
        if not spans:
            continue
        s, e = spans[0]
        snippet = tokenizer.decode(ids[s:e], skip_special_tokens=False)
        print(f"Decoded supervised tokens of span 0 at cap={cap}: {snippet!r}\n")
        break


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    show_synthetic_demo(tokenizer, args.chat_template, args.caps)
    show_real_row(
        parquet_path=args.parquet,
        row_id=args.row_id,
        tokenizer=tokenizer,
        chat_template=args.chat_template,
        caps=args.caps,
    )


if __name__ == "__main__":
    main()
