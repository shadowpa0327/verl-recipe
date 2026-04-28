#!/usr/bin/env python3
"""CPU diagnostic script for loss mask validation.

Validates loss mask computation on sample data from parquet files.
Reports masked spans and valid token counts.

Usage:
    python diag_loss_mask.py \
        --input_dir /path/to/data \
        --model_path Qwen/Qwen3-8B \
        --num_samples 10 \
        --loss_mask_mode all_assistant_turns
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import torch


def load_sample_data(input_dir: str, num_samples: int) -> List[dict]:
    """Load sample data from parquet files."""
    import pyarrow.parquet as pq

    input_path = Path(input_dir)
    parquet_files = list(input_path.glob("*.parquet"))

    if not parquet_files:
        print(f"ERROR: No parquet files found in {input_dir}")
        sys.exit(1)

    samples = []
    for pq_file in parquet_files:
        table = pq.read_table(pq_file)
        for i in range(min(num_samples - len(samples), table.num_rows)):
            row = table.slice(i, 1).to_pydict()
            # Convert single-element lists to values
            sample = {}
            for key, val in row.items():
                sample[key] = val[0] if val else None
            samples.append(sample)
            if len(samples) >= num_samples:
                break
        if len(samples) >= num_samples:
            break

    return samples


def get_tokenizer(model_path: str):
    """Load tokenizer from model path."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True,
    )
    return tokenizer


def format_messages_for_display(messages: List[dict]) -> str:
    """Format messages for display."""
    lines = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        # Truncate long content
        if len(content) > 100:
            content = content[:97] + "..."
        lines.append(f"  [{role}]: {content}")
    return "\n".join(lines)


def decode_masked_spans(
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    tokenizer,
) -> List[dict]:
    """Decode spans where loss_mask == 1."""
    spans = []
    in_span = False
    span_start = 0

    for i in range(len(loss_mask)):
        if loss_mask[i] == 1 and not in_span:
            in_span = True
            span_start = i
        elif loss_mask[i] == 0 and in_span:
            in_span = False
            span_tokens = input_ids[span_start:i]
            spans.append({
                "start": span_start,
                "end": i,
                "length": i - span_start,
                "text": tokenizer.decode(span_tokens, skip_special_tokens=False),
            })

    # Handle span that extends to end
    if in_span:
        span_tokens = input_ids[span_start:]
        spans.append({
            "start": span_start,
            "end": len(loss_mask),
            "length": len(loss_mask) - span_start,
            "text": tokenizer.decode(span_tokens, skip_special_tokens=False),
        })

    return spans


def validate_sample(
    sample: dict,
    tokenizer,
    loss_mask_mode: str,
    verbose: bool = True,
) -> dict:
    """Validate loss mask for a single sample."""
    from recipe.drafter_cotraining.loss_mask_utils import (
        QWEN3_ASSISTANT_HEADER,
        QWEN3_END_OF_TURN,
        build_loss_mask_for_conversation,
        has_thinking_content,
    )

    result = {
        "sample_id": sample.get("id", "unknown"),
        "valid": True,
        "errors": [],
        "valid_tokens": 0,
        "total_tokens": 0,
        "spans": [],
    }

    # Extract messages (try multiple column names)
    messages = sample.get("messages") or sample.get("conversations")
    if messages is None:
        # Try legacy format
        prompt_messages = sample.get("prompt_messages", [])
        response = sample.get("response", "")
        if prompt_messages and response:
            messages = list(prompt_messages)
            messages.append({"role": "assistant", "content": response})
        else:
            result["valid"] = False
            result["errors"].append("No messages or conversations found")
            return result

    # Normalize messages format (conversations may use 'from' instead of 'role')
    normalized_messages = []
    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role") or msg.get("from", "unknown")
            content = msg.get("content") or msg.get("value", "")
            # Normalize role names
            if role in ("human", "user"):
                role = "user"
            elif role in ("gpt", "assistant", "bot"):
                role = "assistant"
            elif role in ("system",):
                role = "system"
            normalized_messages.append({"role": role, "content": content})
        else:
            result["valid"] = False
            result["errors"].append(f"Invalid message format: {type(msg)}")
            return result

    if not normalized_messages:
        result["valid"] = False
        result["errors"].append("Empty messages after normalization")
        return result

    messages = normalized_messages

    if verbose:
        print(f"\n{'='*60}")
        print(f"Sample ID: {result['sample_id']}")
        print(f"Messages ({len(messages)} turns):")
        for i, msg in enumerate(messages[:5]):  # Show first 5 turns
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if len(content) > 80:
                content = content[:77] + "..."
            print(f"  [{i}] {role}: {content}")
        if len(messages) > 5:
            print(f"  ... and {len(messages) - 5} more turns")
        print(f"Has thinking content: {has_thinking_content(messages)}")

    # Build loss mask using the high-level API
    try:
        mask_result = build_loss_mask_for_conversation(
            tokenizer=tokenizer,
            messages=messages,
            max_length=4096,
            loss_mask_mode=loss_mask_mode,
        )
    except Exception as e:
        result["valid"] = False
        result["errors"].append(f"Failed to build loss mask: {e}")
        return result

    if mask_result is None:
        result["valid"] = False
        result["errors"].append("Loss mask returned None (no supervised tokens)")
        return result

    input_ids = mask_result.input_ids
    loss_mask = mask_result.loss_mask
    result["total_tokens"] = len(input_ids)
    result["valid_tokens"] = mask_result.valid_tokens

    if result["valid_tokens"] == 0:
        result["valid"] = False
        result["errors"].append("Loss mask has no supervised tokens")

    # Decode masked spans for reporting
    spans = decode_masked_spans(input_ids, loss_mask, tokenizer)
    result["spans"] = spans

    if verbose:
        print(f"\nTotal tokens: {result['total_tokens']}")
        print(f"Valid tokens: {result['valid_tokens']}")
        print(f"\nMasked spans ({len(spans)} total):")
        for i, span in enumerate(spans):
            text = span["text"]
            if len(text) > 80:
                text = text[:77] + "..."
            print(f"  Span {i+1}: pos {span['start']}-{span['end']} ({span['length']} tokens)")
            print(f"    Text: {text!r}")

        if result["errors"]:
            print(f"\nERRORS: {result['errors']}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Diagnose loss mask computation")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing parquet files",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Model path for tokenizer",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of samples to validate",
    )
    parser.add_argument(
        "--loss_mask_mode",
        type=str,
        choices=["all_assistant_turns", "last_assistant_only", "auto"],
        default="all_assistant_turns",
        help="Loss mask mode",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed output",
    )
    args = parser.parse_args()

    print(f"Loading tokenizer from {args.model_path}...")
    tokenizer = get_tokenizer(args.model_path)

    print(f"Loading {args.num_samples} samples from {args.input_dir}...")
    samples = load_sample_data(args.input_dir, args.num_samples)
    print(f"Loaded {len(samples)} samples")

    print(f"\nLoss mask mode: {args.loss_mask_mode}")
    print("="*60)

    results = []
    valid_count = 0
    total_valid_tokens = 0
    total_tokens = 0

    for sample in samples:
        result = validate_sample(
            sample,
            tokenizer,
            args.loss_mask_mode,
            verbose=not args.quiet,
        )
        results.append(result)

        if result["valid"]:
            valid_count += 1
            total_valid_tokens += result["valid_tokens"]
            total_tokens += result["total_tokens"]

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("="*60)
    print(f"Total samples: {len(results)}")
    print(f"Valid samples: {valid_count}")
    print(f"Invalid samples: {len(results) - valid_count}")
    print(f"Total tokens: {total_tokens}")
    print(f"Total valid tokens: {total_valid_tokens}")
    print(f"Coverage: {100 * total_valid_tokens / total_tokens:.1f}%" if total_tokens > 0 else "N/A")

    # Report errors
    errors_by_type = {}
    for result in results:
        for error in result["errors"]:
            errors_by_type[error] = errors_by_type.get(error, 0) + 1

    if errors_by_type:
        print("\nErrors by type:")
        for error, count in sorted(errors_by_type.items(), key=lambda x: -x[1]):
            print(f"  {error}: {count}")

    # Exit with error if any invalid
    if valid_count < len(results):
        print("\nSome samples failed validation!")
        sys.exit(1)
    else:
        print("\nAll samples validated successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
