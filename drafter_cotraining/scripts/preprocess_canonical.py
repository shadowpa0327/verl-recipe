#!/usr/bin/env python3
"""Canonical data preprocessing and turn analysis for Eagle3 drafter pretraining.

This script:
1. Loads all parquet files from input directory
2. Normalizes conversations to canonical format (messages with role/content)
3. Analyzes turn statistics (turn counts, role distribution, content lengths)
4. Optionally outputs preprocessed data to new parquet files

Usage:
    python preprocess_canonical.py \
        --input_dir /path/to/data \
        --output_dir /path/to/output \
        --analyze_only \
        --num_samples 1000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq


# ─────────────────────────────────────────────────────────────────────────────
# Role Normalization
# ─────────────────────────────────────────────────────────────────────────────

ROLE_ALIASES = {
    # User roles
    "human": "user",
    "user": "user",
    "question": "user",
    
    # Assistant roles
    "assistant": "assistant",
    "gpt": "assistant",
    "bot": "assistant",
    "model": "assistant",
    "answer": "assistant",
    
    # System role
    "system": "system",
    
    # Tool role
    "tool": "tool",
    "function": "tool",
}


def normalize_role(role: str) -> str:
    """Normalize role name to canonical form."""
    role_lower = str(role).lower().strip()
    return ROLE_ALIASES.get(role_lower, role_lower)


def normalize_message(msg: Any) -> Optional[Dict[str, Any]]:
    """Normalize a single message to canonical format."""
    if isinstance(msg, dict):
        # Try various role keys
        role = msg.get("role") or msg.get("from") or msg.get("sender")
        
        # Try various content keys
        content = msg.get("content") or msg.get("text") or msg.get("value") or msg.get("message")
        
        if role:
            return {
                "role": normalize_role(role),
                "content": str(content) if content else "",
            }
    elif isinstance(msg, (list, tuple)) and len(msg) >= 2:
        # Format: [role, content]
        return {
            "role": normalize_role(str(msg[0])),
            "content": str(msg[1]) if len(msg) > 1 else "",
        }
    
    return None


def normalize_conversations(conversations: List[Any]) -> List[Dict[str, Any]]:
    """Normalize conversations to canonical format."""
    normalized = []
    for msg in conversations:
        norm_msg = normalize_message(msg)
        if norm_msg:
            normalized.append(norm_msg)
    return normalized


# ─────────────────────────────────────────────────────────────────────────────
# Turn Analysis
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ConversationStats:
    """Statistics for a single conversation."""
    sample_id: str
    num_turns: int
    num_user_turns: int
    num_assistant_turns: int
    num_system_turns: int
    num_tool_turns: int
    total_user_chars: int
    total_assistant_chars: int
    avg_user_chars: float
    avg_assistant_chars: float
    has_thinking: bool
    roles_sequence: List[str]
    is_multi_turn: bool


@dataclass
class DatasetStats:
    """Aggregate statistics for a dataset."""
    total_samples: int = 0
    total_turns: int = 0
    
    # Turn count distribution
    turn_counts: Counter = field(default_factory=Counter)
    
    # Role distribution
    role_counts: Counter = field(default_factory=Counter)
    
    # Multi-turn stats
    single_turn_samples: int = 0
    multi_turn_samples: int = 0
    
    # Thinking content
    samples_with_thinking: int = 0
    
    # Content length stats
    user_char_counts: List[int] = field(default_factory=list)
    assistant_char_counts: List[int] = field(default_factory=list)
    
    # Role sequences (for pattern analysis)
    role_sequences: Counter = field(default_factory=Counter)
    
    # Invalid samples
    invalid_samples: int = 0
    invalid_reasons: Counter = field(default_factory=Counter)
    
    def add_conversation(self, stats: ConversationStats):
        """Add a conversation's stats to the aggregate."""
        self.total_samples += 1
        self.total_turns += stats.num_turns
        
        self.turn_counts[stats.num_turns] += 1
        self.role_counts["user"] += stats.num_user_turns
        self.role_counts["assistant"] += stats.num_assistant_turns
        self.role_counts["system"] += stats.num_system_turns
        self.role_counts["tool"] += stats.num_tool_turns
        
        if stats.is_multi_turn:
            self.multi_turn_samples += 1
        else:
            self.single_turn_samples += 1
        
        if stats.has_thinking:
            self.samples_with_thinking += 1
        
        self.user_char_counts.append(stats.total_user_chars)
        self.assistant_char_counts.append(stats.total_assistant_chars)
        
        # Track role sequence pattern (simplified)
        seq_key = "-".join(stats.roles_sequence[:10])  # Limit to first 10
        self.role_sequences[seq_key] += 1
    
    def add_invalid(self, reason: str):
        """Record an invalid sample."""
        self.invalid_samples += 1
        self.invalid_reasons[reason] += 1
    
    def summarize(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        import statistics
        
        summary = {
            "total_samples": self.total_samples,
            "total_turns": self.total_turns,
            "avg_turns_per_sample": self.total_turns / max(self.total_samples, 1),
            
            "turn_distribution": dict(self.turn_counts.most_common(20)),
            
            "role_distribution": dict(self.role_counts),
            
            "single_turn_samples": self.single_turn_samples,
            "multi_turn_samples": self.multi_turn_samples,
            "multi_turn_ratio": self.multi_turn_samples / max(self.total_samples, 1),
            
            "samples_with_thinking": self.samples_with_thinking,
            "thinking_ratio": self.samples_with_thinking / max(self.total_samples, 1),
            
            "invalid_samples": self.invalid_samples,
            "invalid_reasons": dict(self.invalid_reasons.most_common(10)),
        }
        
        if self.user_char_counts:
            summary["user_chars"] = {
                "min": min(self.user_char_counts),
                "max": max(self.user_char_counts),
                "mean": statistics.mean(self.user_char_counts),
                "median": statistics.median(self.user_char_counts),
            }
        
        if self.assistant_char_counts:
            summary["assistant_chars"] = {
                "min": min(self.assistant_char_counts),
                "max": max(self.assistant_char_counts),
                "mean": statistics.mean(self.assistant_char_counts),
                "median": statistics.median(self.assistant_char_counts),
            }
        
        # Top role sequences
        summary["top_role_sequences"] = dict(self.role_sequences.most_common(10))
        
        return summary


def analyze_conversation(sample_id: str, messages: List[Dict[str, Any]]) -> ConversationStats:
    """Analyze a single conversation."""
    num_user_turns = 0
    num_assistant_turns = 0
    num_system_turns = 0
    num_tool_turns = 0
    total_user_chars = 0
    total_assistant_chars = 0
    roles_sequence = []
    has_thinking = False
    
    # Thinking detection pattern
    import re
    thinking_pattern = re.compile(r"_filled_thinking_(.+?)<\|im_end\|>", re.DOTALL)
    
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        content_len = len(content) if content else 0
        
        roles_sequence.append(role)
        
        if role == "user":
            num_user_turns += 1
            total_user_chars += content_len
        elif role == "assistant":
            num_assistant_turns += 1
            total_assistant_chars += content_len
            
            # Check for thinking content
            if content and thinking_pattern.search(content):
                has_thinking = True
            # Also check for thinking fields
            if msg.get("thinking") or msg.get("thinking_content") or msg.get("reasoning_content"):
                has_thinking = True
        elif role == "system":
            num_system_turns += 1
        elif role == "tool":
            num_tool_turns += 1
    
    num_turns = len(messages)
    
    return ConversationStats(
        sample_id=sample_id,
        num_turns=num_turns,
        num_user_turns=num_user_turns,
        num_assistant_turns=num_assistant_turns,
        num_system_turns=num_system_turns,
        num_tool_turns=num_tool_turns,
        total_user_chars=total_user_chars,
        total_assistant_chars=total_assistant_chars,
        avg_user_chars=total_user_chars / max(num_user_turns, 1),
        avg_assistant_chars=total_assistant_chars / max(num_assistant_turns, 1),
        has_thinking=has_thinking,
        roles_sequence=roles_sequence,
        is_multi_turn=num_assistant_turns > 1,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading and Processing
# ─────────────────────────────────────────────────────────────────────────────

def load_parquet_files(input_dir: str) -> List[Path]:
    """Find all parquet files in directory."""
    input_path = Path(input_dir)
    parquet_files = sorted(input_path.glob("*.parquet"))
    
    if not parquet_files:
        # Check subdirectories
        parquet_files = sorted(input_path.rglob("*.parquet"))
    
    return parquet_files


def process_parquet_file(
    file_path: Path,
    max_samples: int = -1,
    analyze: bool = True,
    output_dir: Optional[Path] = None,
) -> tuple[DatasetStats, int]:
    """Process a single parquet file."""
    stats = DatasetStats()
    processed_count = 0
    
    table = pq.read_table(file_path)
    total_rows = table.num_rows
    
    # Determine sample limit
    sample_limit = min(max_samples, total_rows) if max_samples > 0 else total_rows
    
    # Prepare output data if needed
    output_rows = [] if output_dir else None
    
    for i in range(sample_limit):
        row = table.slice(i, 1).to_pydict()
        
        # Extract sample ID
        sample_id = row.get("id", [f"sample_{i}"])[0]
        if isinstance(sample_id, list):
            sample_id = sample_id[0] if sample_id else f"sample_{i}"
        
        # Extract conversations
        conversations = None
        for key in ["conversations", "messages", "dialogue", "history"]:
            if key in row and row[key]:
                conversations = row[key][0]
                break
        
        # Try legacy format
        if conversations is None:
            prompt_messages = row.get("prompt_messages", [None])[0]
            response = row.get("response", [None])[0]
            if prompt_messages and response:
                conversations = list(prompt_messages) if isinstance(prompt_messages, list) else []
                conversations.append({"role": "assistant", "content": response})
        
        if not conversations:
            stats.add_invalid("No conversations found")
            continue
        
        # Normalize conversations
        normalized = normalize_conversations(conversations)
        
        if not normalized:
            stats.add_invalid("Empty after normalization")
            continue
        
        # Validate: must have at least one assistant turn
        has_assistant = any(m.get("role") == "assistant" for m in normalized)
        if not has_assistant:
            stats.add_invalid("No assistant turn")
            continue
        
        # Analyze
        if analyze:
            conv_stats = analyze_conversation(sample_id, normalized)
            stats.add_conversation(conv_stats)
        
        # Prepare output
        if output_dir:
            output_rows.append({
                "id": sample_id,
                "messages": normalized,
                "num_turns": len(normalized),
                "num_assistant_turns": sum(1 for m in normalized if m.get("role") == "assistant"),
            })
        
        processed_count += 1
    
    # Write output
    if output_dir and output_rows:
        output_file = output_dir / file_path.name
        output_table = pa.Table.from_pylist(output_rows)
        pq.write_table(output_table, output_file)
        print(f"  Written: {output_file}")
    
    return stats, processed_count


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(stats: DatasetStats):
    """Print formatted summary."""
    summary = stats.summarize()
    
    print("\n" + "=" * 70)
    print("DATASET ANALYSIS SUMMARY")
    print("=" * 70)
    
    print(f"\n{'SAMPLES':^70}")
    print("-" * 70)
    print(f"  Total samples processed:  {summary['total_samples']:,}")
    print(f"  Invalid samples:          {summary['invalid_samples']:,}")
    if summary['invalid_reasons']:
        print("  Invalid reasons:")
        for reason, count in summary['invalid_reasons'].items():
            print(f"    - {reason}: {count}")
    
    print(f"\n{'TURNS':^70}")
    print("-" * 70)
    print(f"  Total turns:              {summary['total_turns']:,}")
    print(f"  Avg turns per sample:     {summary['avg_turns_per_sample']:.2f}")
    print(f"  Single-turn samples:      {summary['single_turn_samples']:,} ({100*summary['single_turn_samples']/max(summary['total_samples'],1):.1f}%)")
    print(f"  Multi-turn samples:       {summary['multi_turn_samples']:,} ({100*summary['multi_turn_ratio']:.1f}%)")
    
    print(f"\n{'TURN COUNT DISTRIBUTION':^70}")
    print("-" * 70)
    for turn_count, count in list(summary['turn_distribution'].items())[:10]:
        pct = 100 * count / summary['total_samples']
        bar = "█" * int(pct / 2)
        print(f"  {turn_count:3d} turns: {count:6,} ({pct:5.1f}%) {bar}")
    
    print(f"\n{'ROLE DISTRIBUTION':^70}")
    print("-" * 70)
    for role, count in sorted(summary['role_distribution'].items(), key=lambda x: -x[1]):
        pct = 100 * count / summary['total_turns']
        print(f"  {role:12s}: {count:8,} ({pct:5.1f}%)")
    
    print(f"\n{'THINKING CONTENT':^70}")
    print("-" * 70)
    print(f"  Samples with thinking:    {summary['samples_with_thinking']:,} ({100*summary['thinking_ratio']:.1f}%)")
    
    if 'user_chars' in summary:
        print(f"\n{'USER CONTENT LENGTH (chars)':^70}")
        print("-" * 70)
        uc = summary['user_chars']
        print(f"  Min:     {uc['min']:,}")
        print(f"  Max:     {uc['max']:,}")
        print(f"  Mean:    {uc['mean']:.1f}")
        print(f"  Median:  {uc['median']:.1f}")
    
    if 'assistant_chars' in summary:
        print(f"\n{'ASSISTANT CONTENT LENGTH (chars)':^70}")
        print("-" * 70)
        ac = summary['assistant_chars']
        print(f"  Min:     {ac['min']:,}")
        print(f"  Max:     {ac['max']:,}")
        print(f"  Mean:    {ac['mean']:.1f}")
        print(f"  Median:  {ac['median']:.1f}")
    
    print(f"\n{'TOP ROLE SEQUENCES':^70}")
    print("-" * 70)
    for seq, count in list(summary['top_role_sequences'].items())[:5]:
        pct = 100 * count / summary['total_samples']
        print(f"  {seq[:50]:50s}: {count:6,} ({pct:5.1f}%)")
    
    print("\n" + "=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Canonical data preprocessing and turn analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze only
  python preprocess_canonical.py --input_dir /path/to/data --analyze_only

  # Preprocess and output
  python preprocess_canonical.py --input_dir /path/to/data --output_dir /path/to/output

  # Limit samples
  python preprocess_canonical.py --input_dir /path/to/data --num_samples 1000 --analyze_only
        """,
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory containing parquet files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for preprocessed parquet (default: analyze only)",
    )
    parser.add_argument(
        "--analyze_only",
        action="store_true",
        help="Only analyze, don't write output files",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=-1,
        help="Maximum samples to process per file (-1 for all)",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Output JSON file for statistics",
    )
    args = parser.parse_args()
    
    # Validate input
    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"ERROR: Input directory not found: {args.input_dir}")
        sys.exit(1)
    
    # Setup output
    output_dir = None
    if args.output_dir and not args.analyze_only:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir}")
    
    # Find parquet files
    parquet_files = load_parquet_files(args.input_dir)
    if not parquet_files:
        print(f"ERROR: No parquet files found in {args.input_dir}")
        sys.exit(1)
    
    print(f"Found {len(parquet_files)} parquet file(s)")
    for f in parquet_files:
        print(f"  - {f}")
    
    # Process files
    total_stats = DatasetStats()
    total_processed = 0
    
    for file_path in parquet_files:
        print(f"\nProcessing: {file_path.name}")
        stats, processed = process_parquet_file(
            file_path,
            max_samples=args.num_samples,
            analyze=True,
            output_dir=output_dir,
        )
        
        # Merge stats
        total_stats.total_samples += stats.total_samples
        total_stats.total_turns += stats.total_turns
        total_stats.turn_counts.update(stats.turn_counts)
        total_stats.role_counts.update(stats.role_counts)
        total_stats.single_turn_samples += stats.single_turn_samples
        total_stats.multi_turn_samples += stats.multi_turn_samples
        total_stats.samples_with_thinking += stats.samples_with_thinking
        total_stats.user_char_counts.extend(stats.user_char_counts)
        total_stats.assistant_char_counts.extend(stats.assistant_char_counts)
        total_stats.role_sequences.update(stats.role_sequences)
        total_stats.invalid_samples += stats.invalid_samples
        total_stats.invalid_reasons.update(stats.invalid_reasons)
        
        total_processed += processed
        print(f"  Processed: {processed:,} samples")
    
    # Print summary
    print_summary(total_stats)
    
    # Output JSON
    if args.output_json:
        summary = total_stats.summarize()
        with open(args.output_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nStatistics written to: {args.output_json}")
    
    print(f"\nTotal samples processed: {total_processed:,}")


if __name__ == "__main__":
    main()
