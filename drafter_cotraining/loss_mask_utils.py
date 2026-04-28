"""Loss mask utilities for Eagle3 drafter pretraining.

Provides:
- compute_assistant_loss_mask: JIT-compiled O(N) scan for assistant content spans
- pack_loss_mask / unpack_loss_mask: Run-length encoding for compact transport
- has_thinking_content: Detect non-empty thinking blocks in conversations
- build_loss_mask_for_conversation: High-level API for loss mask computation

Qwen3 Template Constants (from SpecForge):
    ASSISTANT_HEADER = "<|im_start|>assistant\n"
    END_OF_TURN = "<|im_end|>\n"
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

import torch

# ─────────────────────────────────────────────────────────────────────────────
# Qwen3 Template Constants (from SpecForge qwen template)
# ─────────────────────────────────────────────────────────────────────────────

QWEN3_ASSISTANT_HEADER = "<|im_start|>assistant\n"
QWEN3_END_OF_TURN = "<|im_end|>\n"

# Regex pattern for detecting non-empty thinking content
# Matches: _filled_thinking_<non-empty-content><|im_end|>
_THINKING_PATTERN = re.compile(r"_filled_thinking_(.+?)<\|im_end\|>", re.DOTALL)


# ─────────────────────────────────────────────────────────────────────────────
# Run-Length Encoding for Loss Mask Transport
# ─────────────────────────────────────────────────────────────────────────────


def pack_loss_mask(mask: torch.Tensor) -> List[Tuple[int, int]]:
    """Pack a binary loss mask using run-length encoding.

    Args:
        mask: 1D tensor of 0s and 1s

    Returns:
        List of (value, count) pairs. Empty list for empty mask.

    Example:
        >>> mask = torch.tensor([0, 0, 1, 1, 1, 0, 0])
        >>> pack_loss_mask(mask)
        [(0, 2), (1, 3), (0, 2)]
    """
    if len(mask) == 0:
        return []

    packed = []
    current_val = int(mask[0].item())
    current_count = 1

    for i in range(1, len(mask)):
        val = int(mask[i].item())
        if val == current_val:
            current_count += 1
        else:
            packed.append((current_val, current_count))
            current_val = val
            current_count = 1

    packed.append((current_val, current_count))
    return packed


def unpack_loss_mask(packed: List[Tuple[int, int]]) -> torch.Tensor:
    """Unpack a run-length encoded loss mask.

    Args:
        packed: List of (value, count) pairs from pack_loss_mask

    Returns:
        1D tensor of 0s and 1s

    Example:
        >>> packed = [(0, 2), (1, 3), (0, 2)]
        >>> unpack_loss_mask(packed)
        tensor([0, 0, 1, 1, 1, 0, 0])
    """
    if not packed:
        return torch.tensor([], dtype=torch.long)

    # Pre-compute total length
    total_len = sum(count for _, count in packed)
    result = torch.zeros(total_len, dtype=torch.long)

    offset = 0
    for val, count in packed:
        if val == 1:
            result[offset : offset + count] = 1
        offset += count

    return result


def serialize_packed_loss_mask(packed: List[Tuple[int, int]]) -> str:
    """Serialize packed loss mask to string for transport.

    Format: "v1:c,v2:c,..." (e.g., "0:2,1:3,0:2")

    Args:
        packed: List of (value, count) pairs

    Returns:
        String representation
    """
    if not packed:
        return ""
    return ",".join(f"{v}:{c}" for v, c in packed)


def deserialize_packed_loss_mask(serialized: str) -> List[Tuple[int, int]]:
    """Deserialize string to packed loss mask.

    Args:
        serialized: String from serialize_packed_loss_mask

    Returns:
        List of (value, count) pairs
    """
    if not serialized:
        return []

    packed = []
    for part in serialized.split(","):
        if ":" not in part:
            continue
        val_str, count_str = part.split(":", 1)
        packed.append((int(val_str), int(count_str)))
    return packed


# ─────────────────────────────────────────────────────────────────────────────
# Assistant Loss Mask Computation
# ─────────────────────────────────────────────────────────────────────────────


def compute_assistant_loss_mask(
    input_ids: torch.Tensor,
    assistant_header_ids: List[int],
    end_token_ids: List[int],
    last_turn_only: bool = False,
    skip_after_header: int = 0,
) -> torch.Tensor:
    """Compute loss mask for assistant content spans.

    Scans input_ids for assistant header sequences and marks content between
    headers and end tokens as supervised (mask=1).

    Args:
        input_ids: 1D tensor of token IDs
        assistant_header_ids: Token IDs for assistant header (e.g., "<|im_start|>assistant\n")
        end_token_ids: Token IDs for end-of-turn marker (e.g., "<|im_end|>\n")
        last_turn_only: If True, only mask the last assistant turn
        skip_after_header: Number of tokens to skip after header match (for BPE-merged newlines)

    Returns:
        1D tensor of 0s and 1s, same length as input_ids

    Example:
        >>> # Simulated: [header, header, content, content, end, end]
        >>> input_ids = torch.tensor([10, 20, 1, 2, 30, 40])
        >>> mask = compute_assistant_loss_mask(input_ids, [10, 20], [30, 40])
        >>> mask
        tensor([0, 0, 1, 1, 0, 0])
    """
    seq_len = len(input_ids)
    mask = torch.zeros(seq_len, dtype=torch.long)

    if seq_len == 0:
        return mask

    header_len = len(assistant_header_ids)
    end_len = len(end_token_ids)

    if header_len == 0:
        return mask

    # Find all assistant header positions
    header_positions = []
    for i in range(seq_len - header_len + 1):
        match = True
        for j in range(header_len):
            if input_ids[i + j].item() != assistant_header_ids[j]:
                match = False
                break
        if match:
            header_positions.append(i)

    if not header_positions:
        return mask

    # If last_turn_only, keep only the last header
    if last_turn_only:
        header_positions = header_positions[-1:]

    # For each header, find content span until end token or sequence end
    for header_pos in header_positions:
        content_start = header_pos + header_len + skip_after_header

        if content_start >= seq_len:
            continue

        # Find end token position after content start
        end_pos = seq_len  # Default to end of sequence
        for i in range(content_start, seq_len - end_len + 1):
            match = True
            for j in range(end_len):
                if input_ids[i + j].item() != end_token_ids[j]:
                    match = False
                    break
            if match:
                end_pos = i
                break

        # Mark content tokens as supervised
        if content_start < end_pos:
            mask[content_start:end_pos] = 1

    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Thinking Content Detection
# ─────────────────────────────────────────────────────────────────────────────


def has_thinking_content(messages: List[dict]) -> bool:
    """Detect if conversation has non-empty thinking content.

    Checks for:
    1. Non-empty _filled_thinking_...<|im_end|> pattern in assistant content
    2. Non-empty 'thinking', 'thinking_content', or 'reasoning_content' fields

    Args:
        messages: List of message dicts with 'role' and 'content' keys

    Returns:
        True if non-empty thinking content is found in assistant messages
    """
    for msg in messages:
        if msg.get("role") != "assistant":
            continue

        # Check explicit thinking fields
        for field in ("thinking", "thinking_content", "reasoning_content"):
            thinking_val = msg.get(field)
            if thinking_val and str(thinking_val).strip():
                return True

        # Check for _filled_thinking_ pattern in content
        content = msg.get("content", "")
        if not content:
            continue

        match = _THINKING_PATTERN.search(content)
        if match:
            # Check if there's actual content between markers
            inner = match.group(1)
            if inner.strip():
                return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# High-Level API
# ─────────────────────────────────────────────────────────────────────────────

from dataclasses import dataclass


@dataclass
class LossMaskResult:
    """Result of loss mask computation for a conversation."""

    input_ids: torch.Tensor
    loss_mask: torch.Tensor
    valid_tokens: int
    formatted_text: Optional[str] = None


def build_loss_mask_for_conversation(
    tokenizer,
    messages: List[dict],
    max_length: int = 2048,
    loss_mask_mode: str = "all_assistant_turns",
    chat_template: Optional[str] = None,
) -> Optional[LossMaskResult]:
    """Build loss mask for a full conversation.

    This is the main entry point for computing loss masks during data preprocessing.
    Tokenizes the conversation and computes the loss mask for assistant content.

    Args:
        tokenizer: HuggingFace tokenizer with chat template support.
        messages: List of message dicts with 'role' and 'content' keys.
        max_length: Maximum sequence length (truncation applied).
        loss_mask_mode: One of:
            - "all_assistant_turns": Supervise all assistant content (default for pretraining)
            - "last_assistant_only": Supervise only the final assistant message
            - "auto": Use last-turn-only if thinking content detected, else all turns
        chat_template: Optional template name (currently only 'qwen' supported).

    Returns:
        LossMaskResult with input_ids, loss_mask, valid_tokens.
        Returns None if the resulting mask has no supervised tokens.
    """
    if not messages:
        return None

    # Determine last_turn_only based on mode
    if loss_mask_mode == "auto":
        last_turn_only = has_thinking_content(messages)
    elif loss_mask_mode == "last_assistant_only":
        last_turn_only = True
    else:  # "all_assistant_turns"
        last_turn_only = False

    # Tokenize full conversation
    try:
        formatted_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception:
        # Fallback for tokenizers without chat_template
        formatted_text = _fallback_format_conversation(messages, chat_template or "qwen")

    encoding = tokenizer(
        formatted_text,
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = encoding.input_ids[0]

    # Build loss mask
    loss_mask = _build_loss_mask_from_text(
        tokenizer=tokenizer,
        formatted_text=formatted_text,
        input_ids=input_ids,
        last_turn_only=last_turn_only,
        chat_template=chat_template,
    )

    # Zero the final position (no valid next-token target)
    if len(loss_mask) > 0:
        loss_mask[-1] = 0

    valid_tokens = int(loss_mask.sum().item())
    if valid_tokens == 0:
        return None

    return LossMaskResult(
        input_ids=input_ids,
        loss_mask=loss_mask,
        valid_tokens=valid_tokens,
        formatted_text=formatted_text,
    )


def _build_loss_mask_from_text(
    tokenizer,
    formatted_text: str,
    input_ids: torch.Tensor,
    last_turn_only: bool,
    chat_template: Optional[str],
) -> torch.Tensor:
    """Build loss mask from formatted text using token ID scanning."""
    # Get header and end token IDs for Qwen3 template
    assistant_header = QWEN3_ASSISTANT_HEADER
    end_of_turn = QWEN3_END_OF_TURN

    # Encode header and end tokens
    header_ids = tokenizer.encode(assistant_header, add_special_tokens=False)
    end_ids = tokenizer.encode(end_of_turn, add_special_tokens=False)

    # Determine skip_after_header: check if header ends with newline that might merge
    # Qwen3 header "<|im_start|>assistant\n" ends with newline
    stripped_header = assistant_header.rstrip("\n")
    stripped_ids = tokenizer.encode(stripped_header, add_special_tokens=False)
    skip_after_header = len(header_ids) - len(stripped_ids)

    return compute_assistant_loss_mask(
        input_ids=input_ids,
        assistant_header_ids=header_ids,
        end_token_ids=end_ids,
        last_turn_only=last_turn_only,
        skip_after_header=skip_after_header,
    )


def _fallback_format_conversation(
    messages: List[dict],
    chat_template: str = "qwen",
) -> str:
    """Fallback conversation formatting for tokenizers without chat_template."""
    QWEN3_USER_HEADER = "<|im_start|>user\n"
    QWEN3_SYSTEM_HEADER = "<|im_start|>system\n"

    parts = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            parts.append(f"{QWEN3_SYSTEM_HEADER}{content}{QWEN3_END_OF_TURN}")
        elif role == "user":
            parts.append(f"{QWEN3_USER_HEADER}{content}{QWEN3_END_OF_TURN}")
        elif role == "assistant":
            parts.append(f"{QWEN3_ASSISTANT_HEADER}{content}{QWEN3_END_OF_TURN}")

    return "".join(parts)
