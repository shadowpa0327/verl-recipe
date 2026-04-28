"""Unit tests for loss mask utilities.

Tests for:
- pack_loss_mask / unpack_loss_mask round-trip
- compute_assistant_loss_mask correctness
- has_thinking_content detection
"""

import pytest
import torch

from recipe.drafter_cotraining.loss_mask_utils import (
    compute_assistant_loss_mask,
    has_thinking_content,
    pack_loss_mask,
    serialize_packed_loss_mask,
    deserialize_packed_loss_mask,
    unpack_loss_mask,
)


class TestPackUnpackLossMask:
    """Tests for pack_loss_mask and unpack_loss_mask round-trip."""

    def test_basic_round_trip(self):
        """Simple mask with one assistant span."""
        mask = torch.tensor([0, 0, 1, 1, 1, 0, 0], dtype=torch.long)
        packed = pack_loss_mask(mask)
        unpacked = unpack_loss_mask(packed)
        assert torch.equal(mask, unpacked)

    def test_multiple_spans(self):
        """Mask with multiple assistant spans."""
        mask = torch.tensor([0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0], dtype=torch.long)
        packed = pack_loss_mask(mask)
        unpacked = unpack_loss_mask(packed)
        assert torch.equal(mask, unpacked)

    def test_leading_zeros(self):
        """Mask starting with zeros."""
        mask = torch.tensor([0, 0, 0, 0, 1, 1, 1], dtype=torch.long)
        packed = pack_loss_mask(mask)
        unpacked = unpack_loss_mask(packed)
        assert torch.equal(mask, unpacked)

    def test_all_zeros(self):
        """All-zero mask."""
        mask = torch.tensor([0, 0, 0, 0, 0], dtype=torch.long)
        packed = pack_loss_mask(mask)
        unpacked = unpack_loss_mask(packed)
        assert torch.equal(mask, unpacked)

    def test_all_ones(self):
        """All-ones mask (starts with implicit zero-length prompt)."""
        mask = torch.tensor([1, 1, 1, 1, 1], dtype=torch.long)
        packed = pack_loss_mask(mask)
        unpacked = unpack_loss_mask(packed)
        assert torch.equal(mask, unpacked)

    def test_empty_mask(self):
        """Empty mask."""
        mask = torch.tensor([], dtype=torch.long)
        packed = pack_loss_mask(mask)
        assert packed == []
        unpacked = unpack_loss_mask(packed)
        assert len(unpacked) == 0

    def test_serialize_deserialize(self):
        """Serialization to string format."""
        mask = torch.tensor([0, 0, 1, 1, 1, 0, 0, 1, 1], dtype=torch.long)
        packed = pack_loss_mask(mask)
        serialized = serialize_packed_loss_mask(packed)
        deserialized = deserialize_packed_loss_mask(serialized)
        assert packed == deserialized
        unpacked = unpack_loss_mask(deserialized)
        assert torch.equal(mask, unpacked)


class TestComputeAssistantLossMask:
    """Tests for compute_assistant_loss_mask with simulated token IDs."""

    def test_single_assistant_span(self):
        """Single assistant content span between header and end token."""
        # Simulate: [header_id, header_id, content, content, end_id, end_id]
        input_ids = torch.tensor([10, 20, 1, 2, 30, 40], dtype=torch.long)
        header_ids = [10, 20]
        end_ids = [30, 40]

        mask = compute_assistant_loss_mask(input_ids, header_ids, end_ids)

        # Content tokens should be masked as 1, header and end as 0
        expected = torch.tensor([0, 0, 1, 1, 0, 0], dtype=torch.long)
        assert torch.equal(mask, expected)

    def test_multiple_assistant_spans(self):
        """Multiple assistant turns."""
        # Two assistant spans
        input_ids = torch.tensor([10, 20, 1, 2, 30, 40, 10, 20, 5, 30, 40], dtype=torch.long)
        header_ids = [10, 20]
        end_ids = [30, 40]

        mask = compute_assistant_loss_mask(input_ids, header_ids, end_ids)

        expected = torch.tensor([0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0], dtype=torch.long)
        assert torch.equal(mask, expected)

    def test_last_turn_only(self):
        """last_turn_only=True should mask only the last assistant turn."""
        input_ids = torch.tensor([10, 20, 1, 2, 30, 40, 10, 20, 5, 30, 40], dtype=torch.long)
        header_ids = [10, 20]
        end_ids = [30, 40]

        mask = compute_assistant_loss_mask(input_ids, header_ids, end_ids, last_turn_only=True)

        # Only the second assistant span should be masked
        expected = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], dtype=torch.long)
        assert torch.equal(mask, expected)

    def test_no_header_match(self):
        """No header found should return all zeros."""
        input_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
        header_ids = [10, 20]
        end_ids = [30, 40]

        mask = compute_assistant_loss_mask(input_ids, header_ids, end_ids)

        expected = torch.tensor([0, 0, 0, 0, 0], dtype=torch.long)
        assert torch.equal(mask, expected)

    def test_incomplete_span(self):
        """Header found but no end token - should mask to end of sequence."""
        input_ids = torch.tensor([10, 20, 1, 2, 3], dtype=torch.long)
        header_ids = [10, 20]
        end_ids = [30, 40]

        mask = compute_assistant_loss_mask(input_ids, header_ids, end_ids)

        # Should mask content tokens to the end
        expected = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
        assert torch.equal(mask, expected)

    def test_skip_after_header(self):
        """Skip tokens after header (e.g., for BPE-merged newlines)."""
        # Header + newline + content
        input_ids = torch.tensor([10, 20, 99, 1, 2, 30, 40], dtype=torch.long)
        header_ids = [10, 20]
        end_ids = [30, 40]

        # Skip 1 token after header
        mask = compute_assistant_loss_mask(input_ids, header_ids, end_ids, skip_after_header=1)

        # Token 99 (newline) should be skipped
        expected = torch.tensor([0, 0, 0, 1, 1, 0, 0], dtype=torch.long)
        assert torch.equal(mask, expected)


class TestHasThinkingContent:
    """Tests for thinking content detection."""

    def test_think_tag_with_content(self):
        """Non-empty thinking tag should be detected."""
        conv = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "_filled_thinking_reasoning here<|im_end|>Hello!"},
        ]
        assert has_thinking_content(conv) is True

    def test_empty_think_tag(self):
        """Empty thinking tag should not be detected."""
        conv = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "_filled_thinking_<|im_end|>Hello!"},
        ]
        assert has_thinking_content(conv) is False

    def test_no_think_tag(self):
        """No thinking tag should not be detected."""
        conv = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        assert has_thinking_content(conv) is False

    def test_thinking_field(self):
        """`thinking` field should be detected."""
        conv = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!", "thinking": "some reasoning"},
        ]
        assert has_thinking_content(conv) is True

    def test_thinking_content_field(self):
        """`thinking_content` field should be detected."""
        conv = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!", "thinking_content": "reasoning"},
        ]
        assert has_thinking_content(conv) is True

    def test_reasoning_content_field(self):
        """`reasoning_content` field should be detected."""
        conv = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!", "reasoning_content": "step by step"},
        ]
        assert has_thinking_content(conv) is True

    def test_empty_thinking_field(self):
        """Empty thinking field should not be detected."""
        conv = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!", "thinking": ""},
        ]
        assert has_thinking_content(conv) is False

    def test_user_thinking_ignored(self):
        """Thinking in user messages should be ignored."""
        conv = [
            {"role": "user", "content": "_filled_thinking_user thought<|im_end|>Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        assert has_thinking_content(conv) is False

    def test_multi_turn_thinking_in_earlier_turn(self):
        """Thinking in earlier turns should be detected."""
        conv = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "_filled_thinking_thought<|im_end|>A1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "_filled_thinking_<|im_end|>A2"},
        ]
        assert has_thinking_content(conv) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
