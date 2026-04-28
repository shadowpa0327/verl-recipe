"""Cross-validation tests for loss mask implementation.

Mirrors TorchSpec's test_loss_mask_cross_validation.py to verify the local
implementation matches reference behavior for Qwen3 template.

These tests validate:
- Multi-turn conversations with all_assistant_turns mode
- Multi-turn conversations with last_assistant_only mode
- Thinking content detection for auto mode
- Loss mask alignment with input_ids
"""

import pytest
import torch

from recipe.drafter_cotraining.loss_mask_utils import (
    QWEN3_ASSISTANT_HEADER,
    QWEN3_END_OF_TURN,
    compute_assistant_loss_mask,
    has_thinking_content,
    pack_loss_mask,
    unpack_loss_mask,
)


class TestMultiTurnNoThinking:
    """Tests for multi-turn conversations without thinking content."""

    def test_all_assistant_turns_masks_all(self):
        """all_assistant_turns mode should mask all assistant content."""
        # Multi-turn conversation: user1, assistant1, user2, assistant2
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2 equals 4."},
            {"role": "user", "content": "What about 3+3?"},
            {"role": "assistant", "content": "3+3 equals 6."},
        ]

        # Simulate tokenized input_ids with markers
        # Format: [user]...<|im_end|>[assistant]content<|im_end|>...
        # Simplified representation:
        header_ids = [100, 101]  # Simulated assistant header
        end_ids = [200]  # Simulated end token

        input_ids = torch.tensor([
            # User turn 1
            1, 2, 3,  # user content
            200,  # end
            # Assistant turn 1
            100, 101,  # header
            10, 11, 12,  # assistant content
            200,  # end
            # User turn 2
            4, 5, 6,  # user content
            200,  # end
            # Assistant turn 2
            100, 101,  # header
            20, 21, 22, 23,  # assistant content
            200,  # end
        ], dtype=torch.long)

        mask = compute_assistant_loss_mask(
            input_ids=input_ids,
            assistant_header_ids=header_ids,
            end_token_ids=end_ids,
            last_turn_only=False,
        )

        # Assistant content spans should be masked (not headers or end tokens)
        expected = torch.tensor([
            0, 0, 0,  # user content
            0,  # end
            0, 0,  # header
            1, 1, 1,  # assistant content
            0,  # end
            0, 0, 0,  # user content
            0,  # end
            0, 0,  # header
            1, 1, 1, 1,  # assistant content
            0,  # end
        ], dtype=torch.long)

        assert torch.equal(mask, expected), f"Expected {expected}, got {mask}"

    def test_last_assistant_only_masks_last(self):
        """last_assistant_only mode should mask only the final assistant turn."""
        header_ids = [100, 101]
        end_ids = [200]

        input_ids = torch.tensor([
            # User turn 1
            1, 2, 3, 200,
            # Assistant turn 1
            100, 101, 10, 11, 12, 200,
            # User turn 2
            4, 5, 6, 200,
            # Assistant turn 2
            100, 101, 20, 21, 22, 23, 200,
        ], dtype=torch.long)

        mask = compute_assistant_loss_mask(
            input_ids=input_ids,
            assistant_header_ids=header_ids,
            end_token_ids=end_ids,
            last_turn_only=True,
        )

        # Only the second assistant content should be masked
        expected = torch.tensor([
            0, 0, 0, 0,  # user turn 1
            0, 0, 0, 0, 0, 0,  # assistant turn 1 (not masked)
            0, 0, 0, 0,  # user turn 2
            0, 0, 1, 1, 1, 1, 0,  # assistant turn 2 (masked)
        ], dtype=torch.long)

        assert torch.equal(mask, expected)

    def test_auto_mode_matches_all_assistant_when_no_thinking(self):
        """auto mode should use all_assistant_turns when no thinking detected."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm doing well, thanks!"},
        ]

        # No thinking content
        assert has_thinking_content(messages) is False

        # auto mode should behave like all_assistant_turns
        # (verified through has_thinking_content returning False)
        header_ids = [100, 101]
        end_ids = [200]

        input_ids = torch.tensor([
            1, 2, 200,  # user1
            100, 101, 10, 11, 200,  # assistant1
            3, 4, 200,  # user2
            100, 101, 20, 21, 22, 200,  # assistant2
        ], dtype=torch.long)

        mask_auto = compute_assistant_loss_mask(
            input_ids=input_ids,
            assistant_header_ids=header_ids,
            end_token_ids=end_ids,
            last_turn_only=False,  # auto without thinking -> all_assistant_turns
        )

        mask_all = compute_assistant_loss_mask(
            input_ids=input_ids,
            assistant_header_ids=header_ids,
            end_token_ids=end_ids,
            last_turn_only=False,
        )

        assert torch.equal(mask_auto, mask_all)


class TestMultiTurnWithThinking:
    """Tests for multi-turn conversations with thinking content."""

    def test_all_assistant_turns_includes_thinking(self):
        """all_assistant_turns mode should include thinking tokens."""
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "_filled_thinking_Let me calculate...<|im_end|>2+2 equals 4."},
            {"role": "user", "content": "Thanks!"},
            {"role": "assistant", "content": "You're welcome!"},
        ]

        assert has_thinking_content(messages) is True

    def test_auto_mode_uses_last_only_when_thinking(self):
        """auto mode should use last_assistant_only when thinking is detected."""
        messages = [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "_filled_thinking_reasoning<|im_end|>Answer"},
            {"role": "user", "content": "Follow-up"},
            {"role": "assistant", "content": "Response"},
        ]

        # Thinking content detected
        assert has_thinking_content(messages) is True

        # auto mode should behave like last_assistant_only
        # (verified through has_thinking_content returning True)
        header_ids = [100, 101]
        end_ids = [200]

        input_ids = torch.tensor([
            1, 200,  # user1
            100, 101, 10, 11, 12, 13, 200,  # assistant1 (with thinking)
            2, 200,  # user2
            100, 101, 20, 21, 200,  # assistant2
        ], dtype=torch.long)

        mask_auto = compute_assistant_loss_mask(
            input_ids=input_ids,
            assistant_header_ids=header_ids,
            end_token_ids=end_ids,
            last_turn_only=True,  # auto with thinking -> last_assistant_only
        )

        mask_last = compute_assistant_loss_mask(
            input_ids=input_ids,
            assistant_header_ids=header_ids,
            end_token_ids=end_ids,
            last_turn_only=True,
        )

        assert torch.equal(mask_auto, mask_last)


class TestLeadingNewlines:
    """Tests for assistant content with leading newlines."""

    def test_leading_newline_in_content(self):
        """Assistant content may start with newline after header."""
        header_ids = [100]  # Simplified header
        end_ids = [200]

        # Header followed by newline token, then content
        input_ids = torch.tensor([
            1, 2, 200,  # user
            100,  # header
            99,  # newline token
            10, 11,  # content
            200,  # end
        ], dtype=torch.long)

        # Input: [user, user, end, header, newline, content, content, end]
        # Tokens: 1, 2, 200, 100, 99, 10, 11, 200 (8 tokens total)
        # Positions: 0, 1, 2, 3, 4, 5, 6, 7

        # Without skip_after_header, newline is treated as content
        mask_no_skip = compute_assistant_loss_mask(
            input_ids=input_ids,
            assistant_header_ids=header_ids,
            end_token_ids=end_ids,
            skip_after_header=0,
        )

        # Expected: user(0), user(0), end(0), header(0), newline(1), content(1), content(1), end(0)
        expected_no_skip = torch.tensor([0, 0, 0, 0, 1, 1, 1, 0], dtype=torch.long)
        assert torch.equal(mask_no_skip, expected_no_skip)

        # With skip_after_header=1, newline is skipped
        mask_skip = compute_assistant_loss_mask(
            input_ids=input_ids,
            assistant_header_ids=header_ids,
            end_token_ids=end_ids,
            skip_after_header=1,
        )

        # Expected: user(0), user(0), end(0), header(0), newline(0), content(1), content(1), end(0)
        expected_skip = torch.tensor([0, 0, 0, 0, 0, 1, 1, 0], dtype=torch.long)
        assert torch.equal(mask_skip, expected_skip)


class TestLossMaskAlignment:
    """Tests for loss mask alignment with input_ids."""

    def test_mask_matches_input_length(self):
        """Loss mask length should match input_ids length."""
        header_ids = [100]
        end_ids = [200]

        for seq_len in [10, 50, 100, 256]:
            input_ids = torch.randint(1, 500, (seq_len,), dtype=torch.long)
            mask = compute_assistant_loss_mask(input_ids, header_ids, end_ids)
            assert len(mask) == len(input_ids), f"Length mismatch for seq_len={seq_len}"

    def test_empty_input(self):
        """Empty input should return empty mask."""
        input_ids = torch.tensor([], dtype=torch.long)
        mask = compute_assistant_loss_mask(input_ids, [100], [200])
        assert len(mask) == 0

    def test_single_assistant_span(self):
        """Single assistant span should be correctly identified."""
        header_ids = [100, 101]
        end_ids = [200, 201]

        input_ids = torch.tensor([
            1, 2, 3,  # prompt
            100, 101,  # header
            50, 51, 52, 53,  # content
            200, 201,  # end
        ], dtype=torch.long)

        mask = compute_assistant_loss_mask(input_ids, header_ids, end_ids)

        expected = torch.tensor([
            0, 0, 0,  # prompt
            0, 0,  # header
            1, 1, 1, 1,  # content
            0, 0,  # end
        ], dtype=torch.long)

        assert torch.equal(mask, expected)


class TestPackUnpackRoundTrip:
    """Tests for pack/unpack preserving mask semantics."""

    def test_complex_multi_turn_mask(self):
        """Complex multi-turn mask should round-trip correctly."""
        # Simulate realistic multi-turn mask
        mask = torch.tensor([
            # Turn 1: user (0s)
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            # Turn 1: assistant content (1s)
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            # Turn 2: user (0s)
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            # Turn 2: assistant content (1s)
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            # Final padding (0s)
            0, 0, 0, 0,
        ], dtype=torch.long)

        packed = pack_loss_mask(mask)
        unpacked = unpack_loss_mask(packed)

        assert torch.equal(mask, unpacked)

        # Verify valid_tokens count
        valid_tokens = int(mask.sum().item())
        assert valid_tokens == 24


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_no_assistant_turn(self):
        """Conversation without assistant should return all zeros."""
        header_ids = [100]
        end_ids = [200]

        # Only user messages
        input_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
        mask = compute_assistant_loss_mask(input_ids, header_ids, end_ids)

        expected = torch.zeros(5, dtype=torch.long)
        assert torch.equal(mask, expected)

    def test_incomplete_header_at_end(self):
        """Incomplete header at sequence end should not cause errors."""
        header_ids = [100, 101]
        end_ids = [200]

        # Sequence ends with partial header
        input_ids = torch.tensor([1, 2, 3, 100], dtype=torch.long)
        mask = compute_assistant_loss_mask(input_ids, header_ids, end_ids)

        expected = torch.tensor([0, 0, 0, 0], dtype=torch.long)
        assert torch.equal(mask, expected)

    def test_consecutive_assistant_turns(self):
        """Consecutive assistant turns (no user between) should both be masked."""
        header_ids = [100]
        end_ids = [200]

        input_ids = torch.tensor([
            100, 10, 11, 200,  # assistant 1
            100, 20, 21, 200,  # assistant 2
        ], dtype=torch.long)

        mask = compute_assistant_loss_mask(input_ids, header_ids, end_ids)

        expected = torch.tensor([
            0, 1, 1, 0,  # assistant 1
            0, 1, 1, 0,  # assistant 2
        ], dtype=torch.long)

        assert torch.equal(mask, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
