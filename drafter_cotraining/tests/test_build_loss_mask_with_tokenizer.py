"""End-to-end tests for loss mask computation with real tokenizers.

Mirrors SpecForge's test_build_eagle3_dataset.py to validate:
- Multi-turn conversations with real Qwen3 tokenizer
- Tool-use conversations with special formatting
- Visual output showing supervised tokens (in red)
- Loss mask alignment with input_ids
"""

import os
import tempfile
import unittest

import torch
from datasets import Dataset
from transformers import AutoTokenizer

from recipe.drafter_cotraining.loss_mask_utils import (
    LossMaskResult,
    build_loss_mask_for_conversation,
    compute_assistant_loss_mask,
    has_thinking_content,
    pack_loss_mask,
    unpack_loss_mask,
)

# ANSI color codes
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def print_with_loss_mask(tokenizer, input_ids, loss_mask, title=""):
    """Print text with loss_mask=1 (assistant) parts in RED."""
    input_ids = input_ids.flatten() if input_ids.dim() > 1 else input_ids
    loss_mask = loss_mask.flatten() if loss_mask.dim() > 1 else loss_mask

    print(f"\n{'=' * 60}")
    print(f"{title}")
    print("=" * 60)

    if len(input_ids) == 0:
        print("(empty)")
        return

    # Group consecutive tokens by loss_mask value
    current_mask = loss_mask[0].item()
    current_ids = [input_ids[0].item()]

    for i in range(1, len(input_ids)):
        if loss_mask[i].item() == current_mask:
            current_ids.append(input_ids[i].item())
        else:
            # Decode and print current group
            text = tokenizer.decode(current_ids, skip_special_tokens=False)
            if current_mask == 1:
                print(f"{RED}{text}{RESET}", end="")
            else:
                print(text, end="")
            current_ids = [input_ids[i].item()]
            current_mask = loss_mask[i].item()

    # Print remaining tokens
    if current_ids:
        text = tokenizer.decode(current_ids, skip_special_tokens=False)
        if current_mask == 1:
            print(f"{RED}{text}{RESET}")
        else:
            print(text)

    print("=" * 60)
    # Summary stats
    total = len(loss_mask)
    supervised = int(loss_mask.sum().item())
    print(f"Total tokens: {total}, Supervised: {supervised} ({100*supervised/total:.1f}%)")


# Tool definitions for tool-use test
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {
                        "type": "string",
                        "description": "The unit of temperature",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["location"],
            },
        },
    },
]

# Tool-use conversation from SpecForge
TOOL_USE_CONVERSATION = [
    {"role": "user", "content": "我想知道今天北京和上海的天气怎么样？"},
    {
        "role": "assistant",
        "content": "我来帮您查询北京和上海的天气情况。",
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": {"location": "北京", "date": "today"},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": {"location": "上海", "date": "today"},
                },
            },
        ],
    },
    {
        "role": "tool",
        "content": '{"location": "北京", "temperature": 25, "condition": "晴朗", "humidity": "45%"}',
    },
    {
        "role": "tool",
        "content": '{"location": "上海", "temperature": 28, "condition": "多云", "humidity": "65%"}',
    },
    {
        "role": "assistant",
        "content": "根据查询结果，北京今天晴朗，25°C；上海多云，28°C。两地都比较适合出行。",
    },
]

# Multi-turn conversation without tools
MULTI_TURN_CONVERSATION = [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."},
    {"role": "user", "content": "What about Germany?"},
    {"role": "assistant", "content": "The capital of Germany is Berlin."},
    {"role": "user", "content": "And Italy?"},
    {"role": "assistant", "content": "The capital of Italy is Rome."},
]

# Conversation with thinking content
THINKING_CONVERSATION = [
    {"role": "user", "content": "What is 15 * 23?"},
    {
        "role": "assistant",
        "content": "_filled_thinking_Let me calculate: 15 * 23 = 15 * 20 + 15 * 3 = 300 + 45 = 345<|im_end|>The answer is 345.",
    },
]


class TestBuildLossMaskWithTokenizer(unittest.TestCase):
    """Test loss mask computation with real Qwen3 tokenizer."""

    @classmethod
    def setUpClass(cls):
        # Try local path first, then fall back to HF
        local_path = "/mnt/hdfs/ccchang_hldy/Qwen3-8B"
        if os.path.exists(local_path):
            cls.model_name = local_path
        else:
            cls.model_name = "Qwen/Qwen3-8B"

        print(f"\n{YELLOW}Loading tokenizer from {cls.model_name}...{RESET}")
        cls.tokenizer = AutoTokenizer.from_pretrained(
            cls.model_name, trust_remote_code=True
        )
        cls.max_length = 4096

    def test_multi_turn_conversation_all_assistant_turns(self):
        """Test multi-turn conversation with all_assistant_turns mode."""
        result = build_loss_mask_for_conversation(
            tokenizer=self.tokenizer,
            messages=MULTI_TURN_CONVERSATION,
            max_length=self.max_length,
            loss_mask_mode="all_assistant_turns",
        )

        self.assertIsNotNone(result, "Result should not be None")
        self.assertIsInstance(result, LossMaskResult)

        # Verify shapes
        self.assertEqual(result.input_ids.dim(), 1)
        self.assertEqual(result.loss_mask.dim(), 1)
        self.assertEqual(len(result.input_ids), len(result.loss_mask))

        # Verify we have supervised tokens
        self.assertGreater(result.valid_tokens, 0)

        # Should have 3 assistant turns, all supervised
        print_with_loss_mask(
            self.tokenizer,
            result.input_ids,
            result.loss_mask,
            title="[all_assistant_turns] Multi-turn conversation (RED = supervised):",
        )

    def test_multi_turn_conversation_last_only(self):
        """Test multi-turn conversation with last_assistant_only mode."""
        result = build_loss_mask_for_conversation(
            tokenizer=self.tokenizer,
            messages=MULTI_TURN_CONVERSATION,
            max_length=self.max_length,
            loss_mask_mode="last_assistant_only",
        )

        self.assertIsNotNone(result)

        # Should have fewer supervised tokens than all_assistant_turns
        result_all = build_loss_mask_for_conversation(
            tokenizer=self.tokenizer,
            messages=MULTI_TURN_CONVERSATION,
            max_length=self.max_length,
            loss_mask_mode="all_assistant_turns",
        )

        self.assertLess(result.valid_tokens, result_all.valid_tokens)

        print_with_loss_mask(
            self.tokenizer,
            result.input_ids,
            result.loss_mask,
            title="[last_assistant_only] Multi-turn conversation (RED = supervised):",
        )

    def test_tool_use_conversation(self):
        """Test tool-use conversation with special formatting."""
        result = build_loss_mask_for_conversation(
            tokenizer=self.tokenizer,
            messages=TOOL_USE_CONVERSATION,
            max_length=self.max_length,
            loss_mask_mode="all_assistant_turns",
        )

        self.assertIsNotNone(result, "Result should not be None")

        # Verify we have supervised tokens
        self.assertGreater(result.valid_tokens, 0)

        print_with_loss_mask(
            self.tokenizer,
            result.input_ids,
            result.loss_mask,
            title="[tool_use] Tool-use conversation (RED = supervised):",
        )

    def test_thinking_conversation_detection(self):
        """Test thinking content detection."""
        # Should detect thinking content
        self.assertTrue(has_thinking_content(THINKING_CONVERSATION))

        # Should NOT detect thinking in normal conversation
        self.assertFalse(has_thinking_content(MULTI_TURN_CONVERSATION))

    def test_thinking_conversation_auto_mode(self):
        """Test auto mode with thinking content."""
        # auto mode should use last_turn_only when thinking detected
        result_auto = build_loss_mask_for_conversation(
            tokenizer=self.tokenizer,
            messages=THINKING_CONVERSATION,
            max_length=self.max_length,
            loss_mask_mode="auto",
        )

        result_last = build_loss_mask_for_conversation(
            tokenizer=self.tokenizer,
            messages=THINKING_CONVERSATION,
            max_length=self.max_length,
            loss_mask_mode="last_assistant_only",
        )

        # auto mode with thinking should behave like last_assistant_only
        self.assertEqual(result_auto.valid_tokens, result_last.valid_tokens)

        print_with_loss_mask(
            self.tokenizer,
            result_auto.input_ids,
            result_auto.loss_mask,
            title="[auto with thinking] (RED = supervised):",
        )

    def test_pack_unpack_round_trip(self):
        """Test pack/unpack preserves mask for real tokenized data."""
        result = build_loss_mask_for_conversation(
            tokenizer=self.tokenizer,
            messages=MULTI_TURN_CONVERSATION,
            max_length=self.max_length,
            loss_mask_mode="all_assistant_turns",
        )

        # Test pack/unpack directly on the loss_mask tensor
        from recipe.drafter_cotraining.loss_mask_utils import pack_loss_mask, unpack_loss_mask
        packed = pack_loss_mask(result.loss_mask)
        unpacked = unpack_loss_mask(packed)

        # Should match original
        self.assertTrue(
            torch.equal(result.loss_mask, unpacked),
            "Unpacked mask should match original"
        )

    def test_loss_mask_positions_are_assistant(self):
        """Verify that masked positions are actually assistant content."""
        result = build_loss_mask_for_conversation(
            tokenizer=self.tokenizer,
            messages=MULTI_TURN_CONVERSATION,
            max_length=self.max_length,
            loss_mask_mode="all_assistant_turns",
        )

        # Find positions where loss_mask == 1
        supervised_indices = torch.where(result.loss_mask == 1)[0]

        # Decode the supervised tokens
        supervised_tokens = result.input_ids[supervised_indices]
        decoded = self.tokenizer.decode(supervised_tokens, skip_special_tokens=False)

        # Should not contain user markers
        self.assertNotIn("<|im_start|>user", decoded)

        print(f"\n{GREEN}Supervised content sample:{RESET}")
        print(decoded[:200] + "..." if len(decoded) > 200 else decoded)


class TestEdgeCasesWithTokenizer(unittest.TestCase):
    """Test edge cases with real tokenizer."""

    @classmethod
    def setUpClass(cls):
        local_path = "/mnt/hdfs/ccchang_hldy/Qwen3-8B"
        if os.path.exists(local_path):
            cls.model_name = local_path
        else:
            cls.model_name = "Qwen/Qwen3-8B"

        cls.tokenizer = AutoTokenizer.from_pretrained(
            cls.model_name, trust_remote_code=True
        )

    def test_single_turn_conversation(self):
        """Test single turn conversation."""
        messages = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there! How can I help you?"},
        ]

        result = build_loss_mask_for_conversation(
            tokenizer=self.tokenizer,
            messages=messages,
            max_length=1024,
            loss_mask_mode="all_assistant_turns",
        )

        self.assertIsNotNone(result)
        self.assertGreater(result.valid_tokens, 0)

    def test_minimal_assistant_content(self):
        """Test conversation with minimal assistant content."""
        # Note: Qwen3 tokenizer injects thinking tags for empty content
        # So we test with minimal but non-empty content
        messages = [
            {"role": "user", "content": "Say 'ok'"},
            {"role": "assistant", "content": "ok"},
        ]

        result = build_loss_mask_for_conversation(
            tokenizer=self.tokenizer,
            messages=messages,
            max_length=1024,
            loss_mask_mode="all_assistant_turns",
        )

        self.assertIsNotNone(result)
        self.assertGreater(result.valid_tokens, 0)

    def test_truncation_preserves_alignment(self):
        """Test that truncation preserves input_ids/loss_mask alignment."""
        # Long conversation that will be truncated
        messages = [
            {"role": "user", "content": "Hello " * 100},
            {"role": "assistant", "content": "Hi there! " * 100},
        ]

        max_length = 256
        result = build_loss_mask_for_conversation(
            tokenizer=self.tokenizer,
            messages=messages,
            max_length=max_length,
            loss_mask_mode="all_assistant_turns",
        )

        self.assertIsNotNone(result)
        # Should be truncated to max_length
        self.assertLessEqual(len(result.input_ids), max_length)
        self.assertEqual(len(result.input_ids), len(result.loss_mask))


if __name__ == "__main__":
    unittest.main(verbosity=2)
