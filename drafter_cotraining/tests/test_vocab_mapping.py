"""Tests for the draft-vocab pruning mapping builder.

Verifies that ``process_token_dict_to_mappings`` and
``generate_vocab_mapping_file`` produce tensors with the contract expected
by ``Eagle3DraftModel.set_vocab_buffers`` and the EAGLE3 loss kernel:
  * ``t2d`` is bool of shape ``[V_target]`` with exactly ``V_draft`` Trues.
  * ``d2t`` is int64 of shape ``[V_draft]`` with ``d2t[i] + i == used_tokens[i]``
    for the sorted-ascending list of selected ids.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from collections import Counter

import torch

from recipe.drafter_cotraining.utils.vocab_mapping import (
    generate_vocab_mapping_file,
    load_vocab_mapping,
    process_token_dict_to_mappings,
)


class TestProcessTokenDictToMappings(unittest.TestCase):
    def test_top_k_selection_and_inverse(self):
        # Target vocab size 10, draft vocab size 4. Frequencies favor the
        # ids {2, 5, 7, 9}; the rest should be excluded.
        token_dict = Counter({0: 1, 1: 1, 2: 100, 3: 1, 5: 50, 7: 30, 9: 20})
        d2t, t2d = process_token_dict_to_mappings(
            token_dict, draft_vocab_size=4, target_vocab_size=10
        )

        self.assertEqual(t2d.dtype, torch.bool)
        self.assertEqual(t2d.shape, (10,))
        self.assertEqual(d2t.dtype, torch.int64)
        self.assertEqual(d2t.shape, (4,))

        used_expected = sorted({2, 5, 7, 9})
        self.assertEqual(int(t2d.sum()), 4)
        for token_id in used_expected:
            self.assertTrue(bool(t2d[token_id]))
        for token_id in (0, 1, 3, 4, 6, 8):
            self.assertFalse(bool(t2d[token_id]))

        # d2t[i] + i reconstructs the i-th sorted used token.
        for i, token_id in enumerate(used_expected):
            self.assertEqual(int(d2t[i]) + i, token_id)

    def test_padding_when_corpus_is_too_small(self):
        # Only one observed token but draft_vocab_size > 1 — we must pad
        # with unused ids so set_vocab_buffers shapes match.
        token_dict = Counter({3: 5})
        d2t, t2d = process_token_dict_to_mappings(
            token_dict, draft_vocab_size=3, target_vocab_size=8
        )
        self.assertEqual(int(t2d.sum()), 3)
        self.assertTrue(bool(t2d[3]))  # observed token survives top-K
        # d2t/t2d still satisfy the inverse contract on the sorted-ascending list.
        used = [i for i, b in enumerate(t2d.tolist()) if b]
        self.assertEqual(len(used), 3)
        for i, token_id in enumerate(used):
            self.assertEqual(int(d2t[i]) + i, token_id)


class TestGenerateVocabMappingFile(unittest.TestCase):
    def test_round_trip_to_disk(self):
        target_vocab_size = 16
        draft_vocab_size = 5

        # Synthesize 4 supervised samples; ids 1, 4, 7, 10, 13 dominate
        # the masked positions.
        samples = [
            (torch.tensor([0, 1, 4, 7, 10, 13, 99]),
             torch.tensor([0, 1, 1, 1, 1, 1, 0])),
            (torch.tensor([1, 4, 7]), torch.tensor([1, 1, 1])),
            (torch.tensor([10, 13]), torch.tensor([1, 1])),
            (torch.tensor([0, 1]), torch.tensor([0, 1])),  # one extra `1`
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "subdir", "mapping.pt")
            written = generate_vocab_mapping_file(
                samples=samples,
                target_vocab_size=target_vocab_size,
                draft_vocab_size=draft_vocab_size,
                output_path=out_path,
            )
            self.assertEqual(written, out_path)
            self.assertTrue(os.path.exists(out_path))

            d2t, t2d = load_vocab_mapping(out_path)
            self.assertEqual(t2d.shape, (target_vocab_size,))
            self.assertEqual(d2t.shape, (draft_vocab_size,))
            # All five high-frequency ids must be in the draft vocab.
            for token_id in (1, 4, 7, 10, 13):
                self.assertTrue(bool(t2d[token_id]))
            self.assertEqual(int(t2d.sum()), draft_vocab_size)

            used = [i for i, b in enumerate(t2d.tolist()) if b]
            for i, token_id in enumerate(used):
                self.assertEqual(int(d2t[i]) + i, token_id)

    def test_rejects_invalid_size(self):
        with self.assertRaises(ValueError):
            with tempfile.TemporaryDirectory() as tmpdir:
                generate_vocab_mapping_file(
                    samples=iter([]),
                    target_vocab_size=8,
                    draft_vocab_size=16,
                    output_path=os.path.join(tmpdir, "x.pt"),
                )


if __name__ == "__main__":
    unittest.main()
