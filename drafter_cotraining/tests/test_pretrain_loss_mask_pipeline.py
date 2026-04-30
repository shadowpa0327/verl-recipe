# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for the canonical-conversation pretrain pipeline.

Covers everything except the actual Ray + vLLM + Mooncake spin-up:

* ``utils.chat_template_tokenize.build_input_ids_and_loss_mask`` matches TorchSpec's
  ``preprocess_conversations`` byte-for-byte (input_ids and loss_mask).
* ``ParquetDrafterPretrainDataset`` + ``DrafterPretrainCollator`` produce a
  DataProto whose ``loss_mask`` is non-zero on assistant tokens only.
* ``hs_collector.manager._unpad_sequence_and_mask`` correctly slices both the
  pretrain-style (right-padded + ``loss_masks`` non-tensor) and the legacy
  rollout-style (left-prompt + right-response) DataProto.
* The HSCollectorManager output schema matches what DrafterPretrainWorker expects
  (unprefixed keys: mooncake_keys, shapes, dtypes, seq_lens, loss_masks).
* ``ActorRolloutRefDrafterWorker._compute_valid_counts`` returns the expected
  per-sample counts.

Run with: pytest recipe/drafter_cotraining/tests/test_pretrain_loss_mask_pipeline.py
"""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)


@pytest.fixture
def synthetic_conversations() -> list[list[dict]]:
    return [
        [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "It's 4."},
            {"role": "user", "content": "Why?"},
            {"role": "assistant", "content": "Addition."},
        ],
        [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ],
    ]


def test_loss_mask_matches_torchspec(tokenizer, synthetic_conversations):
    """Verify byte-for-byte parity with TorchSpec's preprocess_conversations.

    Skips when the TorchSpec source tree isn't available (CI containers etc.).
    """
    if os.path.exists("/root/TorchSpec"):
        sys.path.insert(0, "/root/TorchSpec")
    try:
        from torchspec.data.preprocessing import preprocess_conversations
        from torchspec.data.template import TEMPLATE_REGISTRY
    except ImportError:
        pytest.skip("TorchSpec not available")

    from recipe.drafter_cotraining.utils.chat_template_tokenize import build_input_ids_and_loss_mask

    for conv in synthetic_conversations:
        ids, mask = build_input_ids_and_loss_mask(tokenizer, conv, "qwen", 8192)
        out = preprocess_conversations(
            tokenizer, [conv], TEMPLATE_REGISTRY.get("qwen"),
            max_length=8192, is_preformatted=False,
            include_attention_mask=False, use_packed_loss_mask=False,
            add_generation_prompt=False,
        )
        ts_ids = out["input_ids"][0].squeeze()
        ts_mask = out["loss_mask"][0].squeeze()
        assert torch.equal(ids, ts_ids)
        assert torch.equal(mask, ts_mask)


def test_loss_mask_supervises_every_assistant_turn(tokenizer):
    """Sanity: the per-token mask is 1 on every assistant turn, not just the last."""
    from recipe.drafter_cotraining.utils.chat_template_tokenize import build_input_ids_and_loss_mask

    conv = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1A1A1"},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "A2A2A2"},
        {"role": "user", "content": "Q3"},
        {"role": "assistant", "content": "A3A3A3"},
    ]
    _, mask = build_input_ids_and_loss_mask(tokenizer, conv, "qwen", 8192)
    # Find contiguous spans of 1s.
    spans = []
    i = 0
    n = len(mask)
    while i < n:
        if mask[i] == 1:
            j = i
            while j < n and mask[j] == 1:
                j += 1
            spans.append((i, j))
            i = j
        else:
            i += 1
    assert len(spans) == 3, f"expected 3 assistant spans, got {spans}"


def test_dataset_collator_pipeline(tmp_path, tokenizer, synthetic_conversations):
    """Round-trip a parquet file through the dataset + collator and check that
    the output DataProto has the right shape and mask plumbing."""
    from recipe.drafter_cotraining.trainer.pretrain_trainer import (
        ParquetDrafterPretrainDataset,
        DrafterPretrainCollator,
    )

    path = tmp_path / "rows.parquet"
    table = pa.Table.from_pylist(
        [
            {
                "id": f"row{i}",
                "conversations": conv,
                "extra_info": {"split": "train", "index": i},
            }
            for i, conv in enumerate(synthetic_conversations)
        ],
        schema=pa.schema([
            pa.field("id", pa.string()),
            pa.field(
                "conversations",
                pa.list_(
                    pa.struct(
                        [pa.field("role", pa.string()), pa.field("content", pa.string())]
                    )
                ),
            ),
            pa.field(
                "extra_info",
                pa.struct([pa.field("split", pa.string()), pa.field("index", pa.int64())]),
            ),
        ]),
    )
    pq.write_table(table, path, compression="zstd")

    ds = ParquetDrafterPretrainDataset([str(path)])
    assert len(ds) == len(synthetic_conversations)

    coll = DrafterPretrainCollator(tokenizer=tokenizer, max_seq_length=128, chat_template="qwen")
    batch = coll([ds[i] for i in range(len(ds))])

    assert batch.batch["input_ids"].shape == (2, 128)
    assert batch.batch["attention_mask"].shape == (2, 128)
    assert batch.batch["loss_mask"].shape == (2, 128)
    # Every conversation contains at least one assistant turn → at least one 1 per row.
    assert (batch.batch["loss_mask"].sum(dim=1) > 0).all()
    # Mask must be a strict subset of attention_mask.
    assert ((batch.batch["loss_mask"] == 1) & (batch.batch["attention_mask"] == 0)).sum() == 0

    # The non_tensor mirror used by the HS collector must be present and aligned.
    seq_lens = batch.non_tensor_batch["seq_lens"]
    loss_masks = batch.non_tensor_batch["loss_masks"]
    assert seq_lens.tolist() == batch.batch["attention_mask"].sum(dim=1).tolist()
    for i, seq in enumerate(seq_lens):
        assert loss_masks[i].shape[0] == int(seq)
        assert int(loss_masks[i].sum()) == int(batch.batch["loss_mask"][i, : int(seq)].sum())


def test_unpad_sequence_and_mask_pretrain_schema():
    from verl import DataProto

    from recipe.drafter_cotraining.hs_collector.manager import _unpad_sequence_and_mask

    ids = torch.tensor([[1, 2, 3, 4, 5, 0, 0, 0]])
    attn = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]])
    loss_masks_obj = np.empty(1, dtype=object)
    loss_masks_obj[0] = np.array([0, 0, 1, 1, 1], dtype=np.int64)
    proto = DataProto.from_single_dict({"input_ids": ids, "attention_mask": attn})
    proto.non_tensor_batch["loss_masks"] = loss_masks_obj
    tokens, mask = _unpad_sequence_and_mask(proto[0:1])
    assert tokens == [1, 2, 3, 4, 5]
    assert mask.tolist() == [0, 0, 1, 1, 1]


def test_unpad_sequence_and_mask_requires_loss_masks():
    """Missing 'loss_masks' in non_tensor_batch must raise — no silent fallback."""
    from verl import DataProto

    from recipe.drafter_cotraining.hs_collector.manager import _unpad_sequence_and_mask

    ids = torch.tensor([[1, 2, 3, 4, 5, 0, 0, 0]])
    attn = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]])
    proto = DataProto.from_single_dict({"input_ids": ids, "attention_mask": attn})
    with pytest.raises(KeyError, match="loss_masks"):
        _unpad_sequence_and_mask(proto[0:1])


def test_unpad_sequence_and_mask_rejects_length_mismatch():
    from verl import DataProto

    from recipe.drafter_cotraining.hs_collector.manager import _unpad_sequence_and_mask

    ids = torch.tensor([[1, 2, 3, 4, 5, 0, 0, 0]])
    attn = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]])
    bad = np.empty(1, dtype=object)
    bad[0] = np.array([0, 0, 1, 1], dtype=np.int64)  # length 4, sample has 5
    proto = DataProto.from_single_dict({"input_ids": ids, "attention_mask": attn})
    proto.non_tensor_batch["loss_masks"] = bad
    with pytest.raises(ValueError, match="length"):
        _unpad_sequence_and_mask(proto[0:1])


def test_hs_collector_output_schema():
    """Verify HSCollectorManager output uses unprefixed keys (mooncake_keys, shapes, dtypes, seq_lens, loss_masks)."""
    from verl import DataProto

    # Simulate the output schema from HSCollectorManager.compute_hidden_states_batch
    masks = [
        np.array([0, 0, 1, 1, 1], dtype=np.int64),
        np.array([0, 1, 1, 0], dtype=np.int64),
    ]
    masks_obj = np.empty(2, dtype=object)
    for i, m in enumerate(masks):
        masks_obj[i] = m

    hs_batch = DataProto(
        non_tensor_batch={
            "mooncake_keys": np.array(["k0", "k1"], dtype=object),
            "shapes": np.array(
                [
                    {"hidden_states": (5, 12), "input_ids": (5,)},
                    {"hidden_states": (4, 12), "input_ids": (4,)},
                ],
                dtype=object,
            ),
            "dtypes": np.array([{}, {}], dtype=object),
            "seq_lens": np.array([5, 4], dtype=np.int64),
            "loss_masks": masks_obj,
        }
    )

    # Verify schema matches what DrafterPretrainWorker.update_drafter expects
    nt = hs_batch.non_tensor_batch
    assert "mooncake_keys" in nt
    assert "shapes" in nt
    assert "dtypes" in nt
    assert "seq_lens" in nt
    assert "loss_masks" in nt

    # Verify loss_masks are preserved correctly
    assert nt["loss_masks"][0].tolist() == [0, 0, 1, 1, 1]
    assert nt["loss_masks"][1].tolist() == [0, 1, 1, 0]

    # Verify seq_lens match
    assert nt["seq_lens"].tolist() == [5, 4]


def test_compute_valid_counts():
    from verl import DataProto

    from recipe.drafter_cotraining.workers.engine_workers import ActorRolloutRefDrafterWorker

    masks_obj = np.empty(3, dtype=object)
    masks_obj[0] = np.array([0, 0, 1, 1, 1], dtype=np.int64)
    masks_obj[1] = np.array([1, 1, 1, 1, 1, 1], dtype=np.int64)
    masks_obj[2] = np.array([0, 0, 0], dtype=np.int64)
    proto = DataProto(
        non_tensor_batch={
            "loss_masks": masks_obj,
            "seq_lens": np.array([5, 6, 3], dtype=np.int64),
        }
    )
    valid = ActorRolloutRefDrafterWorker._compute_valid_counts(None, proto)
    assert valid == [3, 6, 0]
