"""
DrafterDataController — driver-side coordinator for the 3-level drafter data pipeline.

Mirrors TorchSpec's AsyncTrainingController but synchronous and in-process.
Lives on RayPPOTrainer (the driver). Owns Levels 1 & 2:

  Level 1: raw_prompts   — complete sequences from rollout
  Level 2: sample_pool   — training-ready samples (metadata + Mooncake keys)

Level 3 (per-rank dispatch) is handled by verl's mesh-based DataProto.chunk()
when update_drafter() is called with the drafter mesh.

Data flow:
  Rollout output     →  push_raw_prompts()
  HS Collector       ←  pull_raw_prompts()
                     →  push_samples()
  Drafter training   ←  drain_as_dataproto()  →  mesh dispatch splits per rank
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from verl.protocol import DataProto

logger = logging.getLogger(__name__)


@dataclass
class SequenceMeta:
    """Metadata for a complete sequence (prompt + response) from rollout.
    Lightweight — lives in Level 1 (raw_prompts) on the driver."""

    input_ids: Any  # np.ndarray or torch.Tensor, shape [seq_len]
    attention_mask: Any  # np.ndarray or torch.Tensor, shape [seq_len]
    prompt_len: int = 0
    response_len: int = 0


@dataclass
class SampleMeta:
    """Metadata for a training-ready sample after HS collection.
    Lightweight — lives in Level 2 (sample_pool) on the driver.
    Actual tensors (hidden states) are in Mooncake, referenced by key.

    ``prompt_len`` and ``response_len`` are the unpadded valid lengths of
    each span in the stored Mooncake tokens (total length = prompt + response).
    Consumed by ``update_drafter`` to build a response-only
    ``loss_mask = [0]*prompt_len + [1]*(response_len - 1)`` (final response
    position dropped — no valid next-token target).

    For canonical format with multi-turn loss masks:
    - ``loss_mask`` contains the numpy array (unpadded) for multi-turn supervision
    - ``valid_tokens`` is the count of supervised positions
    - ``prompt_len`` and ``response_len`` are set to 0 for backward compatibility
    """

    mooncake_key: str
    shapes: Dict[str, Tuple[int, ...]]
    dtypes: Dict[str, Any]  # torch.dtype stored as string for serialization
    seq_len: int = 0
    n_tokens: int = 0
    prompt_len: int = 0
    response_len: int = 0
    loss_mask: Optional[np.ndarray] = None  # Unpadded loss mask (canonical format)
    valid_tokens: int = 0  # Count of supervised positions (from loss_mask.sum())


class DrafterDataController:
    """
    Driver-side coordinator for the 3-level drafter data pipeline.

    Owns Levels 1 & 2. Level 3 dispatch is handled by verl's mesh-based
    DataProto.chunk() — the controller packs metadata into DataProto and
    the dispatch fn on update_drafter() splits it per DP rank automatically.

    Usage in RayPPOTrainer.fit():

        # Phase 1: Rollout → raw_prompts
        self._drafter_ctrl.push_raw_prompts(sequences)

        # Phase 2: raw_prompts → HS collection → sample_pool
        raw = self._drafter_ctrl.pull_raw_prompts()
        sample_metadata = actor_rollout_wg.collect_hidden_states(raw)
        self._drafter_ctrl.push_samples(sample_metadata)

        # Dispatch + train (mesh dispatch splits per rank)
        drafter_proto = self._drafter_ctrl.drain_as_dataproto()
        actor_rollout_wg.update_drafter(drafter_proto)
    """

    def __init__(self, dp_size: int):
        self._raw_prompts: List[SequenceMeta] = []  # Level 1 (global)
        self._sample_pool: List[SampleMeta] = []  # Level 2 (global)
        self._dp_size = dp_size

    # ── Level 1: raw_prompts ──────────────────────────────────

    def push_raw_prompts(self, sequences: List[SequenceMeta]):
        """Called by: driver after generate_sequences(). Fills Level 1."""
        self._raw_prompts.extend(sequences)
        logger.debug("push_raw_prompts: %d sequences (total: %d)",
                      len(sequences), len(self._raw_prompts))

    def pull_raw_prompts(self) -> List[SequenceMeta]:
        """Called by: driver, to send to HS collector workers. Drains Level 1."""
        batch = self._raw_prompts
        self._raw_prompts = []
        logger.debug("pull_raw_prompts: returning %d sequences", len(batch))
        return batch

    @property
    def raw_prompts_size(self) -> int:
        return len(self._raw_prompts)

    # ── Level 2: sample_pool ──────────────────────────────────

    def push_samples(self, samples: List[SampleMeta]):
        """Called by: driver after HS collection returns. Fills Level 2."""
        self._sample_pool.extend(samples)
        logger.debug("push_samples: %d samples (total: %d)",
                      len(samples), len(self._sample_pool))

    @property
    def sample_pool_size(self) -> int:
        return len(self._sample_pool)

    # ── Drain Level 2 → DataProto for mesh dispatch ──────────

    def drain_as_dataproto(self) -> Optional[DataProto]:
        """
        Pack all samples into DataProto.non_tensor_batch for mesh dispatch.

        The dispatch fn on update_drafter() (using drafter mesh) calls
        proto.chunk(dp_size), which uses np.array_split on axis 0 to
        split non_tensor_batch values per DP rank.

        non_tensor_batch values must be np.ndarray(dtype=object).

        Returns None if sample_pool is empty.
        """
        if not self._sample_pool:
            logger.debug("drain_as_dataproto: pool empty, returning None")
            return None

        n = len(self._sample_pool)
        logger.debug("drain_as_dataproto: packing %d samples into DataProto", n)

        proto = DataProto(
            non_tensor_batch={
                'mooncake_keys': np.array(
                    [m.mooncake_key for m in self._sample_pool], dtype=object
                ),
                'shapes': np.array(
                    [m.shapes for m in self._sample_pool], dtype=object
                ),
                'dtypes': np.array(
                    [m.dtypes for m in self._sample_pool], dtype=object
                ),
                'seq_lens': np.array(
                    [m.seq_len for m in self._sample_pool], dtype=object
                ),
                'n_tokens': np.array(
                    [m.n_tokens for m in self._sample_pool], dtype=object
                ),
                'prompt_lens': np.array(
                    [m.prompt_len for m in self._sample_pool], dtype=object
                ),
                'response_lens': np.array(
                    [m.response_len for m in self._sample_pool], dtype=object
                ),
                'loss_mask': np.array(
                    [m.loss_mask for m in self._sample_pool], dtype=object
                ),
                'valid_tokens': np.array(
                    [m.valid_tokens for m in self._sample_pool], dtype=np.int64
                ),
            },
        )

        self._sample_pool = []
        return proto

    # ── Status ────────────────────────────────────────────────

    def get_status(self) -> Dict[str, int]:
        return {
            "raw_prompts": len(self._raw_prompts),
            "sample_pool": len(self._sample_pool),
            "dp_size": self._dp_size,
        }
