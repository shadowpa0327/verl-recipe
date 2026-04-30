# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Async server manager for the HS collector.

Mirrors `AsyncTeacherLLMServerManager`. Sends prefill-only prompts
(max_tokens=1) and extracts `kv_transfer_params` (Mooncake keys +
tensor shapes/dtypes) from the response, instead of prompt logprobs.
"""

import asyncio
from typing import Any
from uuid import uuid4

import numpy as np
import ray
from omegaconf import DictConfig

from verl.experimental.agent_loop import AsyncLLMServerManager
from verl.protocol import DataProto


_SAMPLING_PARAMS = {"max_tokens": 1, "temperature": 0.0}


class AsyncHSCollectorServerManager(AsyncLLMServerManager):
    """Async client used to pull hidden states out of the HS-collector vLLM."""

    def __init__(
        self,
        config: DictConfig,
        servers: list[tuple[str, ray.actor.ActorHandle]],
        load_balancer_handle: ray.actor.ActorHandle,
    ):
        super().__init__(config=config, servers=servers, load_balancer_handle=load_balancer_handle)

    async def compute_hidden_states_single(self, sequence_ids: list[int]) -> dict[str, Any]:
        """Run prefill-only on one sequence. Returns dict with mooncake_key, shapes, dtypes."""
        output = await self.generate(
            request_id=uuid4().hex,
            prompt_ids=sequence_ids,
            sampling_params=_SAMPLING_PARAMS,
        )
        kv = output.extra_fields.get("kv_transfer_params") or {}
        return {
            "mooncake_key": kv.get("mooncake_key", ""),
            "shapes": kv.get("tensor_shapes", {}),
            "dtypes": kv.get("tensor_dtypes", {}),
            "seq_len": len(sequence_ids),
        }

    async def compute_hidden_states_batch(self, data: DataProto) -> DataProto:
        """Run prefill-only on each sample in `data`. Returns a DataProto with mooncake metadata.

        Pulls the per-sample loss_mask the trainer prepared (in
        ``data.non_tensor_batch['loss_masks']``) and forwards it unchanged
        in the output non_tensor_batch under ``hs_loss_masks``. The drafter
        worker reads that mask back when building the per-rank training
        batch — every assistant content token is supervised (1), everything
        else is 0. See ``recipe/drafter_cotraining/utils/chat_template_tokenize.py``.
        """
        tasks = []
        loss_masks: list[np.ndarray] = []
        for i in range(len(data)):
            sequence_ids, lm = _unpad_sequence_and_mask(data[i : i + 1])
            loss_masks.append(lm)
            tasks.append(asyncio.create_task(self.compute_hidden_states_single(sequence_ids)))
        results = await asyncio.gather(*tasks)

        loss_masks_obj = np.empty(len(loss_masks), dtype=object)
        for i, lm in enumerate(loss_masks):
            loss_masks_obj[i] = lm

        return DataProto(
            non_tensor_batch={
                "mooncake_keys": np.array([r["mooncake_key"] for r in results], dtype=object),
                "shapes": np.array([r["shapes"] for r in results], dtype=object),
                "dtypes": np.array([r["dtypes"] for r in results], dtype=object),
                "seq_lens": np.array([r["seq_len"] for r in results], dtype=np.int64),
                # Per-token assistant supervision mask, length = seq_len.
                # Carried through as object-dtype because seq_len varies across
                # samples (no rectangular tensor possible without padding).
                "loss_masks": loss_masks_obj,
            },
        )


def _unpad_sequence_and_mask(data: DataProto) -> tuple[list[int], np.ndarray]:
    """Slice a single right-padded row down to its valid prefix.

    Required schema (produced by ``DrafterPretrainCollator``):

    * ``data.batch["input_ids"]``  — ``[1, T_pad]`` right-padded with pad token
    * ``data.batch["attention_mask"]`` — ``[1, T_pad]``, 1 on real tokens
    * ``data.non_tensor_batch["loss_masks"][0]`` — ``np.ndarray`` of int64,
      length equals the sample's valid token count, 1 on every supervised
      (assistant content) position

    The per-token loss mask is *required*: callers must compute it upstream
    and hand it through. Missing or length-mismatched masks raise — the
    drafter pipeline supervises real assistant content only and silently
    falling back to all-ones / response-only would corrupt the loss.

    Returns ``(tokens, mask)``: raw token list and the int64 mask, both of
    length ``seq_len = attention_mask.sum()``.
    """
    input_ids = data.batch["input_ids"][0]
    attention_mask = data.batch["attention_mask"][0]
    seq_len = int(attention_mask.sum().item())
    tokens = input_ids[:seq_len].tolist()

    nt = data.non_tensor_batch
    if "loss_masks" not in nt:
        raise KeyError(
            "DataProto.non_tensor_batch is missing required key 'loss_masks'. "
            "The drafter pipeline requires a per-sample assistant loss_mask "
            "produced upstream (see DrafterPretrainCollator)."
        )
    raw = nt["loss_masks"][0]
    if isinstance(raw, np.ndarray):
        mask = raw.astype(np.int64).reshape(-1)
    else:
        mask = np.asarray(raw, dtype=np.int64).reshape(-1)
    if mask.shape[0] != seq_len:
        raise ValueError(
            f"loss_masks[0] has length {mask.shape[0]} but sample has "
            f"{seq_len} valid tokens; lengths must match exactly."
        )
    return tokens, mask
