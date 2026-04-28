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
import torch
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
        """Run prefill-only on each sample in `data`. Returns a DataProto with mooncake metadata."""
        tasks = []
        prompt_lens: list[int] = []
        response_lens: list[int] = []
        loss_masks: list[np.ndarray] = []
        valid_tokens_list: list[int] = []

        for i in range(len(data)):
            sequence_ids, plen, rlen = _unpad_sequence_ids(data[i : i + 1])
            prompt_lens.append(plen)
            response_lens.append(rlen)
            tasks.append(asyncio.create_task(self.compute_hidden_states_single(sequence_ids)))

            # Extract loss mask info if present (canonical format)
            nt = data.non_tensor_batch
            if "loss_mask" in nt and i < len(nt["loss_mask"]):
                lm = nt["loss_mask"][i]
                if isinstance(lm, np.ndarray):
                    loss_masks.append(lm)
                else:
                    loss_masks.append(np.array([], dtype=np.int64))
            else:
                loss_masks.append(np.array([], dtype=np.int64))

            if "valid_tokens" in nt and i < len(nt["valid_tokens"]):
                valid_tokens_list.append(int(nt["valid_tokens"][i]))
            else:
                valid_tokens_list.append(max(0, rlen - 1) if rlen > 1 else 0)

        results = await asyncio.gather(*tasks)

        return DataProto(
            non_tensor_batch={
                "hs_mooncake_keys": np.array([r["mooncake_key"] for r in results], dtype=object),
                "hs_shapes": np.array([r["shapes"] for r in results], dtype=object),
                "hs_dtypes": np.array([r["dtypes"] for r in results], dtype=object),
                "hs_seq_lens": np.array([r["seq_len"] for r in results], dtype=np.int64),
                # Valid (unpadded) prompt / response lengths per sample. Feeds the
                # drafter-side response-only loss_mask = [0]*prompt_len +
                # [1]*(response_len - 1) (final response position dropped — no
                # valid next-token target). Matches TorchSpec assistant-content
                # mask semantics + sgl_engine_decode.py:249 completion_tokens-1.
                "hs_prompt_lens": np.array(prompt_lens, dtype=np.int64),
                "hs_response_lens": np.array(response_lens, dtype=np.int64),
                # Loss mask for multi-turn supervision (canonical format)
                "loss_mask": np.array(loss_masks, dtype=object),
                "valid_tokens": np.array(valid_tokens_list, dtype=np.int64),
            },
        )


def _unpad_sequence_ids(data: DataProto) -> tuple[list[int], int, int]:
    """Extract (valid_tokens, prompt_len, response_len) from a single sample.

    Supports two formats:
    1. Canonical: input_ids + attention_mask + loss_mask
       - prompt_len = first position where loss_mask == 1
       - response_len = sum(loss_mask)
    2. Legacy: prompts + responses + input_ids + attention_mask
       - Uses prompt_width and response_width from batch
    """
    input_ids = data.batch["input_ids"][0]
    attention_mask = data.batch["attention_mask"][0]

    # Check for canonical format (loss_mask present, no prompts/responses)
    if "loss_mask" in data.batch and "prompts" not in data.batch:
        loss_mask = data.batch["loss_mask"][0]

        # Find prompt_len: first position where loss_mask == 1
        supervised_positions = torch.where(loss_mask == 1)[0]
        if len(supervised_positions) == 0:
            # No supervised tokens - use attention_mask to get sequence length
            seq_len = int(attention_mask.sum().item())
            tokens = input_ids[:seq_len].tolist()
            return tokens, seq_len, 0

        prompt_len = int(supervised_positions[0].item())
        response_len = int(loss_mask.sum().item())

        # Extract valid tokens (remove padding)
        seq_len = int(attention_mask.sum().item())
        tokens = input_ids[:seq_len].tolist()

        return tokens, prompt_len, response_len

    # Legacy format: prompts + responses
    prompt_width = data.batch["prompts"][0].shape[0]
    valid_prompt_length = int(attention_mask[:prompt_width].sum().item())
    valid_response_length = int(attention_mask[-data.batch["responses"][0].shape[0] :].sum().item())
    prompt_pad = prompt_width - valid_prompt_length
    tokens = input_ids[prompt_pad : prompt_width + valid_response_length].tolist()
    return tokens, valid_prompt_length, valid_response_length
