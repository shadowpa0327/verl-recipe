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
        """Run prefill-only on each sample in `data`. Returns a DataProto with mooncake metadata."""
        tasks = []
        for i in range(len(data)):
            sequence_ids = _unpad_sequence_ids(data[i : i + 1])
            tasks.append(asyncio.create_task(self.compute_hidden_states_single(sequence_ids)))
        results = await asyncio.gather(*tasks)

        return DataProto(
            non_tensor_batch={
                "hs_mooncake_keys": np.array([r["mooncake_key"] for r in results], dtype=object),
                "hs_shapes": np.array([r["shapes"] for r in results], dtype=object),
                "hs_dtypes": np.array([r["dtypes"] for r in results], dtype=object),
                "hs_seq_lens": np.array([r["seq_len"] for r in results], dtype=np.int64),
            },
        )


def _unpad_sequence_ids(data: DataProto) -> list[int]:
    """Extract the valid (unpadded) prompt+response token ids from a single sample.

    Left-padded prompt concatenated with right-padded response, same layout as teacher.
    """
    input_ids = data.batch["input_ids"][0]
    attention_mask = data.batch["attention_mask"][0]
    prompt_width = data.batch["prompts"][0].shape[0]
    valid_prompt_length = int(attention_mask[:prompt_width].sum().item())
    valid_response_length = int(attention_mask[-data.batch["responses"][0].shape[0] :].sum().item())
    prompt_pad = prompt_width - valid_prompt_length
    return input_ids[prompt_pad : prompt_width + valid_response_length].tolist()
