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
"""HSCollectorManager.

Clone of `TeacherModelManager` in colocated mode. Spawns a vLLM replica
pool for EAGLE hidden-states extraction. The model inside each replica
is the actor model with a Mooncake KV connector plugged in
(configured via `inference.engine_kwargs["vllm"]` in the config).

Called sync from the trainer after rollout (parallel to
`_compute_teacher_colocate` in `ray_trainer.py`).
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from omegaconf import DictConfig

from verl.base_config import BaseConfig
from verl.protocol import DataProto
from verl.single_controller.ray.base import RayResourcePool, split_resource_pool
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.ray_utils import auto_await
from verl.workers.config import HFModelConfig
from verl.workers.config.rollout import RolloutConfig
from verl.workers.rollout.replica import get_rollout_replica_class

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@dataclass
class HSCollectorConfig(BaseConfig):
    """Minimal config for the EAGLE hidden-states collector.

    Colocate-only: HS collector shares GPUs with the actor/rollout and is
    time-multiplexed via vLLM sleep/wake. The vLLM engine is configured with
    our Mooncake KV connector + `extract_hidden_states` speculative config
    via `inference.engine_kwargs["vllm"]`.
    """

    enabled: bool = False
    model_path: Optional[str] = None
    inference: RolloutConfig = field(default_factory=RolloutConfig)
    # Only used when running the manager standalone (no parent resource pool).
    n_gpus_per_node: int = 0
    nnodes: int = 0


class HSCollectorManager:
    """Owns the HS collector replica pool. Wake → infer → sleep per call."""

    def __init__(self, config: DictConfig, resource_pool: RayResourcePool = None):
        self.config: HSCollectorConfig = omega_conf_to_dataclass(config, dataclass_type=HSCollectorConfig)
        self.resource_pool = resource_pool
        self._check_mooncake_reachable(config)
        self._initialize_llm_servers()
        self._initialize_async_server_manager()
        self.sleep()

    @staticmethod
    def _check_mooncake_reachable(config: DictConfig) -> None:
        """Fail fast (<5s) if the mooncake master isn't reachable.

        Reads from the same kv_connector_extra_config path the vLLM connector reads.
        Skips (with a warning) when the config path isn't populated — preserves
        today's behavior for callers still relying on MOONCAKE_* env vars.
        """
        try:
            extra = (
                config.inference.engine_kwargs.get("vllm", {})
                .get("kv_transfer_config", {})
                .get("kv_connector_extra_config", {})
            ) or {}
        except Exception:
            extra = {}
        master_addr = extra.get("master_server_address")
        metadata_url = extra.get("metadata_server")
        if not master_addr or not metadata_url:
            logger.warning("HSCollectorManager: kv_connector_extra_config missing, skipping reachability probe")
            return
        from recipe.drafter_cotraining.mooncake.master import check_mooncake_master_available
        check_mooncake_master_available(master_addr, metadata_url)

    def _initialize_llm_servers(self):
        cfg = self.config
        inference_cfg = cfg.inference
        replica_world_size = (
            inference_cfg.tensor_model_parallel_size
            * inference_cfg.data_parallel_size
            * inference_cfg.pipeline_model_parallel_size
        )
        world_size = (
            self.resource_pool.world_size
            if self.resource_pool  # colocate
            else cfg.n_gpus_per_node * cfg.nnodes  # standalone
        )
        num_replicas = world_size // replica_world_size

        rollout_replica_class = get_rollout_replica_class(inference_cfg.name)
        model_config = HFModelConfig(path=cfg.model_path)

        self.rollout_replicas = [
            rollout_replica_class(
                replica_rank=replica_rank,
                config=inference_cfg,
                model_config=model_config,
                gpus_per_node=cfg.n_gpus_per_node if not self.resource_pool else 8,
                is_teacher_model=True,  # reuse teacher colocate plumbing in RolloutReplica
            )
            for replica_rank in range(num_replicas)
        ]
        if self.resource_pool:
            split_pools = split_resource_pool(self.resource_pool, split_size=replica_world_size)
            assert len(split_pools) == len(self.rollout_replicas)
            self._run_all(
                [server.init_colocated(pool) for server, pool in zip(self.rollout_replicas, split_pools, strict=True)]
            )
        else:
            self._run_all([server.init_standalone() for server in self.rollout_replicas])

        self.server_handles = [s._server_handle for s in self.rollout_replicas]
        self.server_addresses = [s._server_address for s in self.rollout_replicas]

    def _initialize_async_server_manager(self):
        from verl.experimental.agent_loop.agent_loop import GlobalRequestLoadBalancer

        from .hs_collector_manager import AsyncHSCollectorServerManager

        self.load_balancer_handle = GlobalRequestLoadBalancer.remote(server_actor_ids=self.server_addresses)
        self.server_manager = AsyncHSCollectorServerManager(
            config=self.config,
            servers=list(zip(self.server_addresses, self.server_handles, strict=True)),
            load_balancer_handle=self.load_balancer_handle,
        )

    def compute_hidden_states(self, data: DataProto) -> DataProto:
        """Wake replicas → prefill-only on each sample → sleep → return metadata DataProto."""
        self.wake_up()
        try:
            return self._run_single(self.server_manager.compute_hidden_states_batch(data))
        finally:
            self.sleep()

    def update_weights(self, params) -> None:
        """Push fresh actor weights to all HS-collector replicas (they mirror the actor).

        TODO: wire to verl's checkpoint_engine (same path used for actor → rollout).
        Teacher doesn't need this because its model is frozen; our HS collector mirrors
        the live actor and must re-sync after each update_actor().
        """
        pass

    @auto_await
    async def wake_up(self):
        await self._run_all([replica.wake_up() for replica in self.rollout_replicas])

    @auto_await
    async def sleep(self):
        await self._run_all([replica.sleep() for replica in self.rollout_replicas])

    @auto_await
    async def _run_all(self, tasks):
        await asyncio.gather(*tasks)

    def _run_single(self, task):
        async def run():
            return await task

        return asyncio.run(run())
