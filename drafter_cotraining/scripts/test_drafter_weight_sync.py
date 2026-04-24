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
"""verl-native E2E acceptance test for drafter weight sync.

This is intentionally not the vLLM-native ``LLM(...)`` test. It launches the
normal verl async rollout stack with speculative decoding configured through
``actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config`` and then
drives the drafter mutation through:

    ActorRolloutRefDrafterWorker.update_rollout_drafter_weights_from_snapshot
        -> ServerAdapter.update_drafter_weights
        -> update_weights_from_ipc(target_model="drafter")
        -> BucketedWeightSender / BucketedWeightReceiver
        -> drafter.model.load_weights

Behavior:
    1. Llama-3.x target + EAGLE3 drafter from speculative_config: one rollout
       inference batch, acceptance should be good.
    2. Random EAGLE3 drafter weights sent through verl IPC: one rollout
       inference batch, acceptance should drop sharply.
    3. Original EAGLE3 drafter weights restored through verl IPC: one rollout
       inference batch, acceptance should recover.

Example:
    python recipe/drafter_cotraining/scripts/test_drafter_weight_sync.py \
        actor_rollout_ref.model.path=/path/to/Llama-3.1-8B-Instruct \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.mode=async \
        actor_rollout_ref.rollout.disable_log_stats=False \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config.method=eagle3 \
        actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config.model=/path/to/EAGLE3-LLaMA3.1-Instruct-8B \
        actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config.num_speculative_tokens=3 \
        trainer.val_before_train=False \
        +micro.drop_ratio=0.7 \
        +micro.recover_ratio=0.9
"""

import os
import socket
import sys
import uuid
from pprint import pprint

import hydra
import numpy as np
import ray

from verl import DataProto
from recipe.drafter_cotraining.scripts.test_drafter_rollout_hs import (
    MicroDrafterCTTaskRunner,
    MicroRolloutHSOnlyTrainer,
)
from verl.trainer.main_ppo import run_ppo
from verl.utils.device import auto_set_device


def _sum_counters(counter_reports: list[dict[str, int]]) -> dict[str, int]:
    return {
        "num_drafts": sum(int(r.get("num_drafts", 0)) for r in counter_reports),
        "num_accepted_tokens": sum(int(r.get("num_accepted_tokens", 0)) for r in counter_reports),
    }


def _counter_delta(after: dict[str, int], before: dict[str, int]) -> dict[str, int]:
    return {
        "num_drafts": after["num_drafts"] - before["num_drafts"],
        "num_accepted_tokens": after["num_accepted_tokens"] - before["num_accepted_tokens"],
    }


def _acceptance_length(delta: dict[str, int]) -> float:
    if delta["num_drafts"] <= 0:
        raise RuntimeError(
            "No speculative drafts were recorded. Ensure rollout speculative_config is set and active."
        )
    return 1.0 + (delta["num_accepted_tokens"] / delta["num_drafts"])


class DrafterWeightSyncSmokeTrainer(MicroRolloutHSOnlyTrainer):
    """Run baseline/random/restore using verl rollout and verl IPC weight sync."""

    def _read_spec_decode_counters(self) -> dict[str, int]:
        reports = ray.get(
            [handle.get_spec_decode_counters.remote() for handle in self.async_rollout_manager.server_handles]
        )
        return _sum_counters(reports)

    def _collective_rpc(self, method: str):
        nested = ray.get(
            [handle.collective_rpc.remote(method=method) for handle in self.async_rollout_manager.server_handles]
        )
        flattened = []
        for item in nested:
            if isinstance(item, list):
                flattened.extend(item)
            else:
                flattened.append(item)
        return flattened

    def _probe_target_norms(self) -> list[dict]:
        return self._collective_rpc("probe_target_param_norms")

    def _assert_target_norms_stable(self, before: list[dict], after: list[dict], tol: float):
        for b, a in zip(before, after, strict=False):
            for key in ("target_embed_norm", "target_lm_head_norm"):
                if b.get(key) is None or a.get(key) is None:
                    continue
                delta = abs(float(a[key]) - float(b[key]))
                if delta > tol:
                    raise AssertionError(
                        f"{key} changed on rank {a.get('rank')}: before={b[key]:.6f}, "
                        f"after={a[key]:.6f}, delta={delta:.6f}, tol={tol}"
                    )

    def _next_generation_batch(self, phase_idx: int) -> DataProto:
        if not hasattr(self, "_weight_sync_data_iter"):
            self._weight_sync_data_iter = iter(self.train_dataloader)

        try:
            batch_dict = next(self._weight_sync_data_iter)
        except StopIteration:
            self._weight_sync_data_iter = iter(self.train_dataloader)
            batch_dict = next(self._weight_sync_data_iter)

        batch = DataProto.from_single_dict(batch_dict)
        batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
        batch.non_tensor_batch["uid"] = np.array(
            [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
        )

        gen_batch = self._get_gen_batch(batch)
        gen_batch.meta_info["global_steps"] = phase_idx
        return gen_batch.repeat(
            repeat_times=self.config.actor_rollout_ref.rollout.n,
            interleave=True,
        )

    def _decode_first_response(self, output: DataProto) -> str:
        if "responses" not in output.batch:
            return ""
        response_ids = output.batch["responses"][0].detach().cpu().tolist()
        pad_id = self.tokenizer.pad_token_id
        if pad_id is not None:
            response_ids = [token_id for token_id in response_ids if token_id != pad_id]
        return self.tokenizer.decode(response_ids, skip_special_tokens=True)

    def _run_phase(self, name: str, phase_idx: int) -> tuple[float, list[dict]]:
        before = self._read_spec_decode_counters()
        output = self.async_rollout_manager.generate_sequences(self._next_generation_batch(phase_idx))
        after = self._read_spec_decode_counters()
        delta = _counter_delta(after, before)
        acceptance = _acceptance_length(delta)
        norms = self._probe_target_norms()

        sample = self._decode_first_response(output).replace("\n", " ")[:200]
        print(f"\n=== {name} ===")
        print(
            f"acceptance_length={acceptance:.4f} "
            f"drafts={delta['num_drafts']} accepted={delta['num_accepted_tokens']}"
        )
        for norm in norms:
            print(
                f"  [rank {norm.get('rank')}] "
                f"target_embed_norm={norm.get('target_embed_norm')} "
                f"target_lm_head_norm={norm.get('target_lm_head_norm')}"
            )
        print(f"sample output: {sample!r}")
        return acceptance, norms

    def _sync_rollout_drafter_from_snapshot(self, mode: str, seed: int) -> list[dict]:
        method = "update_rollout_drafter_weights_from_snapshot"
        if hasattr(self.actor_rollout_wg, method):
            refs = getattr(self.actor_rollout_wg, method)(mode=mode, seed=seed)
        else:
            # Colocated WorkerDict exposes methods with the spawned role prefix.
            prefix = getattr(self.actor_rollout_wg, "sub_cls_name", "")
            prefixed_method = f"{prefix}_{method}" if prefix else method
            refs = self.actor_rollout_wg.execute_all_async(prefixed_method, mode=mode, seed=seed)

        reports = ray.get(refs)
        if not isinstance(reports, list):
            reports = [reports]
        failures = [r for r in reports if not isinstance(r, dict) or not r.get("ok", False)]
        if failures:
            raise RuntimeError(f"Failed to {mode} rollout drafter weights: {failures}")
        return reports

    def _validate_rollout_speculative_config(self):
        from omegaconf import OmegaConf

        spec_config = OmegaConf.select(
            self.config, "actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config"
        )
        if not spec_config:
            raise ValueError(
                "Set actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config.* "
                "for this verl-native test. Required keys include method=eagle3, "
                "model=/path/to/EAGLE3, and num_speculative_tokens."
            )

        drafter_cfg = self.config.actor_rollout_ref.get("drafter", {}) or {}
        if drafter_cfg.get("enable", False):
            raise ValueError(
                "This smoke keeps the rollout drafter initialized from vLLM speculative_config. "
                "Run with actor_rollout_ref.drafter.enable=False so checkpoint_manager.update_weights "
                "does not overwrite the baseline drafter before phase 1."
            )

    def fit(self):  # noqa: D401
        cfg_micro = self.config.get("micro", {}) or {}
        drop_ratio = float(cfg_micro.get("drop_ratio", 0.7))
        recover_ratio = float(cfg_micro.get("recover_ratio", 0.9))
        norm_tol = float(cfg_micro.get("target_norm_tol", 1e-4))
        random_seed = int(cfg_micro.get("randomize_seed", 2026))

        self._validate_rollout_speculative_config()

        print("=" * 72)
        print("  verl-native drafter weight-sync smoke")
        print(f"  drop_ratio={drop_ratio} recover_ratio={recover_ratio} target_norm_tol={norm_tol}")
        print("=" * 72)

        # Sync the target model through the normal trainer -> rollout path. The
        # drafter training engine is intentionally disabled, so the rollout
        # drafter remains the EAGLE3 model loaded from speculative_config.
        self._load_checkpoint()
        self.checkpoint_manager.update_weights(0)

        if len(self.async_rollout_manager.server_handles) != 1:
            raise NotImplementedError(
                "test_drafter_weight_sync.py currently expects one rollout replica. "
                "TP within that replica is fine; multi-replica snapshot fan-out needs "
                "an explicit per-replica snapshot/install map."
            )

        sharing_reports = self._collective_rpc("inspect_drafter_sharing")
        print("\n=== Drafter / target sharing ===")
        for report in sharing_reports:
            print(report)

        # update_rollout_drafter_weights_from_snapshot self-caches on first
        # call — no separate snapshot/install step needed.

        baseline, baseline_norms = self._run_phase("Phase 1: baseline EAGLE3", 1)

        self._sync_rollout_drafter_from_snapshot(mode="random", seed=random_seed)
        randomized, randomized_norms = self._run_phase("Phase 2: random EAGLE3 weights via verl IPC", 2)
        self._assert_target_norms_stable(baseline_norms, randomized_norms, norm_tol)

        self._sync_rollout_drafter_from_snapshot(mode="restore", seed=random_seed)
        restored, restored_norms = self._run_phase("Phase 3: restored EAGLE3 weights via verl IPC", 3)
        self._assert_target_norms_stable(baseline_norms, restored_norms, norm_tol)

        drop_ok = randomized <= baseline * drop_ratio
        recover_ok = restored >= baseline * recover_ratio
        ordering_ok = restored > randomized

        print("\n=== Result ===")
        print(f"baseline={baseline:.4f} randomized={randomized:.4f} restored={restored:.4f}")
        print(f"drop_ok={drop_ok} recover_ok={recover_ok} ordering_ok={ordering_ok}")

        assert drop_ok and recover_ok and ordering_ok, (
            "Drafter weight sync acceptance checks failed: "
            f"baseline={baseline:.4f}, randomized={randomized:.4f}, restored={restored:.4f}"
        )


class DrafterWeightSyncTaskRunner(MicroDrafterCTTaskRunner):
    """Same wiring as the drafter micro runner, but with the weight-sync trainer."""

    def run(self, config):
        from omegaconf import OmegaConf

        from verl.utils.config import validate_config
        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.utils.fs import copy_to_local
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        print(f"DrafterWeightSyncTaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)

        validate_config(
            config=config,
            use_reference_policy=False,
            use_critic=False,
        )

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )

        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        resource_pool_manager = self.init_resource_pool_mgr(config)

        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        trainer = DrafterWeightSyncSmokeTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        trainer.init_workers()
        trainer.fit()


@hydra.main(
    config_path="../config",
    config_name="drafter_ct_trainer",
    version_base=None,
)
def main(config):
    from omegaconf import OmegaConf

    auto_set_device(config)

    # This smoke does not use the HS collector or the training drafter. The
    # rollout drafter must remain the EAGLE3 model loaded by vLLM from
    # speculative_config until the script snapshots it.
    OmegaConf.update(config, "hs_collector.enabled", False, merge=False)
    OmegaConf.update(config, "actor_rollout_ref.drafter.enable", False, merge=False)
    OmegaConf.update(config, "actor_rollout_ref.rollout.disable_log_stats", False, merge=False)
    OmegaConf.update(config, "critic.enable", False, merge=False)
    OmegaConf.update(config, "reward.reward_model.enable", False, merge=False)
    OmegaConf.update(config, "algorithm.use_kl_in_reward", False, merge=False)
    OmegaConf.update(config, "actor_rollout_ref.actor.use_kl_loss", False, merge=False)

    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(DrafterWeightSyncTaskRunner))


if __name__ == "__main__":
    main()
