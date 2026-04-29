# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""
Micro test for RayDrafterCTPPOTrainer: rollout + HS collection + drafter shape smoke.

Runs the same Hydra config as `recipe.drafter_cotraining.main_drafter_ct` —
including `ActorRolloutRefDrafterWorker`, `HSCollectorManager`, and
`DrafterDataController` — but replaces fit() with a short loop that
exits after N steps of:

    generate_sequences()
        → _compute_hidden_states_colocate()
        → _sample_metas_from_hs_batch()
        → _drafter_ctrl.push_samples()
        → _drafter_ctrl.drain_as_dataproto()
        → actor_rollout_wg.update_drafter()  (Mooncake fetch + collate + print padded shapes)

Skipped: teacher, reward, old_log_prob, ref_log_prob, advantage, critic
update, actor update (drafter forward/backward), weight sync, checkpoint save, validation.

Usage:
    # Uses the same Hydra config as main_drafter_ct_ppo.py by default
    # (verl/trainer/drafter/config/drafter_ct_trainer.yaml).
    # If that file doesn't exist yet, point at your own config dir via
    # Hydra's --config-path / --config-name flags:
    python scripts/test_drafter_rollout_hs.py \
        --config-path /path/to/configs --config-name drafter_ct_trainer \
        trainer.val_before_train=False \
        +micro.max_steps=2

Extra config keys (all optional, under `micro.*`):
    micro.max_steps          (int)   How many rollout+HS steps to run. Default 1.
    micro.launch_master      (bool)  Start mooncake_master as a Ray actor. Default True.

Mooncake connection info is read from the top-level `mooncake:` block
(master_server_address, metadata_server, etc.) — set there via YAML or CLI override.
"""

import os
import socket
import sys
from pprint import pprint

import hydra
import numpy as np
import ray
import torch

from verl import DataProto
from recipe.drafter_cotraining.trainer.ray_trainer import (
    RayDrafterCTPPOTrainer,
    _sample_metas_from_hs_batch,
)
from recipe.drafter_cotraining.main_drafter_ct import DrafterCTTaskRunner
from verl.trainer.main_ppo import run_ppo
from verl.utils.debug import marked_timer
from verl.utils.device import auto_set_device


class MicroRolloutHSOnlyTrainer(RayDrafterCTPPOTrainer):
    """Partial fit(): rollout + HS collection + controller push/drain. No training."""

    def fit(self):  # noqa: D401
        cfg_micro = self.config.get("micro", {}) or {}
        max_steps = int(cfg_micro.get("max_steps", 1))

        print("=" * 72)
        print(f"  Micro test: rollout + HS collection (max_steps={max_steps})")
        print(f"  use_hs_collector={self.use_hs_collector}")
        print(f"  hs_collector_manager={self.hs_collector_manager!r}")
        print("=" * 72)

        # Load actor weights so rollout + HS extraction use real params.
        self._load_checkpoint()
        self.checkpoint_manager.update_weights(0)

        steps_done = 0
        for batch_dict in self.train_dataloader:
            if steps_done >= max_steps:
                break
            metrics: dict = {}
            timing_raw: dict = {}

            batch = DataProto.from_single_dict(batch_dict)
            batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
            import uuid

            batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
            )

            gen_batch = self._get_gen_batch(batch)
            gen_batch.meta_info["global_steps"] = steps_done + 1
            gen_batch_output = gen_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
            )

            with marked_timer("gen", timing_raw, color="red"):
                gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
                self.checkpoint_manager.sleep_replicas()
                timing_raw.update(gen_batch_output.meta_info.get("timing", {}) or {})
                gen_batch_output.meta_info.pop("timing", None)

            batch = batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
            )
            batch = batch.union(gen_batch_output)

            if not self._should_compute_hidden_states_colocate(batch):
                print("  WARN: hs_collector is disabled — enable config.hs_collector.enabled=True")
                break

            with marked_timer("hs_collect", timing_raw, color="cyan"):
                hs_batch = self._compute_hidden_states_colocate(batch)
                sample_metas = _sample_metas_from_hs_batch(hs_batch)
                self._drafter_ctrl.push_samples(sample_metas)
                drafter_proto = self._drafter_ctrl.drain_as_dataproto()

            print(f"\n── Step {steps_done + 1}/{max_steps} ──")
            print(f"  batch size (post-repeat): {len(batch)}")
            print(f"  hs_batch non_tensor keys: {list(hs_batch.non_tensor_batch.keys())}")
            print(f"  sample_metas: {len(sample_metas)}")

            # Summary across all samples: keys non-empty, unique, consistent dtypes + tensor names.
            keys = [m.mooncake_key for m in sample_metas]
            empty = [i for i, k in enumerate(keys) if not k]
            unique = len(set(keys))
            seq_lens = [m.seq_len for m in sample_metas]
            tensor_names = sorted(sample_metas[0].shapes.keys()) if sample_metas else []
            all_same_tensors = all(sorted(m.shapes.keys()) == tensor_names for m in sample_metas)
            print(
                f"  keys: {len(keys)} total, {len(keys) - len(empty)} non-empty, "
                f"{unique} unique, empty_indices={empty}"
            )
            print(
                f"  seq_lens: min={min(seq_lens) if seq_lens else 0} "
                f"max={max(seq_lens) if seq_lens else 0} "
                f"mean={sum(seq_lens) / max(len(seq_lens), 1):.1f}"
            )
            print(f"  tensor names (consistent across samples={all_same_tensors}): {tensor_names}")
            for i, m in enumerate(sample_metas[: min(3, len(sample_metas))]):
                print(
                    f"  sample[{i}]: key={m.mooncake_key!r} seq_len={m.seq_len} "
                    f"shapes={m.shapes} dtypes={m.dtypes}"
                )
            print(
                "  controller status: "
                f"{self._drafter_ctrl.get_status()} "
                f"drafter_proto={'None' if drafter_proto is None else len(drafter_proto)}"
            )
            print("  timing (s):")
            for k, v in timing_raw.items():
                print(f"    {k:<24s} {v:.3f}")

            # Dispatch drained DataProto to the drafter mesh (each DP rank gets its
            # shard of mooncake keys). update_drafter fetches tensors from Mooncake,
            # runs the collator, and prints per-rank padded-shape lines.
            drafter_cfg = self.config.actor_rollout_ref.get("drafter", {}) or {}
            if drafter_proto is not None and drafter_cfg.get("enable", False):
                from omegaconf import OmegaConf
                drafter_proto.meta_info["mooncake_cfg"] = OmegaConf.to_container(
                    self.config.mooncake, resolve=True
                )
                with marked_timer("drafter_dispatch", timing_raw, color="magenta"):
                    self.actor_rollout_wg.update_drafter(drafter_proto)
                print(f"  drafter_dispatch: {timing_raw['drafter_dispatch']:.3f}s")

            # Rollout was slept after generate_sequences. In HYBRID mode
            # wake_up_replicas() is rejected; the sanctioned path is
            # checkpoint_manager.update_weights(), which (with backend=naive)
            # is effectively a no-op sync that leaves the rollout awake.
            steps_done += 1
            if steps_done < max_steps:
                self.checkpoint_manager.update_weights(steps_done)

        print("\n" + "=" * 72)
        print(f"  Done. Completed {steps_done} step(s). No drafter/actor training run.")
        print("=" * 72)


class MicroDrafterCTTaskRunner(DrafterCTTaskRunner):
    """Same wiring as DrafterCTTaskRunner, but uses the micro trainer."""

    def run(self, config):
        from omegaconf import OmegaConf

        from verl.trainer.ppo.utils import need_critic, need_reference_policy
        from verl.utils.config import validate_config
        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.utils.fs import copy_to_local
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        print(f"MicroDrafterCTTaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)
        self.add_reward_model_resource_pool(config)
        self.add_ref_policy_worker(config, actor_rollout_cls)

        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(config),
            use_critic=need_critic(config),
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

        trainer = MicroRolloutHSOnlyTrainer(
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
    auto_set_device(config)

    # Optionally launch our own mooncake_master as a Ray actor. Requires Ray up.
    micro_cfg = config.get("micro", {}) or {}
    if bool(micro_cfg.get("launch_master", True)):
        if not ray.is_initialized():
            ray.init()
        from types import SimpleNamespace
        from recipe.drafter_cotraining.mooncake.master import launch_mooncake_master

        # Parse metadata port from URL (e.g., "http://localhost:8090/metadata")
        # For P2PHANDSHAKE mode, use a dummy port (master will ignore it)
        metadata_server = config.mooncake.metadata_server
        if metadata_server == "P2PHANDSHAKE":
            metadata_port = 8090  # Dummy port, ignored in IPv6-only mode
        else:
            try:
                metadata_port = int(metadata_server.rsplit(":", 1)[1].split("/")[0])
            except (ValueError, IndexError):
                metadata_port = 8090

        args = SimpleNamespace(
            mooncake_master_server_address=config.mooncake.master_server_address,
            mooncake_metadata_port=metadata_port,
            mooncake_kv_lease_ttl_s=float(config.mooncake.kv_lease_ttl_s),
        )
        if launch_mooncake_master(args) is None:
            sys.exit("mooncake_master failed to launch (binary missing?)")

    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(MicroDrafterCTTaskRunner))


if __name__ == "__main__":
    main()
