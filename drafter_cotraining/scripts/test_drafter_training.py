# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Close-loop drafter training smoke — target model frozen.

Pipeline per step:
    raw prompts -> rollout -> HS (Mooncake) -> mesh dispatch
        -> update_drafter: fetch + collate + Eagle3 forward + 0.8^i backward + optimizer.step()

Skipped for this milestone: teacher, reward, advantage, critic/actor update,
drafter->rollout weight sync, checkpoint save, validation. See
`tasks/drafter-training-milestone.md` for scope and pass criteria.

Pass signal:
    train/loss_weighted trends down
    train/simulated_acc_len trends up
    No NaN/Inf, no OOM, no Mooncake key leak.

Usage (via wrapper):
    scripts/run_drafter_training.sh

Or directly (target_model_path defaults to actor_rollout_ref.model.path):
    python recipe/drafter_cotraining/scripts/test_drafter_training.py \
        actor_rollout_ref.drafter.enable=True \
        actor_rollout_ref.drafter.optimizer_config.total_training_steps=625 \
        data.train_max_samples=1000 \
        trainer.total_epochs=5
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
from recipe.drafter_cotraining.trainer.ray_trainer import _sample_metas_from_hs_batch
from recipe.drafter_cotraining.scripts.test_drafter_rollout_hs import (
    MicroDrafterCTTaskRunner,
    MicroRolloutHSOnlyTrainer,
)
from verl.trainer.main_ppo import run_ppo
from verl.utils.debug import marked_timer
from verl.utils.device import auto_set_device


class DrafterTrainingSmokeTrainer(MicroRolloutHSOnlyTrainer):
    """Full close-loop drafter training smoke."""

    def fit(self):  # noqa: D401
        from omegaconf import OmegaConf

        cfg_micro = self.config.get("micro", {}) or {}
        log_every = int(cfg_micro.get("log_every", 1))

        num_epochs = int(self.config.trainer.total_epochs)
        steps_per_epoch = len(self.train_dataloader)
        total_steps = num_epochs * steps_per_epoch

        drafter_cfg = self.config.actor_rollout_ref.get("drafter", {}) or {}
        assert drafter_cfg.get("enable", False), (
            "drafter.enable must be True for test_drafter_training.py "
            "(the update_drafter step performs a real forward/backward)"
        )
        model_cfg = drafter_cfg.get("model_config", {}) or {}
        assert model_cfg.get("target_model_path"), (
            "drafter.model_config.target_model_path must be set — the drafter "
            "engine is skipped when this is empty, and update_drafter falls "
            "back to the shape-print smoke path (no training). Defaults to "
            "${actor_rollout_ref.model.path} via Hydra interpolation."
        )

        print("=" * 72)
        print(
            f"  Drafter training smoke: epochs={num_epochs}, "
            f"steps/epoch={steps_per_epoch}, total_steps={total_steps}"
        )
        print(f"  drafter.model_config.target_model_path={model_cfg['target_model_path']}")
        if model_cfg.get("local_path"):
            print(f"  drafter.model_config.local_path={model_cfg['local_path']} (template overlay)")
        print(f"  use_hs_collector={self.use_hs_collector}")
        print("=" * 72)

        self._load_checkpoint()
        self.checkpoint_manager.update_weights(0)

        loss_trace: list[float] = []
        acc_len_trace: list[float] = []
        mooncake_cfg_container = OmegaConf.to_container(self.config.mooncake, resolve=True)

        # RayDrafterCTPPOTrainer._save_checkpoint reads self.global_steps for
        # the ``global_step_{N}`` directory name — normally maintained by the
        # real fit(), but our smoke loop skips that path.
        self.global_steps = 0

        for epoch in range(num_epochs):
            for batch_idx, batch_dict in enumerate(self.train_dataloader):
                timing_raw: dict = {}

                batch = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)
                gen_batch.meta_info["global_steps"] = self.global_steps + 1
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
                    print("  WARN: hs_collector is disabled — cannot train drafter without HS")
                    break

                with marked_timer("hs_collect", timing_raw, color="cyan"):
                    hs_batch = self._compute_hidden_states_colocate(batch)
                    sample_metas = _sample_metas_from_hs_batch(hs_batch)
                    self._drafter_ctrl.push_samples(sample_metas)
                    drafter_proto = self._drafter_ctrl.drain_as_dataproto()

                if drafter_proto is None:
                    print(
                        f"  epoch {epoch} step {batch_idx}: "
                        f"drafter controller has no samples to drain, skipping"
                    )
                    continue

                drafter_proto.meta_info["mooncake_cfg"] = mooncake_cfg_container

                with marked_timer("drafter_train", timing_raw, color="magenta"):
                    results = self.actor_rollout_wg.update_drafter(drafter_proto)

                metrics = (
                    results.meta_info.get("train_metrics", {})
                    if results is not None and hasattr(results, "meta_info")
                    else {}
                )
                loss_trace.append(metrics.get("train/loss_weighted", float("nan")))
                acc_len_trace.append(metrics.get("train/simulated_acc_len", float("nan")))

                if self.global_steps % log_every == 0:
                    print(
                        f"epoch {epoch:2d} step {batch_idx:4d} "
                        f"(global {self.global_steps:5d}/{total_steps}) | "
                        f"loss={metrics.get('train/loss_weighted', float('nan')):.4f} "
                        f"loss_raw={metrics.get('train/loss_raw_mean', float('nan')):.4f} "
                        f"acc0={metrics.get('train/acc_0', float('nan')):.4f} "
                        f"acc_len={metrics.get('train/simulated_acc_len', float('nan')):.2f} "
                        f"grad={metrics.get('train/grad_norm', float('nan')):.3f} "
                        f"lr={metrics.get('train/lr', float('nan')):.2e} "
                        f"| gen={timing_raw.get('gen', 0):.2f}s "
                        f"hs={timing_raw.get('hs_collect', 0):.2f}s "
                        f"train={timing_raw.get('drafter_train', 0):.2f}s"
                    )

                self.global_steps += 1
                # Keeps rollout awake (naive backend -> effectively a no-op sync).
                if self.global_steps < total_steps:
                    self.checkpoint_manager.update_weights(self.global_steps)

        # Exercise the drafter save path end-to-end: writes sharded FSDP state
        # + HF-format export to ``{default_local_dir}/global_step_N/drafter/``.
        # Gated on ``micro.save_at_end`` (default True) so the smoke can still
        # be run in no-disk-writes mode by passing +micro.save_at_end=False.
        if bool(cfg_micro.get("save_at_end", True)) and self.global_steps > 0:
            with marked_timer("save_checkpoint", {}, color="green"):
                self._save_checkpoint()
            save_root = os.path.join(
                self.config.trainer.default_local_dir,
                f"global_step_{self.global_steps}",
                "drafter",
            )
            print(f"\n  Drafter checkpoint saved to: {save_root}")
            print(f"    sharded FSDP:  model_world_size_*_rank_*.pt")
            print(f"    HF export:     huggingface/{{config.json, model.safetensors}}")

        print("\n" + "=" * 72)
        print(f"  Completed {len(loss_trace)} training step(s).")
        if len(loss_trace) >= 2:
            print(
                f"  loss[0]={loss_trace[0]:.4f} -> loss[-1]={loss_trace[-1]:.4f} "
                f"(delta={loss_trace[-1] - loss_trace[0]:+.4f}, "
                f"direction={'DOWN' if loss_trace[-1] < loss_trace[0] else 'UP'})"
            )
            print(
                f"  acc_len[0]={acc_len_trace[0]:.2f} -> acc_len[-1]={acc_len_trace[-1]:.2f} "
                f"(delta={acc_len_trace[-1] - acc_len_trace[0]:+.2f}, "
                f"direction={'UP' if acc_len_trace[-1] > acc_len_trace[0] else 'DOWN'})"
            )
        print("=" * 72)


class DrafterTrainingSmokeTaskRunner(MicroDrafterCTTaskRunner):
    """Same wiring as MicroDrafterCTTaskRunner, but uses the training-smoke trainer."""

    def run(self, config):
        from omegaconf import OmegaConf

        from verl.trainer.ppo.utils import need_critic, need_reference_policy
        from verl.utils.config import validate_config
        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.utils.fs import copy_to_local
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        print(f"DrafterTrainingSmokeTaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
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

        trainer = DrafterTrainingSmokeTrainer(
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

    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(DrafterTrainingSmokeTaskRunner))


if __name__ == "__main__":
    main()
