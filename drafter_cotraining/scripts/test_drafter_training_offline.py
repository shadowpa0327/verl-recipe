# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Offline drafter-training diagnostic — no rollout.

Feeds pre-recorded multi-turn conversations (canonical JSONL: one row per
conversation, ``{"id", "conversations": [{"role", "content"}, ...]}``)
directly to the HS collector + drafter trainer. Lets us isolate drafter
training correctness from rollout nondeterminism.

Pipeline per step:
    conversation batch -> render with chat template -> tokenize ->
        per-turn assistant loss_mask (TorchSpec preprocessing semantics)
        -> DataProto(input_ids, attention_mask, position_ids, loss_mask)
        -> HS collector (Mooncake prefill, forwards loss_masks unchanged)
        -> DrafterDataController.push + drain
        -> ActorRolloutRefDrafterWorker.update_drafter
           (fetch + collate + Eagle3 forward + 0.8^i backward + optimizer.step)

Skipped: rollout, teacher, reward, advantage, critic/actor update, weight sync,
checkpoint save, validation.

Pass signal:
    train/loss_weighted trends down
    train/simulated_acc_len trends up
    No NaN/Inf, no OOM.

Usage (via wrapper):
    scripts/run_drafter_training_offline.sh

Extra config keys (all optional, under `micro.*`):
    micro.conversations_path (str)  JSONL path, each row {"conversations":[{"role","content"}, ...]}
    micro.max_steps          (int)  Number of drafter steps. Default 32.
    micro.batch_size         (int)  Samples per drafter step. Default 8.
    micro.shuffle_seed       (int)  Shuffle seed. Default 0.
"""

import json
import os
import socket
import sys
import uuid
from pprint import pprint
from typing import Iterator

import hydra
import numpy as np
import ray
import torch

from verl import DataProto
from recipe.drafter_cotraining.ray_trainer import _sample_metas_from_hs_batch
from recipe.drafter_cotraining.scripts.test_drafter_rollout_hs import (
    MicroDrafterCTTaskRunner,
    MicroRolloutHSOnlyTrainer,
)
from verl.trainer.main_ppo import run_ppo
from verl.utils.debug import marked_timer
from verl.utils.device import auto_set_device


# ── Conversation → DataProto ────────────────────────────────────────────────


def _load_conversations(path: str) -> list[list[dict]]:
    """Load canonical JSONL → list of `conversations` lists.

    Each row must have a `conversations` field with a list of
    {"role", "content"} dicts.
    """
    out: list[list[dict]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            conv = row.get("conversations")
            if conv:
                out.append(conv)
    if not out:
        raise ValueError(f"No conversations found in {path}")
    return out


def _iterate_offline_batches(
    conversations: list[list[dict]],
    tokenizer,
    batch_size: int,
    max_seq_length: int,
    chat_template: str,
    shuffle_seed: int,
) -> Iterator[DataProto]:
    """Yield infinite DataProto batches from a shuffled copy of ``conversations``.

    Uses the same ``DrafterPretrainCollator`` the trainer uses, so the
    offline diagnostic exercises the per-turn assistant loss-mask path
    end-to-end.
    """
    from recipe.drafter_cotraining.draft_model_pretrain_trainer import (
        DrafterPretrainCollator,
    )

    rng = np.random.default_rng(shuffle_seed)
    order = np.arange(len(conversations))
    coll = DrafterPretrainCollator(
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        chat_template=chat_template,
    )
    while True:
        rng.shuffle(order)
        buf: list[tuple[str, list[dict]]] = []
        for idx in order:
            conv = conversations[int(idx)]
            buf.append((str(uuid.uuid4()), conv))
            if len(buf) == batch_size:
                yield coll(buf)
                buf = []
        # Discard partial tail; the next epoch reshuffles.


# ── Trainer ─────────────────────────────────────────────────────────────────


class OfflineDrafterTrainingTrainer(MicroRolloutHSOnlyTrainer):
    """Drafter training smoke with no rollout — conversations fed straight in."""

    def fit(self):  # noqa: D401
        from omegaconf import OmegaConf

        cfg_micro = self.config.get("micro", {}) or {}
        max_steps = int(cfg_micro.get("max_steps", 32))
        log_every = int(cfg_micro.get("log_every", 1))
        batch_size = int(cfg_micro.get("batch_size", self.config.data.train_batch_size))
        shuffle_seed = int(cfg_micro.get("shuffle_seed", 0))
        conversations_path = cfg_micro.get("conversations_path")
        if not conversations_path:
            raise ValueError(
                "micro.conversations_path must be set for the offline diagnostic "
                "(point at a ShareGPT-style JSONL, e.g. "
                "ref/TorchSpec/examples/data/sample_conversations.jsonl)"
            )

        drafter_cfg = self.config.actor_rollout_ref.get("drafter", {}) or {}
        assert drafter_cfg.get("enable", False), (
            "drafter.enable must be True for the offline training diagnostic"
        )
        model_cfg = drafter_cfg.get("model_config", {}) or {}
        assert model_cfg.get("local_path"), (
            "drafter.model_config.local_path must be set — engine is skipped when empty"
        )

        max_seq_length = int(
            self.config.data.get("max_seq_length", 0)
            or (
                int(self.config.data.get("max_prompt_length", 0))
                + int(self.config.data.get("max_response_length", 0))
            )
        )
        if max_seq_length <= 0:
            raise ValueError("data.max_seq_length must be set to a positive integer")
        chat_template = str(self.config.data.get("chat_template", "qwen"))

        print("=" * 72)
        print(f"  Offline drafter training: max_steps={max_steps} batch_size={batch_size}")
        print(f"  conversations_path={conversations_path}")
        print(f"  max_seq_length={max_seq_length} chat_template={chat_template}")
        print(f"  drafter.model_config.local_path={model_cfg['local_path']}")
        print(f"  use_hs_collector={self.use_hs_collector}")
        print("=" * 72)

        conversations = _load_conversations(conversations_path)
        print(f"  loaded {len(conversations)} conversations")

        data_iter = _iterate_offline_batches(
            conversations,
            self.tokenizer,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
            chat_template=chat_template,
            shuffle_seed=shuffle_seed,
        )

        # Not needed (no actor/rollout update), but mirrors the training script
        # so checkpoint-manager state doesn't complain on first call.
        self._load_checkpoint()

        mooncake_cfg_container = OmegaConf.to_container(self.config.mooncake, resolve=True)

        loss_trace: list[float] = []
        acc_len_trace: list[float] = []

        for step in range(max_steps):
            timing_raw: dict = {}
            batch = next(data_iter)
            batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

            if not self._should_compute_hidden_states_colocate(batch):
                print("  ERROR: hs_collector disabled; set hs_collector.enabled=True")
                break

            with marked_timer("hs_collect", timing_raw, color="cyan"):
                hs_batch = self._compute_hidden_states_colocate(batch)
                sample_metas = _sample_metas_from_hs_batch(hs_batch)
                self._drafter_ctrl.push_samples(sample_metas)
                drafter_proto = self._drafter_ctrl.drain_as_dataproto()

            if drafter_proto is None:
                print(f"  step {step}: drafter controller has no samples to drain, skipping")
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

            if step % log_every == 0:
                print(
                    f"step {step:4d} | "
                    f"loss={metrics.get('train/loss_weighted', float('nan')):.4f} "
                    f"loss_raw={metrics.get('train/loss_raw_mean', float('nan')):.4f} "
                    f"acc0={metrics.get('train/acc_0', float('nan')):.4f} "
                    f"acc_len={metrics.get('train/simulated_acc_len', float('nan')):.2f} "
                    f"grad={metrics.get('train/grad_norm', float('nan')):.3f} "
                    f"lr={metrics.get('train/lr', float('nan')):.2e} "
                    f"| hs={timing_raw.get('hs_collect', 0):.2f}s "
                    f"train={timing_raw.get('drafter_train', 0):.2f}s"
                )

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


class OfflineDrafterTrainingTaskRunner(MicroDrafterCTTaskRunner):
    """Same wiring as MicroDrafterCTTaskRunner, but uses the offline trainer."""

    def run(self, config):
        from omegaconf import OmegaConf

        from verl.trainer.ppo.utils import need_critic, need_reference_policy
        from verl.utils.config import validate_config
        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.utils.fs import copy_to_local
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        print(f"OfflineDrafterTrainingTaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
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

        # We still build the RL dataset / sampler just so the trainer's
        # __init__ (which runs _create_dataloader) doesn't blow up. The
        # offline fit() ignores train_dataloader entirely.
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

        trainer = OfflineDrafterTrainingTrainer(
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

    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(OfflineDrafterTrainingTaskRunner))


if __name__ == "__main__":
    main()
