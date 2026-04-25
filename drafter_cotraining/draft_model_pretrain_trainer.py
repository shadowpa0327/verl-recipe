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
"""Draft-model pretraining.

This trainer keeps the drafter pretrain path separate from PPO/co-training.
Each parquet row is an explicit supervised pair:

    prompt_messages + response
        -> DataProto(prompts, responses, input_ids, masks)
        -> HS collector
        -> Mooncake sample metadata
        -> DrafterPretrainWorker.update_drafter

It reuses the existing Ray worker, HS collector, Mooncake, and EAGLE drafter
training implementation, but the fit loop is a plain supervised pretrain loop.
"""

from __future__ import annotations

import logging
import os
import socket
import sys
import uuid
from typing import Any

import hydra
import numpy as np
import ray
import torch
from omegaconf import ListConfig, OmegaConf, open_dict
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.single_controller.ray import RayWorkerGroup, ResourcePoolManager
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.utils.debug import marked_timer
from verl.utils.device import auto_set_device, is_cuda_available
from verl.utils.tracking import Tracking

from recipe.drafter_cotraining.controller import DrafterDataController, SampleMeta
from recipe.drafter_cotraining.fsdp_workers import DrafterPretrainWorker

logger = logging.getLogger(__name__)
DRAFTER_ROLE = "drafter"


def _sample_metas_from_hs_batch(hs_batch: DataProto) -> list[SampleMeta]:
    nt = hs_batch.non_tensor_batch
    if not nt or "hs_mooncake_keys" not in nt:
        return []
    keys = nt["hs_mooncake_keys"]
    prompt_lens = nt.get("hs_prompt_lens")
    response_lens = nt.get("hs_response_lens")
    return [
        SampleMeta(
            mooncake_key=str(keys[i]),
            shapes=nt["hs_shapes"][i] if isinstance(nt["hs_shapes"][i], dict) else {},
            dtypes=nt["hs_dtypes"][i] if isinstance(nt["hs_dtypes"][i], dict) else {},
            seq_len=int(nt["hs_seq_lens"][i]),
            n_tokens=int(nt["hs_seq_lens"][i]),
            prompt_len=int(prompt_lens[i]) if prompt_lens is not None else 0,
            response_len=int(response_lens[i]) if response_lens is not None else 0,
        )
        for i in range(len(keys))
    ]


class ParquetDrafterPretrainDataset(Dataset):
    """Read explicit prompt/response drafter pretrain rows from parquet."""

    def __init__(
        self,
        data_files,
        prompt_messages_key: str = "prompt_messages",
        response_key: str = "response",
        max_samples: int = -1,
    ):
        import pandas as pd

        from verl.utils.fs import copy_local_path_from_hdfs
        from verl.utils.py_functional import convert_nested_value_to_list_recursive

        if not isinstance(data_files, list | ListConfig):
            data_files = [data_files]

        samples: list[tuple[list[dict[str, str]], str]] = []
        for data_file in data_files:
            local_path = copy_local_path_from_hdfs(data_file, verbose=True)
            dataframe = pd.read_parquet(local_path, dtype_backend="pyarrow")
            missing = [k for k in (prompt_messages_key, response_key) if k not in dataframe.columns]
            if missing:
                raise ValueError(f"{local_path} is missing required column(s): {missing}")

            for _, row in dataframe.iterrows():
                prompt_messages = convert_nested_value_to_list_recursive(row[prompt_messages_key])
                response = row[response_key]
                if not isinstance(response, str):
                    continue
                samples.append((prompt_messages, response))
                if max_samples > 0 and len(samples) >= max_samples:
                    break
            if max_samples > 0 and len(samples) >= max_samples:
                break

        if not samples:
            raise ValueError(f"No drafter pretrain samples found in {data_files}")
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[list[dict[str, str]], str]:
        return self.samples[index]


def _tokenize_prompt_response(
    tokenizer,
    prompt_msgs: list[dict[str, Any]],
    response_text: str,
    max_prompt_len: int,
    max_response_len: int,
    apply_chat_template_kwargs: dict[str, Any],
) -> tuple[list[int], list[int]]:
    prompt_text = tokenizer.apply_chat_template(
        prompt_msgs,
        add_generation_prompt=True,
        tokenize=False,
        **apply_chat_template_kwargs,
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response_text, add_special_tokens=False)["input_ids"]

    if len(prompt_ids) > max_prompt_len:
        prompt_ids = prompt_ids[-max_prompt_len:]
    if len(response_ids) > max_response_len:
        response_ids = response_ids[:max_response_len]
    return list(prompt_ids), list(response_ids)


def _pad_batch_to_dataproto(
    samples: list[tuple[list[int], list[int]]],
    pad_token_id: int,
    prompt_length: int,
    response_length: int,
) -> DataProto:
    batch_size = len(samples)
    prompts = torch.full((batch_size, prompt_length), pad_token_id, dtype=torch.long)
    prompt_mask = torch.zeros((batch_size, prompt_length), dtype=torch.long)
    responses = torch.full((batch_size, response_length), pad_token_id, dtype=torch.long)
    response_mask = torch.zeros((batch_size, response_length), dtype=torch.long)

    for i, (prompt_ids, response_ids) in enumerate(samples):
        prompt_len = len(prompt_ids)
        response_len = len(response_ids)
        if prompt_len > 0:
            prompts[i, -prompt_len:] = torch.tensor(prompt_ids, dtype=torch.long)
            prompt_mask[i, -prompt_len:] = 1
        if response_len > 0:
            responses[i, :response_len] = torch.tensor(response_ids, dtype=torch.long)
            response_mask[i, :response_len] = 1

    input_ids = torch.cat([prompts, responses], dim=1)
    attention_mask = torch.cat([prompt_mask, response_mask], dim=1)
    position_ids = (attention_mask.cumsum(dim=1) - 1).clamp(min=0)
    position_ids = position_ids * attention_mask

    batch = DataProto.from_single_dict(
        {
            "prompts": prompts,
            "responses": responses,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "response_mask": response_mask,
        }
    )
    batch.non_tensor_batch["uid"] = np.array(
        [str(uuid.uuid4()) for _ in range(batch_size)], dtype=object
    )
    return batch


class DrafterPretrainCollator:
    def __init__(
        self,
        tokenizer,
        max_prompt_len: int,
        max_response_len: int,
        apply_chat_template_kwargs: dict[str, Any] | None = None,
    ):
        self.tokenizer = tokenizer
        self.max_prompt_len = max_prompt_len
        self.max_response_len = max_response_len
        self.apply_chat_template_kwargs = apply_chat_template_kwargs or {}
        self.pad_token_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )

    def __call__(self, batch_rows: list[tuple[list[dict[str, str]], str]]) -> DataProto:
        samples: list[tuple[list[int], list[int]]] = []
        skipped = 0
        for prompt_msgs, response_text in batch_rows:
            try:
                prompt_ids, response_ids = _tokenize_prompt_response(
                    self.tokenizer,
                    prompt_msgs,
                    response_text,
                    self.max_prompt_len,
                    self.max_response_len,
                    self.apply_chat_template_kwargs,
                )
            except Exception as exc:  # noqa: BLE001
                skipped += 1
                logger.warning("Skipping malformed pretrain row: %s", exc)
                continue
            if not prompt_ids or not response_ids:
                skipped += 1
                continue
            samples.append((prompt_ids, response_ids))

        if not samples:
            raise ValueError(f"All {len(batch_rows)} rows in this batch were invalid")

        batch = _pad_batch_to_dataproto(
            samples=samples,
            pad_token_id=self.pad_token_id,
            prompt_length=self.max_prompt_len,
            response_length=self.max_response_len,
        )
        batch.meta_info["skipped_rows"] = skipped
        return batch


class DraftModelPretrainTrainer:
    """Standalone drafter pretrainer."""

    def __init__(
        self,
        config,
        tokenizer,
        resource_pool_manager: ResourcePoolManager,
        drafter_worker_cls,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.resource_pool_manager = resource_pool_manager
        self.drafter_worker_cls = drafter_worker_cls
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = self.config.trainer.device
        self.use_hs_collector = bool(config.get("hs_collector", {}).get("enabled", False))
        self.hs_collector_manager = None
        self.drafter_wg = None

        dp_size = config.trainer.n_gpus_per_node * config.trainer.nnodes
        self._drafter_ctrl = DrafterDataController(dp_size=dp_size)
        self._create_dataloader(train_dataset=None)

    def init_workers(self):
        """Initialize only the drafter worker group and HS collector."""
        from verl.single_controller.ray import RayClassWithInitArgs
        from verl.single_controller.ray.base import create_colocated_worker_cls

        self._validate_pretrain_sequence_lengths()

        self.resource_pool_manager.create_resource_pool()
        resource_pool = self.resource_pool_manager.get_resource_pool(DRAFTER_ROLE)

        class_dict = {
            DRAFTER_ROLE: RayClassWithInitArgs(
                cls=self.drafter_worker_cls,
                config=self.config.actor_rollout_ref,
                role="drafter",
            )
        }

        wg_kwargs = {}
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = (
                self.config.trainer.ray_wait_register_center_timeout
            )
        wg_kwargs["device_name"] = self.device_name

        worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
        worker_group = self.ray_worker_group_cls(
            resource_pool=resource_pool,
            ray_cls_with_init=worker_dict_cls,
            **wg_kwargs,
        )
        spawned = worker_group.spawn(prefix_set=class_dict.keys())
        self.drafter_wg = spawned[DRAFTER_ROLE]
        self.drafter_wg.init_model()

        if self.use_hs_collector:
            from recipe.drafter_cotraining.hs_collector import HSCollectorManager

            self.hs_collector_manager = HSCollectorManager(
                config=self.config.hs_collector,
                resource_pool=resource_pool,
            )

    def _build_pretrain_dataset(
        self,
        data_files,
        max_samples: int,
    ) -> Dataset | None:
        if not data_files:
            return None
        return ParquetDrafterPretrainDataset(
            data_files,
            prompt_messages_key=self.config.data.get("prompt_messages_key", "prompt_messages"),
            response_key=self.config.data.get("response_key", "response"),
            max_samples=max_samples,
        )

    def _create_dataloader(self, train_dataset):
        data_cfg = self.config.data
        if train_dataset is None:
            train_dataset = self._build_pretrain_dataset(
                data_files=data_cfg.train_files,
                max_samples=data_cfg.get("train_max_samples", -1),
            )
        if train_dataset is None:
            raise ValueError("data.train_files must be set")

        self.train_dataset = train_dataset

        eval_files = data_cfg.get("eval_files", None)
        val_dataset = self._build_pretrain_dataset(
            data_files=eval_files,
            max_samples=data_cfg.get("val_max_samples", -1),
        )
        self.val_dataset = val_dataset

        collator = DrafterPretrainCollator(
            tokenizer=self.tokenizer,
            max_prompt_len=int(data_cfg.max_prompt_length),
            max_response_len=int(data_cfg.max_response_length),
            apply_chat_template_kwargs=data_cfg.get("apply_chat_template_kwargs", {}) or {},
        )

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=int(data_cfg.train_batch_size),
            shuffle=bool(data_cfg.get("shuffle", True)),
            num_workers=int(data_cfg.get("dataloader_num_workers", 0)),
            drop_last=True,
            collate_fn=collator,
        )

        if self.val_dataset is not None:
            val_batch_size = data_cfg.get("val_batch_size", None)
            if val_batch_size is None:
                val_batch_size = data_cfg.train_batch_size
            self.val_dataloader = StatefulDataLoader(
                dataset=self.val_dataset,
                batch_size=int(val_batch_size),
                shuffle=bool(data_cfg.get("validation_shuffle", False)),
                num_workers=int(data_cfg.get("dataloader_num_workers", 0)),
                drop_last=False,
                collate_fn=collator,
            )
        else:
            self.val_dataloader = None

        if len(self.train_dataloader) < 1:
            raise ValueError("Train dataloader is empty")

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps
        self.total_training_steps = int(total_training_steps)

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                self.config.actor_rollout_ref.drafter.optimizer_config.total_training_steps = (
                    self.total_training_steps
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not set drafter total_training_steps: %s", exc)

        print(
            "Draft pretrain dataloader: "
            f"{len(self.train_dataloader)} steps/epoch, "
            f"val_steps={len(self.val_dataloader) if self.val_dataloader is not None else 0}, "
            f"total_training_steps={self.total_training_steps}"
        )

    def _validate_pretrain_config(self):
        drafter_cfg = self.config.actor_rollout_ref.get("drafter", {}) or {}
        if not drafter_cfg.get("enable", False):
            raise ValueError("actor_rollout_ref.drafter.enable must be True")
        if not self.use_hs_collector:
            raise ValueError("hs_collector.enabled must be True")
        if self.hs_collector_manager is None:
            raise ValueError("HSCollectorManager is not initialized")

        self._validate_pretrain_sequence_lengths()

    def _validate_pretrain_sequence_lengths(self):
        max_prompt = int(self.config.data.max_prompt_length)
        max_response = int(self.config.data.max_response_length)
        hs_max_len = int(self.config.hs_collector.inference.max_model_len)
        if hs_max_len <= max_prompt + max_response:
            raise ValueError(
                "hs_collector.inference.max_model_len must be greater than "
                "data.max_prompt_length + data.max_response_length because the "
                "collector sends prompt+response as prefill plus max_tokens=1"
            )

    def _save_drafter_checkpoint(self, step: int):
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir,
            f"global_step_{step}",
        )
        local_path = os.path.join(local_global_step_folder, "drafter")
        remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(
                self.config.trainer.default_hdfs_dir,
                f"global_step_{step}",
                "drafter",
            )
        )
        max_to_keep = self.config.trainer.get(
            "max_drafter_ckpt_to_keep",
            self.config.trainer.get("max_actor_ckpt_to_keep", None),
        )
        self.drafter_wg.save_drafter_checkpoint(
            local_path,
            remote_path,
            step,
            max_ckpt_to_keep=max_to_keep,
        )

    def _compute_hidden_states(self, batch: DataProto) -> DataProto:
        assert self.hs_collector_manager is not None, "HSCollectorManager is not initialized"
        return self.hs_collector_manager.compute_hidden_states(batch)

    def _run_drafter_batch(
        self,
        batch: DataProto,
        mooncake_cfg: dict,
        timing_raw: dict[str, float],
        mode: str,
    ) -> dict[str, float]:
        with marked_timer("drafter_offload", timing_raw, color="yellow"):
            self.drafter_wg.sleep()

        with marked_timer("hs_collect", timing_raw, color="cyan"):
            hs_batch = self._compute_hidden_states(batch)
            sample_metas = _sample_metas_from_hs_batch(hs_batch)
            self._drafter_ctrl.push_samples(sample_metas)
            drafter_proto = self._drafter_ctrl.drain_as_dataproto()

        prefix = "train" if mode == "train" else "val"
        if drafter_proto is None:
            logger.warning("%s step produced no drafter samples", mode)
            return {
                f"{prefix}/samples": 0.0,
                f"{prefix}/skipped_rows": float(
                    batch.meta_info.get("skipped_rows", 0)
                ),
            }

        drafter_proto.meta_info["mooncake_cfg"] = mooncake_cfg
        timer_name = "drafter_train" if mode == "train" else "drafter_eval"
        with marked_timer("drafter_load", timing_raw, color="yellow"):
            self.drafter_wg.wake_up()

        with marked_timer(timer_name, timing_raw, color="magenta"):
            if mode == "train":
                results = self.drafter_wg.update_drafter(drafter_proto)
                result_key = "train_metrics"
            else:
                results = self.drafter_wg.evaluate_drafter(drafter_proto)
                result_key = "eval_metrics"

        metrics: dict[str, float] = {}
        if results is not None and hasattr(results, "meta_info"):
            metrics.update(results.meta_info.get(result_key, {}) or {})

        metrics[f"{prefix}/samples"] = float(len(sample_metas))
        metrics[f"{prefix}/skipped_rows"] = float(
            batch.meta_info.get("skipped_rows", 0)
        )
        return metrics

    def _validate(self, tracker: Tracking, mooncake_cfg: dict, step: int) -> dict[str, float]:
        if self.val_dataloader is None:
            return {}

        max_batches = int(self.config.get("pretrain", {}).get("val_max_batches", -1))
        metric_sums: dict[str, float] = {}
        metric_counts: dict[str, int] = {}
        timing_sums: dict[str, float] = {}
        batches = 0

        for batch in tqdm(self.val_dataloader, desc="Draft Eval", leave=False):
            if max_batches > 0 and batches >= max_batches:
                break
            timing_raw: dict[str, float] = {}
            metrics = self._run_drafter_batch(
                batch=batch,
                mooncake_cfg=mooncake_cfg,
                timing_raw=timing_raw,
                mode="eval",
            )
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    metric_sums[key] = metric_sums.get(key, 0.0) + float(value)
                    metric_counts[key] = metric_counts.get(key, 0) + 1
            for key, value in timing_raw.items():
                timing_sums[f"timing/val_{key}"] = timing_sums.get(f"timing/val_{key}", 0.0) + value
            batches += 1

        if batches == 0:
            return {}

        val_metrics = {
            key: value / metric_counts[key]
            for key, value in metric_sums.items()
            if metric_counts.get(key, 0) > 0
        }
        val_metrics.update({key: value / batches for key, value in timing_sums.items()})
        val_metrics["val/num_batches"] = float(batches)
        tracker.log(data=val_metrics, step=step)
        print(
            f"eval step {step:5d} | "
            f"loss={val_metrics.get('val/loss_weighted', float('nan')):.4f} "
            f"acc_len={val_metrics.get('val/simulated_acc_len', float('nan')):.2f}"
        )
        return val_metrics

    def fit(self):
        self._validate_pretrain_config()
        tracker = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        mooncake_cfg = OmegaConf.to_container(self.config.mooncake, resolve=True)
        log_every = max(1, int(self.config.get("pretrain", {}).get("log_every", 1)))

        print("=" * 72)
        print("  Draft model pretraining")
        print(f"  total_training_steps={self.total_training_steps}")
        print(f"  batch_size={self.config.data.train_batch_size}")
        print(f"  prompt_len={self.config.data.max_prompt_length}")
        print(f"  response_len={self.config.data.max_response_length}")
        print(f"  logger={self.config.trainer.logger}")
        print("=" * 72)

        self.global_steps = 0
        if self.config.trainer.get("val_before_train", False):
            self._validate(tracker=tracker, mooncake_cfg=mooncake_cfg, step=self.global_steps)

        progress_bar = tqdm(total=self.total_training_steps, desc="Draft Pretrain")

        epoch = 0
        total_epochs = int(self.config.trainer.total_epochs)
        explicit_total_steps = self.config.trainer.total_training_steps is not None
        test_freq = int(self.config.trainer.test_freq)
        while self.global_steps < self.total_training_steps:
            if not explicit_total_steps and epoch >= total_epochs:
                break
            for batch in self.train_dataloader:
                self.global_steps += 1
                timing_raw: dict[str, float] = {}
                metrics = self._run_drafter_batch(
                    batch=batch,
                    mooncake_cfg=mooncake_cfg,
                    timing_raw=timing_raw,
                    mode="train",
                )
                metrics.update({f"timing/{k}": v for k, v in timing_raw.items()})

                tracker.log(data=metrics, step=self.global_steps)

                if self.global_steps % log_every == 0:
                    print(
                        f"step {self.global_steps:5d} | "
                        f"loss={metrics.get('train/loss_weighted', float('nan')):.4f} "
                        f"acc_len={metrics.get('train/simulated_acc_len', float('nan')):.2f} "
                        f"grad={metrics.get('train/grad_norm', float('nan')):.3f} "
                        f"lr={metrics.get('train/lr', float('nan')):.2e} "
                        f"| hs={timing_raw.get('hs_collect', 0):.2f}s "
                        f"train={timing_raw.get('drafter_train', 0):.2f}s"
                    )

                save_freq = int(self.config.trainer.save_freq)
                is_last_step = self.global_steps >= self.total_training_steps
                is_valid_step = (
                    self.val_dataloader is not None
                    and ((test_freq > 0 and self.global_steps % test_freq == 0) or is_last_step)
                )
                saved_this_step = False
                if save_freq > 0 and self.global_steps % save_freq == 0:
                    self._save_drafter_checkpoint(self.global_steps)
                    saved_this_step = True

                if is_valid_step:
                    self._validate(
                        tracker=tracker,
                        mooncake_cfg=mooncake_cfg,
                        step=self.global_steps,
                    )

                progress_bar.update(1)
                if is_last_step:
                    save_final = bool(self.config.get("pretrain", {}).get("save_final", True))
                    if save_final and not saved_this_step:
                        self._save_drafter_checkpoint(self.global_steps)
                    progress_bar.close()
                    return

            epoch += 1

        progress_bar.close()


class DraftModelPretrainTaskRunner:
    """Ray task runner for standalone drafter pretraining."""

    def _init_resource_pool_mgr(self, config):
        global_pool_id = "global_pool"
        return ResourcePoolManager(
            resource_pool_spec={global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes},
            mapping={DRAFTER_ROLE: global_pool_id},
        )

    def run(self, config):
        from verl.utils.fs import copy_to_local

        print(f"DraftModelPretrainTaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        OmegaConf.resolve(config)

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )

        from verl.utils import hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        resource_pool_manager = self._init_resource_pool_mgr(config)

        trainer = DraftModelPretrainTrainer(
            config=config,
            tokenizer=tokenizer,
            resource_pool_manager=resource_pool_manager,
            drafter_worker_cls=ray.remote(DrafterPretrainWorker),
            ray_worker_group_cls=RayWorkerGroup,
        )
        trainer.init_workers()
        trainer.fit()


def _launch_mooncake_master_if_needed(config):
    pretrain_cfg = config.get("pretrain", {}) or {}
    if not bool(pretrain_cfg.get("launch_master", True)):
        return

    from types import SimpleNamespace

    from recipe.drafter_cotraining.mooncake.master import launch_mooncake_master

    args = SimpleNamespace(
        mooncake_master_server_address=config.mooncake.master_server_address,
        mooncake_metadata_port=int(
            config.mooncake.metadata_server.rsplit(":", 1)[1].split("/")[0]
        ),
        mooncake_kv_lease_ttl_s=float(config.mooncake.kv_lease_ttl_s),
    )
    if launch_mooncake_master(args) is None:
        sys.exit("mooncake_master failed to launch (binary missing?)")


def _ensure_ray_initialized(config):
    if ray.is_initialized():
        return

    ray_init_kwargs = OmegaConf.to_container(
        config.ray_kwargs.get("ray_init", {}), resolve=True
    ) or {}
    runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {}) or {}

    if OmegaConf.select(config, "transfer_queue.enable"):
        runtime_env_vars = runtime_env_kwargs.get("env_vars", {}) or {}
        runtime_env_vars["TRANSFER_QUEUE_ENABLE"] = "1"
        runtime_env_kwargs["env_vars"] = runtime_env_vars

    runtime_env = OmegaConf.merge(get_ppo_ray_runtime_env(), runtime_env_kwargs)
    ray_init_kwargs["runtime_env"] = OmegaConf.to_container(runtime_env, resolve=True)
    print(f"ray init kwargs: {ray_init_kwargs}")
    ray.init(**ray_init_kwargs)


def run_draft_model_pretrain(config) -> None:
    _ensure_ray_initialized(config)
    task_runner_class = ray.remote(num_cpus=1)(DraftModelPretrainTaskRunner)

    if (
        is_cuda_available
        and OmegaConf.select(config, "global_profiler.tool") == "nsys"
        and OmegaConf.select(config, "global_profiler.steps") is not None
        and len(OmegaConf.select(config, "global_profiler.steps")) > 0
    ):
        from verl.utils.import_utils import is_nvtx_available

        assert is_nvtx_available(), "nvtx is not available in CUDA platform. Please 'pip3 install nvtx'"
        nsight_options = OmegaConf.to_container(
            config.global_profiler.global_tool_config.nsys.controller_nsight_options
        )
        runner = task_runner_class.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = task_runner_class.remote()

    ray.get(runner.run.remote(config))

    timeline_json_file = config.ray_kwargs.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


@hydra.main(config_path="config", config_name="draft_model_pretrain_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    _ensure_ray_initialized(config)
    _launch_mooncake_master_if_needed(config)
    run_draft_model_pretrain(config)


if __name__ == "__main__":
    main()
