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
from recipe.drafter_cotraining.engine_workers import DrafterPretrainWorker

logger = logging.getLogger(__name__)
DRAFTER_ROLE = "drafter"


def _sample_metas_from_hs_batch(hs_batch: DataProto) -> list[SampleMeta]:
    nt = hs_batch.non_tensor_batch
    if not nt or "hs_mooncake_keys" not in nt:
        return []
    keys = nt["hs_mooncake_keys"]
    prompt_lens = nt.get("hs_prompt_lens")
    response_lens = nt.get("hs_response_lens")
    # New fields for canonical format
    loss_masks = nt.get("loss_mask")
    valid_tokens_list = nt.get("valid_tokens")

    metas = []
    for i in range(len(keys)):
        # Prefer loss_mask numpy array if available (canonical format)
        loss_mask_arr = None
        valid_tok = 0
        if loss_masks is not None and i < len(loss_masks):
            lm = loss_masks[i]
            if isinstance(lm, np.ndarray) and len(lm) > 0:
                loss_mask_arr = lm

        if valid_tokens_list is not None and i < len(valid_tokens_list):
            valid_tok = int(valid_tokens_list[i])

        meta = SampleMeta(
            mooncake_key=str(keys[i]),
            shapes=nt["hs_shapes"][i] if isinstance(nt["hs_shapes"][i], dict) else {},
            dtypes=nt["hs_dtypes"][i] if isinstance(nt["hs_dtypes"][i], dict) else {},
            seq_len=int(nt["hs_seq_lens"][i]),
            n_tokens=int(nt["hs_seq_lens"][i]),
            prompt_len=int(prompt_lens[i]) if prompt_lens is not None else 0,
            response_len=int(response_lens[i]) if response_lens is not None else 0,
            loss_mask=loss_mask_arr,
            valid_tokens=valid_tok if valid_tok > 0 else (int(response_lens[i]) - 1 if response_lens is not None and response_lens[i] > 1 else 0),
        )
        metas.append(meta)

    return metas


class ParquetDrafterPretrainDataset(Dataset):
    """Read drafter pretrain rows from parquet in canonical (messages) format."""

    def __init__(
        self,
        data_files,
        messages_key: str = "messages",
        max_samples: int = -1,
    ):
        import pandas as pd

        from verl.utils.fs import copy_local_path_from_hdfs
        from verl.utils.py_functional import convert_nested_value_to_list_recursive

        if not isinstance(data_files, list | ListConfig):
            data_files = [data_files]

        samples: list[list[dict[str, str]]] = []
        sample_ids: list[str] = []

        for data_file in data_files:
            local_path = copy_local_path_from_hdfs(data_file, verbose=True)
            dataframe = pd.read_parquet(local_path, dtype_backend="pyarrow")

            if messages_key not in dataframe.columns:
                raise ValueError(f"{local_path} missing required column: '{messages_key}'")

            for idx, row in dataframe.iterrows():
                messages = convert_nested_value_to_list_recursive(row[messages_key])

                if not messages:
                    continue

                # Validate messages have assistant content
                has_assistant = any(m.get("role") == "assistant" for m in messages)
                if not has_assistant:
                    continue

                samples.append(messages)
                sample_id = row.get("id", f"sample_{len(samples)}")
                sample_ids.append(str(sample_id) if sample_id else f"sample_{len(samples)}")

                if max_samples > 0 and len(samples) >= max_samples:
                    break
            if max_samples > 0 and len(samples) >= max_samples:
                break

        if not samples:
            raise ValueError(f"No drafter pretrain samples found in {data_files}")
        self.samples = samples
        self.sample_ids = sample_ids

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[list[dict[str, str]], str]:
        return self.samples[index], self.sample_ids[index]


class DrafterPretrainCollator:
    """Collator for canonical (messages) format with proper loss mask computation.

    This collator:
    1. Tokenizes the full conversation using tokenizer.apply_chat_template
    2. Computes loss_mask for all assistant turns (or last turn only based on mode)
    3. Packs the loss mask for transport through Mooncake
    4. Handles truncation while preserving token/mask alignment
    """

    def __init__(
        self,
        tokenizer,
        max_seq_len: int,
        loss_mask_mode: str = "all_assistant_turns",
        apply_chat_template_kwargs: dict[str, Any] | None = None,
    ):
        """
        Args:
            tokenizer: HuggingFace tokenizer with chat template support.
            max_seq_len: Maximum total sequence length.
            loss_mask_mode: One of:
                - "all_assistant_turns": Supervise all assistant content (default)
                - "last_assistant_only": Supervise only final assistant message
                - "auto": Use last-turn-only if thinking content detected
            apply_chat_template_kwargs: Extra kwargs for tokenizer.apply_chat_template.
        """
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.loss_mask_mode = loss_mask_mode
        self.apply_chat_template_kwargs = apply_chat_template_kwargs or {}
        self.pad_token_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )
        from recipe.drafter_cotraining.loss_mask_utils import build_loss_mask_for_conversation

        self._build_loss_mask = build_loss_mask_for_conversation

    def __call__(self, batch_rows: list[tuple[list[dict[str, str]], str]]) -> DataProto:
        """Collate a batch of conversations into a DataProto with loss masks.

        Args:
            batch_rows: List of (messages, sample_id) tuples.

        Returns:
            DataProto with input_ids, attention_mask, loss_mask.
        """

        samples: list[dict[str, Any]] = []
        skipped = 0

        for messages, sample_id in batch_rows:
            try:
                result = self._build_loss_mask(
                    tokenizer=self.tokenizer,
                    messages=messages,
                    max_length=self.max_seq_len,
                    loss_mask_mode=self.loss_mask_mode,
                )
            except Exception as exc:  # noqa: BLE001
                skipped += 1
                logger.warning("Skipping malformed pretrain row: %s", exc)
                continue

            if result is None:
                skipped += 1
                logger.warning("Skipping row with empty loss mask: %s", sample_id)
                continue

            samples.append({
                "input_ids": result.input_ids,
                "loss_mask": result.loss_mask,
                "valid_tokens": result.valid_tokens,
                "sample_id": sample_id,
            })

        if not samples:
            raise ValueError(f"All {len(batch_rows)} rows in this batch were invalid")

        # Find max length in batch
        max_len = max(s["input_ids"].shape[0] for s in samples)
        # Round up to 256-token bucket for torch.compile stability
        bucket = 256
        max_len = ((max_len + bucket - 1) // bucket) * bucket

        batch_size = len(samples)
        input_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        loss_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

        # Store unpadded loss_masks as numpy arrays (dtype=object for variable length)
        loss_masks_np: list[np.ndarray] = []
        valid_tokens_list: list[int] = []
        sample_ids: list[str] = []

        for i, sample in enumerate(samples):
            seq_len = sample["input_ids"].shape[0]
            input_ids[i, :seq_len] = sample["input_ids"]
            attention_mask[i, :seq_len] = 1
            loss_mask[i, :seq_len] = sample["loss_mask"]
            # Store unpadded loss_mask as numpy array for transport
            loss_masks_np.append(sample["loss_mask"].cpu().numpy())
            valid_tokens_list.append(sample["valid_tokens"])
            sample_ids.append(sample["sample_id"])

        batch = DataProto.from_single_dict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "loss_mask": loss_mask,
            }
        )
        batch.non_tensor_batch["uid"] = np.array(sample_ids, dtype=object)
        batch.non_tensor_batch["loss_mask"] = np.array(loss_masks_np, dtype=object)
        batch.non_tensor_batch["valid_tokens"] = np.array(valid_tokens_list, dtype=np.int64)

        # Legacy fields for backward compatibility
        batch.non_tensor_batch["hs_prompt_lens"] = np.zeros(batch_size, dtype=np.int64)
        batch.non_tensor_batch["hs_response_lens"] = np.array(valid_tokens_list, dtype=np.int64)

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
        """Build pretrain dataset from canonical (messages) format.

        Args:
            data_files: Path(s) to parquet files with 'messages' column.
            max_samples: Maximum samples to load (-1 for no limit).

        Returns:
            Dataset instance.
        """
        if not data_files:
            return None

        return ParquetDrafterPretrainDataset(
            data_files,
            messages_key=self.config.data.get("messages_key", "messages"),
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

        # Create collator with loss mask support
        loss_mask_mode = data_cfg.get("loss_mask_mode", "all_assistant_turns")
        max_seq_len = int(data_cfg.get("max_seq_len", data_cfg.max_prompt_length + data_cfg.max_response_length))
        collator = DrafterPretrainCollator(
            tokenizer=self.tokenizer,
            max_seq_len=max_seq_len,
            loss_mask_mode=loss_mask_mode,
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

    # Parse metadata port from URL (e.g., "http://localhost:8090/metadata")
    # For P2PHANDSHAKE mode, use a dummy port (master will ignore it)
    metadata_server = config.mooncake.metadata_server
    if metadata_server == "P2PHANDSHAKE":
        metadata_port = 8090  # Dummy port, ignored in IPv6-only mode
    else:
        try:
            metadata_port = int(
                metadata_server.rsplit(":", 1)[1].split("/")[0]
            )
        except (ValueError, IndexError):
            metadata_port = 8090  # Default fallback

    args = SimpleNamespace(
        mooncake_master_server_address=config.mooncake.master_server_address,
        mooncake_metadata_port=metadata_port,
        mooncake_metadata_server=config.mooncake.metadata_server,
        mooncake_local_hostname=config.mooncake.local_hostname,
        mooncake_kv_lease_ttl_s=float(config.mooncake.kv_lease_ttl_s),
    )
    if launch_mooncake_master(args) is None:
        sys.exit("mooncake_master failed to launch (binary missing?)")

    # Propagate resolved addresses back into the Hydra config so all Ray workers
    # (vLLM connector + reachability probe) target the same reachable endpoints.
    # This also avoids localhost/IPv6 ambiguity when running on multi-node Ray.
    try:
        resolved_master = getattr(args, "mooncake_master_server_address", None)
        resolved_meta = getattr(args, "mooncake_metadata_server", None)
        if resolved_master:
            config.mooncake.master_server_address = resolved_master
            # Extract host from address, handling IPv6 bracket notation
            if resolved_master.startswith("["):
                # IPv6: [::1]:port -> ::1
                bracket_end = resolved_master.find("]")
                config.mooncake.local_hostname = resolved_master[1:bracket_end]
            else:
                # IPv4: host:port -> host
                config.mooncake.local_hostname = resolved_master.rsplit(":", 1)[0]
        if resolved_meta:
            config.mooncake.metadata_server = resolved_meta
    except Exception:
        # Best-effort; downstream reachability checks will still fail with a clear error.
        pass


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
