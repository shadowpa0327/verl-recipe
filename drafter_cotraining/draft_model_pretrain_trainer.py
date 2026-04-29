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
Each parquet row is a full multi-turn conversation in canonical form:

    {"id": ..., "conversations": [{"role": ..., "content": ...}, ...]}
        -> DataProto(input_ids, attention_mask, position_ids, loss_mask)
        -> HS collector (prefill the full sequence, capture hidden states)
        -> Mooncake sample metadata + per-sample loss_mask
        -> DrafterPretrainWorker.update_drafter

Loss mask semantics (ported from TorchSpec ``preprocess_conversations``):
every assistant content token is supervised (loss_mask=1); user/system/tool
tokens are 0. Truncation past ``data.max_seq_length`` is implicit — partial
assistant turns contribute their surviving prefix.

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
    """Build SampleMeta list from the HS-collector output DataProto.

    The collector returns per-sample Mooncake key + shape/dtype metadata plus
    the per-token loss mask the trainer prepared (carried through unchanged
    so the supervisor positions stay aligned with the prefilled tokens).
    Both ``hs_mooncake_keys`` and ``hs_loss_masks`` are required.
    """
    nt = hs_batch.non_tensor_batch
    if not nt or "hs_mooncake_keys" not in nt:
        return []
    keys = nt["hs_mooncake_keys"]
    if "hs_loss_masks" not in nt:
        raise KeyError(
            "HS-collector output is missing required key 'hs_loss_masks'."
        )
    loss_masks = nt["hs_loss_masks"]
    metas: list[SampleMeta] = []
    for i in range(len(keys)):
        seq_len = int(nt["hs_seq_lens"][i])
        lm = loss_masks[i]
        if lm is None:
            raise ValueError(f"hs_loss_masks[{i}] is None for key {keys[i]!r}.")
        mask_arr = (
            lm.astype(np.int64).reshape(-1)
            if isinstance(lm, np.ndarray)
            else np.asarray(lm, dtype=np.int64).reshape(-1)
        )
        # Defensive: clip / pad to seq_len so downstream code can assume the
        # mask is exactly seq_len long. Length mismatches indicate a producer
        # bug but shouldn't crash the trainer mid-step.
        if mask_arr.shape[0] != seq_len:
            fixed = np.zeros(seq_len, dtype=np.int64)
            n = min(mask_arr.shape[0], seq_len)
            fixed[:n] = mask_arr[:n]
            mask_arr = fixed
        metas.append(
            SampleMeta(
                mooncake_key=str(keys[i]),
                shapes=nt["hs_shapes"][i] if isinstance(nt["hs_shapes"][i], dict) else {},
                dtypes=nt["hs_dtypes"][i] if isinstance(nt["hs_dtypes"][i], dict) else {},
                loss_mask=mask_arr,
                seq_len=seq_len,
                n_tokens=seq_len,
            )
        )
    return metas


class ParquetDrafterPretrainDataset(Dataset):
    """Read canonical multi-turn conversation rows from parquet.

    Expected schema (produced by ``scripts/data_preprocess/jsonl_to_parquet.py``):

        id:            string
        conversations: list<struct<role: string, content: string>>

    The dataset returns ``(id, conversations)`` per row; tokenization and
    loss-mask construction happen in the collator so the dataset can stay
    cheap and tokenizer-free.
    """

    def __init__(
        self,
        data_files,
        conversations_key: str = "conversations",
        id_key: str = "id",
        max_samples: int = -1,
    ):
        import pandas as pd

        from verl.utils.fs import copy_local_path_from_hdfs
        from verl.utils.py_functional import convert_nested_value_to_list_recursive

        if not isinstance(data_files, list | ListConfig):
            data_files = [data_files]

        samples: list[tuple[str, list[dict[str, str]]]] = []
        for data_file in data_files:
            local_path = copy_local_path_from_hdfs(data_file, verbose=True)
            dataframe = pd.read_parquet(local_path, dtype_backend="pyarrow")
            if conversations_key not in dataframe.columns:
                raise ValueError(
                    f"{local_path} is missing required column {conversations_key!r}"
                )

            for idx, row in dataframe.iterrows():
                conv = convert_nested_value_to_list_recursive(row[conversations_key])
                if not isinstance(conv, list) or not conv:
                    continue
                row_id = str(row[id_key]) if id_key in dataframe.columns else f"{idx}"
                samples.append((row_id, conv))
                if max_samples > 0 and len(samples) >= max_samples:
                    break
            if max_samples > 0 and len(samples) >= max_samples:
                break

        if not samples:
            raise ValueError(f"No drafter pretrain samples found in {data_files}")
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[str, list[dict[str, str]]]:
        return self.samples[index]


def _build_pretrain_dataproto(
    samples: list[tuple[str, torch.Tensor, torch.Tensor]],
    pad_token_id: int,
    max_seq_length: int,
) -> DataProto:
    """Right-pad input_ids/attention_mask/loss_mask to ``max_seq_length`` and
    pack into a ``DataProto`` shaped ``[B, T]``.

    Includes ``seq_lens`` (per-sample valid token count) in the non-tensor
    batch so downstream consumers can slice off padding without re-summing
    the attention mask. Carries ``loss_masks`` per sample as a list-of-arrays
    in the non-tensor batch — these are forwarded through HS collection back
    to the drafter, which uses them to set up per-turn supervision.
    """
    batch_size = len(samples)
    input_ids = torch.full((batch_size, max_seq_length), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_seq_length), dtype=torch.long)
    loss_mask = torch.zeros((batch_size, max_seq_length), dtype=torch.long)
    seq_lens = np.zeros(batch_size, dtype=np.int64)
    loss_masks_obj = np.empty(batch_size, dtype=object)
    uids = np.empty(batch_size, dtype=object)

    for i, (uid, ids, mask) in enumerate(samples):
        n = ids.shape[0]
        n = min(n, max_seq_length)
        input_ids[i, :n] = ids[:n]
        attention_mask[i, :n] = 1
        loss_mask[i, :n] = mask[:n]
        seq_lens[i] = n
        loss_masks_obj[i] = mask[:n].numpy().astype(np.int64)
        uids[i] = uid

    position_ids = (attention_mask.cumsum(dim=1) - 1).clamp(min=0) * attention_mask

    proto = DataProto.from_single_dict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }
    )
    proto.non_tensor_batch["uid"] = uids
    proto.non_tensor_batch["seq_lens"] = seq_lens
    proto.non_tensor_batch["loss_masks"] = loss_masks_obj
    return proto


class DrafterPretrainCollator:
    """Render → tokenize → build per-turn assistant loss mask → pack to DataProto.

    The collator owns the tokenizer because the per-row work depends on the
    chat template registered by the user. It drops rows that produce a
    zero-supervision mask (matches TorchSpec's ``min_loss_tokens`` filter at
    a min of 1) so the macro-step doesn't waste a slot on a row the drafter
    can't learn from.
    """

    def __init__(
        self,
        tokenizer,
        max_seq_length: int,
        chat_template: str,
        apply_chat_template_kwargs: dict[str, Any] | None = None,
    ):
        from recipe.drafter_cotraining.data_preprocessing import (
            build_input_ids_and_loss_mask,
        )

        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.chat_template = chat_template
        self.apply_chat_template_kwargs = apply_chat_template_kwargs or {}
        self.pad_token_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )
        self._build = build_input_ids_and_loss_mask

    def __call__(
        self, batch_rows: list[tuple[str, list[dict[str, str]]]]
    ) -> DataProto:
        samples: list[tuple[str, torch.Tensor, torch.Tensor]] = []
        skipped = 0
        for row_id, conv in batch_rows:
            try:
                ids, mask = self._build(
                    self.tokenizer,
                    conv,
                    self.chat_template,
                    self.max_seq_length,
                    self.apply_chat_template_kwargs,
                )
            except Exception as exc:  # noqa: BLE001
                skipped += 1
                logger.warning("Skipping malformed pretrain row %s: %s", row_id, exc)
                continue
            if ids.numel() == 0 or int(mask.sum()) == 0:
                skipped += 1
                continue
            samples.append((str(row_id) or str(uuid.uuid4()), ids, mask))

        if not samples:
            raise ValueError(
                f"All {len(batch_rows)} rows in this batch were invalid or had "
                "no assistant tokens to supervise"
            )

        batch = _build_pretrain_dataproto(
            samples=samples,
            pad_token_id=self.pad_token_id,
            max_seq_length=self.max_seq_length,
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
            conversations_key=self.config.data.get("conversations_key", "conversations"),
            id_key=self.config.data.get("id_key", "id"),
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
            max_seq_length=int(data_cfg.max_seq_length),
            chat_template=str(data_cfg.chat_template),
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
        max_seq = int(self.config.data.max_seq_length)
        hs_max_len = int(self.config.hs_collector.inference.max_model_len)
        if hs_max_len <= max_seq:
            raise ValueError(
                "hs_collector.inference.max_model_len must be greater than "
                "data.max_seq_length — the collector prefills the full "
                "tokenized conversation plus one max_tokens=1 sample."
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
        print(f"  max_seq_length={self.config.data.max_seq_length}")
        print(f"  chat_template={self.config.data.chat_template}")
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
