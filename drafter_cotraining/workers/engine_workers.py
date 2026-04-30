"""
ActorRolloutRefDrafterWorker — extends ActorRolloutRefWorker with EAGLE drafter co-training.

Adds:
- self.drafter: TrainingWorker (FSDPDrafterEngine for EAGLE model training)

HS collection is owned by HSCollectorManager on the trainer driver
(see verl/experimental/hs_collector/), not by this worker.

This worker:
- Receives per-rank data via mesh dispatch (update_drafter)
- Fetches tensors from Mooncake
- Runs drafter training

See claude_docs/rfc-drafter-trainer-integration.md for the full design.
"""

import logging
import math
from typing import List, Optional

import numpy as np
import torch
from omegaconf import DictConfig

from verl.protocol import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.workers.engine_workers import ActorRolloutRefWorker

logger = logging.getLogger(__name__)


class ActorRolloutRefDrafterWorker(ActorRolloutRefWorker):
    """Extends ActorRolloutRefWorker with EAGLE drafter co-training.

    Worker hierarchy:
        self.actor           (TrainingWorker → FSDPEngine)       [inherited]
        self.ref             (TrainingWorker → FSDPEngine)       [inherited]
        self.rollout         (BaseRollout → vLLM)                [inherited]
        self.drafter         (TrainingWorker → FSDPDrafterEngine)       [NEW]

    HS collection lives on the driver (HSCollectorManager), not in this worker.
    Per-step order: rollout generates → HS collector (driver-owned) prefills → drafter trains → actor updates.
    """

    def __init__(self, config: DictConfig, role: str, **kwargs):
        super().__init__(config, role, **kwargs)
        self.drafter = None
        drafter_cfg = config.get("drafter", {}) or {}
        self._drafter_enabled = drafter_cfg.get("enable", False)
        self._mooncake_store = None  # lazy-init in update_drafter

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # 1-4. Actor, ref, rollout, checkpoint — all inherited
        super().init_model()

        if not self._drafter_enabled:
            return

        # 5. Build drafter training engine (skipped when no model config supplied —
        #    lets you smoke-test the rollout → HS collector → mesh dispatch →
        #    update_drafter (Mooncake fetch + collate + print) path without
        #    a real Eagle3 model checkpoint).
        drafter_cfg = self.config.get("drafter", {}) or {}
        model_cfg = drafter_cfg.get("model_config", {}) or {}
        # Drafter engine init is gated on target_model_path: the auto-derive
        # path needs the target's HF AutoConfig to spec the draft architecture,
        # and the frozen-weight load needs target safetensors. Without it we
        # fall back to the shape-print smoke (rollout+HS dispatch only).
        if model_cfg.get("target_model_path"):
            self._init_drafter()
        else:
            logger.info(
                "drafter.model_config.target_model_path not set — skipping "
                "_init_drafter; update_drafter will run the fetch+collate "
                "smoke path only."
            )

        # 6. Register drafter mesh (pure DP — every rank is unique).
        import torch.distributed as dist
        self._register_dispatch_collect_info(
            mesh_name="drafter",
            dp_rank=dist.get_rank(),  # world_rank = dp_rank (pure DP)
            is_collect=True,
        )

        logger.info("Drafter co-training initialized")

    def _init_drafter(self):
        """Initialize the drafter training engine.

        Uses FSDPDrafterEngine (registered as model_type="drafter_model").
        Shares embed_tokens/lm_head from actor (frozen, zero copy).
        """
        from verl.trainer.config import CheckpointConfig
        from verl.workers.config import FSDPEngineConfig, FSDPOptimizerConfig
        from verl.workers.engine_workers import TrainingWorker, TrainingWorkerConfig

        from recipe.drafter_cotraining.workers.drafter_engine import (
            DrafterModelConfig,
            build_drafter_subconfig,
        )

        from omegaconf import DictConfig, OmegaConf

        drafter_cfg = self.config.drafter

        # Surface the top-level ``drafter.model_path`` (pretrained drafter
        # checkpoint dir) into the nested ``model_config`` dict so the
        # DrafterModelConfig dataclass picks it up. The top-level location
        # mirrors how users naturally write the knob in YAML/CLI overrides;
        # the engine reads it as ``model_config.model_path`` internally.
        raw_model_cfg = drafter_cfg.get("model_config", {}) or {}
        if isinstance(raw_model_cfg, DictConfig):
            raw_model_cfg = OmegaConf.to_container(raw_model_cfg, resolve=True)
        else:
            raw_model_cfg = dict(raw_model_cfg)
        top_level_model_path = drafter_cfg.get("model_path", None)
        if top_level_model_path is not None and "model_path" not in raw_model_cfg:
            raw_model_cfg["model_path"] = top_level_model_path

        drafter_training_config = TrainingWorkerConfig(
            model_type="drafter_model",
            model_config=build_drafter_subconfig(
                raw_model_cfg, DrafterModelConfig
            ),
            engine_config=build_drafter_subconfig(
                drafter_cfg.get("engine_config", {}), FSDPEngineConfig
            ),
            optimizer_config=build_drafter_subconfig(
                drafter_cfg.get("optimizer_config", {}), FSDPOptimizerConfig
            ),
            checkpoint_config=build_drafter_subconfig(
                drafter_cfg.get("checkpoint_config", {}), CheckpointConfig
            ),
        )
        self.drafter = TrainingWorker(config=drafter_training_config)
        self.drafter.reset()

        # Set custom loss function (Forward KL for EAGLE)
        if drafter_cfg.get("loss_fn", None) == "eagle_forward_kl":
            from recipe.drafter_cotraining.eagle3.ops.loss import compiled_forward_kl_loss
            self.drafter.set_loss_fn(compiled_forward_kl_loss)

        # Initial sync of frozen modules (embed_tokens, lm_head) from actor.
        # These are weight copies, not references — must be re-synced after
        # each update_actor() since the actor trains every RL step.
        self._sync_drafter_frozen_modules()

        logger.info("Drafter TrainingWorker initialized")

    def _sync_drafter_frozen_modules(self):
        """Copy frozen weights from actor into drafter (co-training only).

        Syncs: embed_tokens, target_lm_head_weight, verifier_norm (final RMSNorm).
        All frozen (requires_grad=False). embed_tokens for drafter input,
        target_lm_head_weight for target distribution, verifier_norm for
        pre-norm correction. The drafter's own lm_head is trainable and NOT synced.

        Called at init and after each update_actor(). In verl the actor trains
        every RL step (unlike TorchSpec where the target is fixed), so the
        drafter's frozen copies must stay synchronized.

        TODO(co-training): Implement FSDP-aware gathering of actor params.
        Currently only works when actor/drafter are on the same FSDP unit.
        Pretrain worker overrides this with a no-op (frozen weights from disk).
        """
        if self.actor is None or self.drafter is None:
            return
        if not hasattr(self.drafter.engine, "sync_frozen_modules_from_actor"):
            return

        actor_module = self.actor.engine.module

        # Get actor's final norm (model.norm — the RMSNorm before lm_head).
        # This is the "verifier_norm" needed because vLLM captures
        # last_hidden_states pre-norm.
        actor_norm = None
        if hasattr(actor_module, "model") and hasattr(actor_module.model, "norm"):
            actor_norm = actor_module.model.norm

        self.drafter.engine.sync_frozen_modules_from_actor(
            actor_embed_tokens=actor_module.model.embed_tokens,
            actor_lm_head=actor_module.lm_head,
            actor_norm=actor_norm,
        )

    # ── Drafter Training ──────────────────────────────────────

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="drafter"))
    def update_drafter(self, data: DataProto):
        """Receive per-rank shard of Mooncake keys, fetch tensors paged, train.

        Macro-step shape (mirrors verl's canonical forward_backward_batch):
          1. Filter out samples with empty loss-mask (pure metadata, no fetch).
          2. Preflight total_valid_global across DP for an exact mean divisor.
          3. Pre-compute T_pad_macro across all micro-batches so torch.compile
             doesn't recompile per micro-batch.
          4. engine.train_mode: per micro-batch, paged Mooncake.get → forward+TTT
             → weighted backward (set_requires_gradient_sync(is_last) on FSDP2
             suppresses inter-rank reduce-scatter on all-but-last micro-batch).
          5. optimizer_step + lr_scheduler_step.
          6. Aggregate per-mb metrics → DP all-reduce → meta_info.

        When the drafter engine is absent the function falls back to the
        single-shot shape-print smoke path so the rollout+HS+dispatch pipeline
        is still exercisable without a real Eagle3 checkpoint.
        """
        if len(data) == 0:
            return DataProto(non_tensor_batch={})

        mooncake_keys = data.non_tensor_batch.get("mooncake_keys", [])
        if len(mooncake_keys) == 0:
            return DataProto(non_tensor_batch={})

        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_initialized() else 0

        # Smoke-only fallback (no drafter engine): single-shot fetch + log shapes.
        if self.drafter is None:
            batch = self._fetch_drafter_batch_from_mooncake(data, rank)
            if batch is None:
                return DataProto(non_tensor_batch=data.non_tensor_batch)
            self._log_drafter_batch_shapes(batch, rank)
            return DataProto(non_tensor_batch=data.non_tensor_batch)

        # ── Step 1: metadata-only empty-mask filter (TorchSpec data_fetcher.py:177)
        loss_masks = data.non_tensor_batch.get("loss_masks", None)
        valid_counts = self._compute_valid_counts(data)
        if loss_masks is not None and len(loss_masks) > 0:
            keep = [v > 0 for v in valid_counts]
            n_dropped = sum(1 for k in keep if not k)
            if n_dropped > 0:
                logger.warning(
                    "[drafter rank=%d] dropping %d samples with zero loss-mask positions",
                    rank, n_dropped,
                )
                # Eagerly free Mooncake keys for dropped samples so producer can
                # reuse buffers — mirrors per-key remove_eagle3_tensors below.
                store = self._get_mooncake_store(data.meta_info.get("mooncake_cfg", {}), rank)
                if store is not None:
                    for i, k in enumerate(keep):
                        if not k:
                            try:
                                store.remove_eagle3_tensors(
                                    key=str(mooncake_keys[i]),
                                    has_last_hidden_states=True,
                                )
                            except Exception as exc:
                                logger.debug(
                                    "Mooncake remove for dropped sample %d failed: %s", i, exc,
                                )
                kept_indices = [i for i, k in enumerate(keep) if k]
                data = self._select_data_indices(data, kept_indices)
                mooncake_keys = data.non_tensor_batch.get("mooncake_keys", [])
                valid_counts = self._compute_valid_counts(data)

        if len(mooncake_keys) == 0:
            logger.warning("[drafter rank=%d] all samples filtered; skipping macro-step", rank)
            return DataProto(non_tensor_batch=data.non_tensor_batch)

        # ── Step 2: preflight total_valid_global from metadata (no fetch needed)
        local_total_valid = int(sum(valid_counts))
        total_valid_global = self._allreduce_sum_int(local_total_valid)
        if total_valid_global == 0:
            logger.warning("[drafter rank=%d] total_valid_global=0; skipping", rank)
            return DataProto(non_tensor_batch=data.non_tensor_batch)

        # ── Step 3: T_pad_macro across all micro-batches AND across DP ranks.
        # All-reducing MAX makes T_pad identical on every rank → torch.compile
        # cache doesn't fragment per-rank. Also keeps the resulting
        # train_metrics dict bitwise-equal across ranks (verl's DataProto.concat
        # asserts on conflicting meta_info values).
        seq_lens_list = data.non_tensor_batch.get("seq_lens", [])
        local_t_pad = int(max(seq_lens_list)) if len(seq_lens_list) > 0 else 0
        t_pad_macro = self._allreduce_max_int(local_t_pad)

        # ── Step 4-5: micro-batch loop + optimizer step
        # self.config is already the actor_rollout_ref slice (see trainer/ray_trainer.py:717
        # and trainer/pretrain_trainer.py:288).
        micro_size = int(
            self.config.drafter.engine_config.get("micro_batch_size_per_gpu", 1)
        )
        accum_steps = max(1, math.ceil(len(mooncake_keys) / micro_size))
        metrics = self._drafter_train_step_micro(
            data,
            rank=rank,
            micro_size=micro_size,
            accum_steps=accum_steps,
            total_valid_global=total_valid_global,
            t_pad_macro=t_pad_macro,
        )

        # Don't add per-rank diagnostic fields to train_metrics — verl's
        # DataProto.concat asserts on conflicting values and per-rank
        # accum_steps may differ under uneven dispatch. Log on rank 0 instead.
        if rank == 0:
            logger.info(
                "[drafter] macro-step done: accum_steps=%d total_valid_global=%d t_pad_macro=%d",
                accum_steps, total_valid_global, t_pad_macro,
            )

        return DataProto(
            non_tensor_batch=data.non_tensor_batch,
            meta_info={"train_metrics": metrics},
        )

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="drafter"))
    def evaluate_drafter(self, data: DataProto):
        """Receive Mooncake keys, fetch tensors, and report drafter eval metrics.

        This mirrors update_drafter's fetch/collate path but runs the Eagle3
        forward under no_grad and never steps the optimizer.
        """
        if len(data) == 0:
            return DataProto(non_tensor_batch={})

        mooncake_keys = data.non_tensor_batch.get("mooncake_keys", [])
        if len(mooncake_keys) == 0:
            return DataProto(non_tensor_batch={})

        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_initialized() else 0

        batch = self._fetch_drafter_batch_from_mooncake(data, rank)
        if batch is None:
            return DataProto(non_tensor_batch=data.non_tensor_batch)

        if self.drafter is None:
            self._log_drafter_batch_shapes(batch, rank)
            return DataProto(non_tensor_batch=data.non_tensor_batch)

        metrics = self._drafter_eval_step(batch, rank)
        return DataProto(
            non_tensor_batch=data.non_tensor_batch,
            meta_info={"eval_metrics": metrics},
        )

    def _fetch_drafter_batch_from_mooncake(
        self, data: DataProto, rank: int, t_pad_override: Optional[int] = None,
    ):
        """Fetch tensors for the keys in `data` and collate to a rectangular batch.

        Reads the per-token assistant loss_mask out of
        ``data.non_tensor_batch['loss_masks']`` (one ``np.ndarray`` per
        sample, matching the prefilled token sequence) and aligns it to the
        Mooncake-returned ``input_ids`` length. This replaces the older
        prompt+response-derived mask path; we now supervise every assistant
        content token across all turns (TorchSpec semantics).

        t_pad_override: when provided, collator pads to at least this length
            (still snapped to a 256-token bucket). Lets the macro-step
            pre-compute T_pad once across all micro-batches so torch.compile
            doesn't recompile per micro-batch.
        """
        mooncake_keys = data.non_tensor_batch.get("mooncake_keys", [])
        shapes_list = data.non_tensor_batch.get("shapes", [])
        dtypes_list = data.non_tensor_batch.get("dtypes", [])
        loss_masks = data.non_tensor_batch.get("loss_masks", None)
        if loss_masks is None or len(loss_masks) != len(mooncake_keys):
            raise KeyError(
                "DataProto.non_tensor_batch must carry 'loss_masks' aligned 1:1 "
                "with 'mooncake_keys'; the drafter pipeline supervises only "
                "the assistant tokens identified by this mask."
            )

        store = self._get_mooncake_store(data.meta_info.get("mooncake_cfg", {}), rank)
        if store is None:
            return None

        device = torch.device("cuda", torch.cuda.current_device())

        from recipe.drafter_cotraining.data.collator import DataCollatorWithPadding
        collator = DataCollatorWithPadding()

        features = []
        for i in range(len(mooncake_keys)):
            key = str(mooncake_keys[i])
            shapes = shapes_list[i] if isinstance(shapes_list[i], dict) else {}
            raw_dtypes = dtypes_list[i] if isinstance(dtypes_list[i], dict) else {}
            dtypes = {
                k: (getattr(torch, v) if isinstance(v, str) and hasattr(torch, v) else v)
                for k, v in raw_dtypes.items()
            }
            out = store.get(key=key, shapes=shapes, dtypes=dtypes, device=device)
            # Add batch dim — collator expects [1, T] / [1, T, D] per sample.
            ids = out.input_ids.unsqueeze(0) if out.input_ids.dim() == 1 else out.input_ids
            hs = out.hidden_states.unsqueeze(0) if out.hidden_states.dim() == 2 else out.hidden_states
            seq_len = int(ids.shape[-1])

            # Trainer-supplied per-token assistant mask. seq_len here is what
            # Mooncake actually returned (same as the prefill input length),
            # so we just truncate / right-pad with zeros to match.
            raw_mask = loss_masks[i]
            if raw_mask is None:
                raise ValueError(
                    f"loss_masks[{i}] is None for key={key!r}; "
                    "per-sample mask is required."
                )
            if isinstance(raw_mask, torch.Tensor):
                mask_np = raw_mask.detach().cpu().numpy().astype(np.int64).reshape(-1)
            elif isinstance(raw_mask, np.ndarray):
                mask_np = raw_mask.astype(np.int64).reshape(-1)
            else:
                mask_np = np.asarray(raw_mask, dtype=np.int64).reshape(-1)
            if mask_np.shape[0] >= seq_len:
                mask_np = mask_np[:seq_len]
            else:
                fixed = np.zeros(seq_len, dtype=np.int64)
                fixed[: mask_np.shape[0]] = mask_np
                mask_np = fixed
            loss_mask = torch.from_numpy(mask_np).long().to(ids.device).unsqueeze(0)

            feat = {
                "input_ids": ids,
                "hidden_states": hs,
                "loss_mask": loss_mask,
            }
            if out.last_hidden_states is not None:
                lhs = out.last_hidden_states
                feat["last_hidden_states"] = lhs.unsqueeze(0) if lhs.dim() == 2 else lhs
            features.append(feat)
            # Force-delete Mooncake keys right after fetch (matches TorchSpec
            # data_fetcher._cleanup_mooncake_data — frees the buffer for the
            # next prefill while the GPU consumes this sample).
            store.remove_eagle3_tensors(
                key=key,
                has_last_hidden_states=out.last_hidden_states is not None,
            )

        return collator(features, bucket_size_override=t_pad_override)

    def _log_drafter_batch_shapes(self, batch, rank: int):
        lhs_shape = (
            tuple(batch["last_hidden_states"].shape) if "last_hidden_states" in batch else None
        )
        print("=" * 100)
        print(
            f"[drafter rank {rank}] B={batch['input_ids'].shape[0]} "
            f"T_pad={batch['input_ids'].shape[1]}  "
            f"input_ids={tuple(batch['input_ids'].shape)} "
            f"hidden_states={tuple(batch['hidden_states'].shape)} "
            f"last_hs={lhs_shape} "
            f"attn_mask={tuple(batch['attention_mask'].shape)} "
            f"loss_mask={tuple(batch['loss_mask'].shape)}"
        )
        print("=" * 100)

    # ── Micro-batch helpers ──────────────────────────────────

    def _compute_valid_counts(self, data: DataProto) -> List[int]:
        """Per-sample count of supervised positions (loss_mask sum).

        Used as the empty-mask filter and as the per-mb / total scale divisor.
        Required: ``data.non_tensor_batch['loss_masks']`` must be present —
        the upstream pipeline always sets it, and there's no sensible
        fallback for the macro-step divisor.
        """
        loss_masks = data.non_tensor_batch.get("loss_masks", None)
        if loss_masks is None:
            raise KeyError(
                "DataProto.non_tensor_batch is missing required key 'loss_masks'."
            )
        counts: List[int] = []
        for i, lm in enumerate(loss_masks):
            if lm is None:
                raise ValueError(f"loss_masks[{i}] is None; per-sample mask is required.")
            if isinstance(lm, np.ndarray):
                counts.append(int(lm.astype(np.int64).sum()))
            elif isinstance(lm, torch.Tensor):
                counts.append(int(lm.long().sum().item()))
            else:
                counts.append(int(np.asarray(lm, dtype=np.int64).sum()))
        return counts

    def _allreduce_sum_int(self, value: int) -> int:
        """All-reduce SUM of a Python int across the drafter DP group."""
        import torch.distributed as dist
        if not dist.is_initialized():
            return int(value)
        device = torch.device("cuda", torch.cuda.current_device())
        t = torch.tensor([int(value)], device=device, dtype=torch.long)
        dp_group = self.drafter.engine.get_data_parallel_group()
        dist.all_reduce(t, op=dist.ReduceOp.SUM, group=dp_group)
        return int(t.item())

    def _allreduce_max_int(self, value: int) -> int:
        """All-reduce MAX of a Python int across the drafter DP group.

        Used for T_pad_macro so all ranks pad to the same length →
        torch.compile cache stays unified across ranks AND collated metrics
        don't conflict.
        """
        import torch.distributed as dist
        if not dist.is_initialized():
            return int(value)
        device = torch.device("cuda", torch.cuda.current_device())
        t = torch.tensor([int(value)], device=device, dtype=torch.long)
        dp_group = self.drafter.engine.get_data_parallel_group()
        dist.all_reduce(t, op=dist.ReduceOp.MAX, group=dp_group)
        return int(t.item())

    def _select_data_indices(self, data: DataProto, indices: List[int]) -> DataProto:
        """Filter a DataProto's non_tensor_batch (and tensor batch if present)
        to the given indices. Used by the metadata-time empty-mask filter and
        by the per-micro-batch slicer."""
        new_nt = {}
        for k, v in data.non_tensor_batch.items():
            if isinstance(v, np.ndarray):
                new_nt[k] = v[np.asarray(indices, dtype=np.int64)]
            elif isinstance(v, list):
                new_nt[k] = [v[i] for i in indices]
            else:
                new_nt[k] = v
        new_tb = None
        if data.batch is not None:
            new_tb = data.batch[indices]
        return DataProto(batch=new_tb, non_tensor_batch=new_nt, meta_info=dict(data.meta_info))

    def _iter_micro_batch_keys(self, data: DataProto, micro_size: int):
        """Yield (mb_idx, sub_data) DataProto slices over Mooncake keys."""
        n = len(data.non_tensor_batch.get("mooncake_keys", []))
        if n == 0:
            return
        for mb_idx, start in enumerate(range(0, n, micro_size)):
            end = min(start + micro_size, n)
            sub = self._select_data_indices(data, list(range(start, end)))
            yield mb_idx, sub

    def _drafter_train_step_micro(
        self,
        data: DataProto,
        rank: int,
        micro_size: int,
        accum_steps: int,
        total_valid_global: int,
        t_pad_macro: int,
    ) -> dict:
        """Outer loop: train_mode + per-micro-batch fetch+forward+backward + optimizer step.

        Mirrors verl's canonical forward_backward_batch divisor pattern: each
        micro-batch's contribution is `mb_valid / total_valid_global`, so summing
        N backwards reproduces single-batch mean semantics exactly.

        FSDP2-only: set_requires_gradient_sync(is_last) suppresses inter-rank
        reduce-scatter on all-but-last micro-batch (mirrors TorchSpec). On FSDP1
        the attribute is missing → no-op via getattr guard.
        """
        engine = self.drafter.engine
        device = torch.device("cuda", torch.cuda.current_device())

        fsdp_root = engine.module
        set_grad_sync = getattr(fsdp_root, "set_requires_gradient_sync", None)

        accum_metrics = []
        with engine.train_mode():
            for mb_idx, mb_data in self._iter_micro_batch_keys(data, micro_size):
                is_last = mb_idx == accum_steps - 1
                if set_grad_sync is not None:
                    set_grad_sync(is_last)

                mb_batch = self._fetch_drafter_batch_from_mooncake(
                    mb_data, rank, t_pad_override=t_pad_macro,
                )
                if mb_batch is None:
                    continue

                # First-mb-only shape print (smoke continuity).
                if mb_idx == 0 and not getattr(self, "_drafter_shapes_logged", False):
                    self._log_drafter_batch_shapes(mb_batch, rank)
                    self._drafter_shapes_logged = True

                mb_metrics = self._drafter_micro_step(
                    mb_batch,
                    rank=rank,
                    total_valid_global=total_valid_global,
                    device=device,
                )
                accum_metrics.append(mb_metrics)

            # Re-enable sync before optimizer step (defensive; optimizer_step
            # itself doesn't trigger comms but keeps engine state predictable).
            if set_grad_sync is not None:
                set_grad_sync(True)

            grad_norm = engine.optimizer_step()
            lr = engine.lr_scheduler_step()

        return self._aggregate_micro_metrics(accum_metrics, grad_norm, lr, rank)

    def _drafter_micro_step(
        self, mb_batch, rank: int, total_valid_global: int, device,
    ) -> dict:
        """Per-micro-batch: prepare → forward → weighted backward → free."""
        engine = self.drafter.engine
        batch_dev = {
            k: (v.to(device) if torch.is_tensor(v) else v) for k, v in mb_batch.items()
        }
        prepared = engine.prepare_model_inputs(batch_dev)
        plosses, _, acces = engine.module(**prepared)

        num_ttt = len(plosses)
        loss_weights = [0.8 ** i for i in range(num_ttt)]

        # Local valid count drives the per-mb scale factor. Use position_mask
        # (vocab pruning subset) when present; else use loss_mask.
        target_obj = prepared["target"]
        position_mask = getattr(target_obj, "position_mask", None)
        if position_mask is not None:
            mb_valid = int(position_mask.sum().item())
        else:
            mb_valid = int(prepared["loss_mask"].sum().item())

        if mb_valid == 0 or total_valid_global <= 0:
            # Skip backward; in-kernel fallback already produced zero-grad
            # touching all params, so FSDP's reduce-scatter on the LAST mb still
            # works. Don't accumulate this mb's loss into the macro divisor.
            return {
                "plosses": [p.detach() for p in plosses],
                "acces": [a.detach() for a in acces],
                "loss_weights": loss_weights,
                "mb_valid": 0,
            }

        # Exact mean divisor with FSDP grad-averaging cancellation:
        #   per-rank: L_r = Σ_k (mb_valid_k / total_valid_global) · per_pos_mean_k
        #   Σ_r L_r = global-token mean (the construction).
        # FSDP2 reduce-scatter averages grads across DP → grad lands as
        # grad(L_global) / dp_size. Multiplying scale by dp_size cancels that
        # averaging so backward sees grad(L_global). Mirrors verl's canonical
        # global-token loss pattern (verl/workers/utils/losses.py:50,
        # verl/trainer/ppo/core_algos.py:1173/1181/1195: `... / batch_num_tokens * dp_size`).
        scale = mb_valid / total_valid_global * engine.get_data_parallel_size()
        weighted = sum(w * p * scale for w, p in zip(loss_weights, plosses))
        weighted.backward()

        out = {
            "plosses": [p.detach() for p in plosses],
            "acces": [a.detach() for a in acces],
            "loss_weights": loss_weights,
            "mb_valid": mb_valid,
        }
        # Eagerly free heavy tensors before next fetch.
        del prepared, plosses, acces, weighted, batch_dev, target_obj
        return out

    def _aggregate_micro_metrics(
        self, accum_metrics: list, grad_norm, lr, rank: int,
    ) -> dict:
        """Combine per-micro-batch losses (already weighted by mb_valid/total) and
        accuracies (weighted by mb_valid). Reuses _aggregate_drafter_metrics for
        the DP all-reduce + final dict shape so smoke output is unchanged."""
        if not accum_metrics:
            # No effective work this macro-step; return a minimal metrics dict.
            return {
                "train/loss_weighted": 0.0,
                "train/avg_acc": 0.0,
                "train/simulated_acc_len": 0.0,
            }
        num_ttt = len(accum_metrics[0]["plosses"])
        loss_weights = accum_metrics[0]["loss_weights"]
        total_valid = sum(m["mb_valid"] for m in accum_metrics)

        device = accum_metrics[0]["plosses"][0].device
        if total_valid == 0:
            zero = torch.zeros((), device=device)
            combined_plosses = [zero for _ in range(num_ttt)]
            combined_acces = [zero for _ in range(num_ttt)]
        else:
            combined_plosses = []
            combined_acces = []
            for ttt_i in range(num_ttt):
                ploss_sum = sum(
                    m["plosses"][ttt_i] * m["mb_valid"] for m in accum_metrics
                ) / total_valid
                acc_sum = sum(
                    m["acces"][ttt_i] * m["mb_valid"] for m in accum_metrics
                ) / total_valid
                combined_plosses.append(ploss_sum)
                combined_acces.append(acc_sum)

        return self._aggregate_drafter_metrics(
            plosses=combined_plosses,
            acces=combined_acces,
            loss_weights=loss_weights,
            grad_norm=grad_norm,
            lr=lr,
            rank=rank,
            prefix="train",
        )

    def _drafter_eval_step(self, batch, rank: int) -> dict:
        """Eagle3 forward for evaluation only; no backward or optimizer step."""
        engine = self.drafter.engine
        device = torch.device("cuda", torch.cuda.current_device())

        batch_dev = {
            k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
        }

        with engine.eval_mode(), torch.no_grad():
            prepared = engine.prepare_model_inputs(batch_dev)
            plosses, _, acces = engine.module(**prepared)

        loss_weights = [0.8 ** i for i in range(len(plosses))]
        return self._aggregate_drafter_metrics(
            plosses=plosses,
            acces=acces,
            loss_weights=loss_weights,
            grad_norm=None,
            lr=None,
            rank=rank,
            prefix="val",
        )

    def _aggregate_drafter_metrics(
        self, plosses, acces, loss_weights, grad_norm, lr, rank: int, prefix: str = "train"
    ) -> dict:
        """All-reduce per-TTT-step plosses/acces across DP and build metrics dict."""
        import torch.distributed as dist

        # Each ploss / acc is a scalar tensor per TTT step; stack to [L].
        avg_plosses = torch.stack([p.detach() for p in plosses])
        avg_acces = torch.stack([a.detach() for a in acces])

        dp_group = self.drafter.engine.get_data_parallel_group()
        dist.all_reduce(avg_plosses, op=dist.ReduceOp.AVG, group=dp_group)
        dist.all_reduce(avg_acces, op=dist.ReduceOp.AVG, group=dp_group)

        plosses_list = avg_plosses.float().tolist()
        acces_list = avg_acces.float().tolist()

        # simulated_acc_len = acc_0 + acc_0·acc_1 + acc_0·acc_1·acc_2 + ...
        cumulative, simulated_acc_len = 1.0, 0.0
        for a in acces_list:
            cumulative *= a
            simulated_acc_len += cumulative

        w = torch.tensor(loss_weights, device=avg_plosses.device, dtype=avg_plosses.dtype)
        weighted_loss = (avg_plosses * w).sum().item() / float(sum(loss_weights))
        raw_mean_loss = avg_plosses.float().mean().item()
        avg_acc = sum(acces_list) / max(len(acces_list), 1)

        metrics = {
            f"{prefix}/loss_weighted": float(weighted_loss),
            f"{prefix}/loss_raw_mean": float(raw_mean_loss),
            f"{prefix}/avg_acc": float(avg_acc),
            f"{prefix}/simulated_acc_len": float(simulated_acc_len),
        }
        if grad_norm is not None:
            metrics[f"{prefix}/grad_norm"] = float(grad_norm)
        if lr is not None:
            metrics[f"{prefix}/lr"] = float(lr)
        for i, p in enumerate(plosses_list):
            metrics[f"{prefix}/ploss_{i}"] = float(p)
        for i, a in enumerate(acces_list):
            metrics[f"{prefix}/acc_{i}"] = float(a)

        if rank == 0:
            print(
                f"[drafter {prefix}] loss={metrics[f'{prefix}/loss_weighted']:.4f} "
                f"acc0={metrics[f'{prefix}/acc_0']:.4f} "
                f"acc_len={metrics[f'{prefix}/simulated_acc_len']:.2f} "
                f"grad={metrics.get(f'{prefix}/grad_norm', 0.0):.3f} "
                f"lr={metrics.get(f'{prefix}/lr', 0.0):.2e}"
            )

        return metrics

    def _get_mooncake_store(self, mooncake_cfg: dict, rank: int):
        """Lazy-init a single EagleMooncakeStore, cached on self."""
        if self._mooncake_store is not None:
            return self._mooncake_store
        if not mooncake_cfg:
            logger.warning("[drafter rank %d] no mooncake_cfg in meta_info; skipping fetch", rank)
            return None
        from recipe.drafter_cotraining.mooncake import EagleMooncakeStore
        from recipe.drafter_cotraining.mooncake.config import MooncakeConfig
        mc_cfg = MooncakeConfig(**mooncake_cfg)
        device = torch.device("cuda", torch.cuda.current_device())
        store = EagleMooncakeStore(mc_cfg)
        store.setup(device=device)
        self._mooncake_store = store
        return store

    # ── Checkpoint ────────────────────────────────────────────

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_drafter_checkpoint(
        self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None
    ):
        """Save drafter-engine checkpoint. Bypasses the parent's actor-only assert.

        ``self.drafter`` is a TrainingWorker whose ``save_checkpoint`` forwards
        to ``FSDPDrafterEngine.save_checkpoint``, which writes sharded FSDP
        state + ``huggingface/{config.json, model.safetensors}``.
        """
        if self.drafter is None:
            return
        self.drafter.save_checkpoint(local_path, hdfs_path, global_step, max_ckpt_to_keep)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_drafter_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        """Load drafter-engine checkpoint.

        ``self.drafter`` is a TrainingWorker whose ``load_checkpoint`` forwards
        to ``FSDPDrafterEngine.load_checkpoint``, which restores sharded FSDP
        state + optimizer state.
        """
        if self.drafter is None:
            return
        self.drafter.load_checkpoint(local_path, hdfs_path, del_local_after_load)

    # ── Weight Sync ───────────────────────────────────────────

    # TODO(co-training): Restore _load_rollout_drafter_snapshot and
    # _iter_rollout_drafter_snapshot when re-enabling the rollout weight
    # sync test (requires vllm_rollout drafter APIs). Removed for
    # pretrain-only scope.

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def update_rollout_drafter_weights_from_snapshot(self, mode: str = "restore", seed: int = 2026):
        """TEST_ONLY: push random/restored drafter weights through verl IPC.

        TODO(co-training): Requires rollout.get_drafter_weights() and
        rollout.update_drafter_weights() APIs in vllm_rollout, which were
        reverted for the pretrain-only scope. Re-enable when co-training
        is activated.
        """
        raise NotImplementedError(
            "update_rollout_drafter_weights_from_snapshot requires "
            "rollout.get_drafter_weights / rollout.update_drafter_weights "
            "APIs (reverted for pretrain-only scope). "
            "See TODO(co-training) in weight-sync-flows.md Flow 4."
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def update_weights(self, global_steps: int = None):
        """Sync actor + drafter weights to rollout.

        Extended from parent to also:
        1. Re-sync frozen modules (embed_tokens, lm_head) from actor → drafter
           (actor weights changed during update_actor())
        2. Sync drafter weights to rollout (for speculative decoding)

        Note: actor → HS collector sync is driven from the trainer via
        HSCollectorManager.update_weights, not here.

        TODO(co-training): Step 2 (drafter → rollout) requires the
        update_drafter_weights API in verl/workers/rollout/vllm_rollout/,
        which was reverted for the pretrain-only scope.
        """
        # Actor → rollout (inherited)
        await super().update_weights(global_steps=global_steps)

        # Actor → drafter frozen modules (embed_tokens, lm_head changed after training)
        if self.drafter is not None:
            self._sync_drafter_frozen_modules()

        if self.drafter is not None and self.rollout is not None:
            # TODO(co-training): Drafter → rollout weight sync (Flow 4).
            # Requires update_drafter_weights API in vllm_rollout (reverted
            # for pretrain-only scope). Re-enable when co-training is activated.
            raise NotImplementedError(
                "Drafter → rollout weight sync is not yet implemented. "
                "Required for co-training only (speculative decoding at inference time). "
                "See TODO(co-training) in weight-sync-flows.md Flow 4."
            )


class DrafterPretrainWorker(Worker):
    """Drafter-only worker for standalone pretraining.

    No actor, ref, rollout, or PPO micro-batch config is required. The decorated
    drafter methods are reused from ActorRolloutRefDrafterWorker so the mesh
    dispatch contract stays identical without inheriting actor/rollout state.
    """

    _init_drafter = ActorRolloutRefDrafterWorker._init_drafter
    update_drafter = ActorRolloutRefDrafterWorker.update_drafter
    evaluate_drafter = ActorRolloutRefDrafterWorker.evaluate_drafter
    _fetch_drafter_batch_from_mooncake = ActorRolloutRefDrafterWorker._fetch_drafter_batch_from_mooncake
    _log_drafter_batch_shapes = ActorRolloutRefDrafterWorker._log_drafter_batch_shapes
    # Micro-batch helpers (Phase C):
    _compute_valid_counts = ActorRolloutRefDrafterWorker._compute_valid_counts
    _allreduce_sum_int = ActorRolloutRefDrafterWorker._allreduce_sum_int
    _allreduce_max_int = ActorRolloutRefDrafterWorker._allreduce_max_int
    _select_data_indices = ActorRolloutRefDrafterWorker._select_data_indices
    _iter_micro_batch_keys = ActorRolloutRefDrafterWorker._iter_micro_batch_keys
    _drafter_train_step_micro = ActorRolloutRefDrafterWorker._drafter_train_step_micro
    _drafter_micro_step = ActorRolloutRefDrafterWorker._drafter_micro_step
    _aggregate_micro_metrics = ActorRolloutRefDrafterWorker._aggregate_micro_metrics
    _drafter_eval_step = ActorRolloutRefDrafterWorker._drafter_eval_step
    _aggregate_drafter_metrics = ActorRolloutRefDrafterWorker._aggregate_drafter_metrics
    _get_mooncake_store = ActorRolloutRefDrafterWorker._get_mooncake_store
    save_drafter_checkpoint = ActorRolloutRefDrafterWorker.save_drafter_checkpoint
    load_drafter_checkpoint = ActorRolloutRefDrafterWorker.load_drafter_checkpoint

    def __init__(self, config: DictConfig, role: str = "drafter", **kwargs):
        del role, kwargs
        Worker.__init__(self)
        self.config = config
        self.role = "drafter"
        self.actor = None
        self.ref = None
        self.rollout = None
        self.drafter = None
        self._drafter_enabled = (config.get("drafter", {}) or {}).get("enable", False)
        self._mooncake_store = None

    def _sync_drafter_frozen_modules(self):
        """No-op for pretrain: frozen weights come from target_model_path on disk."""


    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        if not self._drafter_enabled:
            raise ValueError("actor_rollout_ref.drafter.enable must be True for pretraining")

        drafter_cfg = self.config.get("drafter", {}) or {}
        model_cfg = drafter_cfg.get("model_config", {}) or {}
        if not model_cfg.get("target_model_path"):
            raise ValueError(
                "actor_rollout_ref.drafter.model_config.target_model_path must be set "
                "for drafter pretraining"
            )

        self._init_drafter()

        import torch.distributed as dist

        self._register_dispatch_collect_info(
            mesh_name="drafter",
            dp_rank=dist.get_rank(),
            is_collect=True,
        )
        logger.info("Drafter pretrain worker initialized")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def to(self, device, model=True, optimizer=True, grad=True):
        if self.drafter is None:
            return
        self.drafter.to(device=device, model=model, optimizer=optimizer, grad=grad)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def sleep(self):
        self.to("cpu", model=True, optimizer=True, grad=True)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def wake_up(self):
        self.to("device", model=True, optimizer=True, grad=True)
