"""
ActorRolloutRefDrafterWorker — extends ActorRolloutRefWorker with EAGLE drafter co-training.

Adds:
- self.drafter: TrainingWorker (FSDPDrafterEngine for EAGLE model training)

HS collection is owned by HSCollectorManager on the trainer driver
(see verl/experimental/hs_collector/), not by this worker.

The DrafterDataController also lives on the driver. This worker only:
- Receives per-rank data via mesh dispatch (update_drafter)
- Fetches tensors from Mooncake
- Runs drafter training

See claude_docs/rfc-drafter-trainer-integration.md for the full design.
"""

import logging
from typing import Optional

import torch
from omegaconf import DictConfig

from verl.protocol import DataProto
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

        from recipe.drafter_cotraining.drafter_engine import (
            DrafterModelConfig,
            build_drafter_subconfig,
        )

        drafter_cfg = self.config.drafter
        drafter_training_config = TrainingWorkerConfig(
            model_type="drafter_model",
            model_config=build_drafter_subconfig(
                drafter_cfg.get("model_config", {}), DrafterModelConfig
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
        """Copy frozen weights from actor into drafter.

        Syncs: embed_tokens, target_lm_head_weight, verifier_norm (final RMSNorm).
        All frozen (requires_grad=False). embed_tokens for drafter input,
        target_lm_head_weight for target distribution, verifier_norm for
        pre-norm correction. The drafter's own lm_head is trainable and NOT synced.

        Called at init and after each update_actor(). In verl the actor trains
        every RL step (unlike TorchSpec where the target is fixed), so the
        drafter's frozen copies must stay synchronized.
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
        """Receive per-rank shard of Mooncake keys, fetch tensors, collate, train.

        data is already split per DP rank by the drafter mesh dispatch fn.
        Each entry contains Mooncake keys — actual tensors fetched here.

        When the drafter engine is initialized (``drafter.model_config.target_model_path``
        is set) this runs the full Eagle3 training step (prepare_model_inputs →
        7-step TTT forward → 0.8^i-weighted backward → optimizer step) and
        returns all-reduced metrics in ``meta_info['train_metrics']``. When the
        engine is absent it falls back to the shape-print smoke path so the
        rollout+HS+dispatch pipeline can be exercised without a real draft model.
        """
        if len(data) == 0:
            return DataProto(non_tensor_batch={})

        mooncake_keys = data.non_tensor_batch.get("mooncake_keys", [])
        shapes_list = data.non_tensor_batch.get("shapes", [])
        dtypes_list = data.non_tensor_batch.get("dtypes", [])
        prompt_lens = data.non_tensor_batch.get("prompt_lens", [])
        response_lens = data.non_tensor_batch.get("response_lens", [])
        if len(mooncake_keys) == 0:
            return DataProto(non_tensor_batch={})

        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_initialized() else 0

        store = self._get_mooncake_store(data.meta_info.get("mooncake_cfg", {}), rank)
        if store is None:
            return DataProto(non_tensor_batch=data.non_tensor_batch)

        device = torch.device("cuda", torch.cuda.current_device())

        from recipe.drafter_cotraining.eagle3_collator import Eagle3Collator
        collator = Eagle3Collator()

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
            # Response-only loss mask — zeros on prompt positions, ones on
            # response positions EXCEPT the last (next-token prediction has no
            # valid target at the final response position; matches TorchSpec
            # sgl_engine_decode.py:249 `completion_tokens - 1` and the
            # `loss_mask[0, -1] = 0` in preprocessing.py:329-331). When either
            # length is missing we fall back to all-ones so the engine still
            # runs (degraded signal).
            seq_len = int(ids.shape[-1])
            plen = int(prompt_lens[i]) if i < len(prompt_lens) else 0
            rlen = int(response_lens[i]) if i < len(response_lens) else 0
            if rlen > 1:
                loss_mask = torch.zeros_like(ids).long()
                # rlen-1 ones: positions [plen, plen+rlen-1) — drops the last.
                end = min(plen + rlen - 1, seq_len)
                loss_mask[..., plen:end] = 1
            else:
                loss_mask = torch.ones_like(ids).long()
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

        batch = collator(features)

        if self.drafter is None:
            # Smoke-only fallback: no drafter engine configured, so just log
            # padded shapes per rank and exit. Keeps test_drafter_rollout_hs.py
            # runnable without a real Eagle3 checkpoint.
            self._log_drafter_batch_shapes(batch, rank)
            return DataProto(non_tensor_batch=data.non_tensor_batch)

        metrics = self._drafter_train_step(batch, rank)

        # Also log shapes the first time through so the smoke output still
        # contains the padded-shape sanity line.
        if not getattr(self, "_drafter_shapes_logged", False):
            self._log_drafter_batch_shapes(batch, rank)
            self._drafter_shapes_logged = True

        return DataProto(
            non_tensor_batch=data.non_tensor_batch,
            meta_info={"train_metrics": metrics},
        )

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

    def _drafter_train_step(self, batch, rank: int) -> dict:
        """Real Eagle3 forward + 0.8^i-weighted backward + optimizer step.

        Uses the drafter engine's ``train_mode`` context so parameter/optimizer
        offload is handled consistently with how actor training runs. The
        context also zeros grads on exit. Target model stays frozen — actor
        updates and drafter→rollout sync are out of scope for this milestone.
        """
        engine = self.drafter.engine
        device = torch.device("cuda", torch.cuda.current_device())

        # Move collator output onto GPU (some entries may already be there if
        # Eagle3Collator preserved device from per-sample tensors).
        batch_dev = {
            k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
        }

        with engine.train_mode():
            prepared = engine.prepare_model_inputs(batch_dev)
            plosses, _, acces = engine.module(**prepared)

            num_ttt = len(plosses)
            loss_weights = [0.8 ** i for i in range(num_ttt)]
            # accumulation_steps=1 for this milestone — one rollout batch per
            # optimizer step (see tasks/drafter-training-milestone.md §3.2).
            loss = sum(w * p for w, p in zip(loss_weights, plosses)) / 1.0
            loss.backward()

            grad_norm = engine.optimizer_step()
            lr = engine.lr_scheduler_step()

        return self._aggregate_drafter_metrics(
            plosses=plosses,
            acces=acces,
            loss_weights=loss_weights,
            grad_norm=grad_norm,
            lr=lr,
            rank=rank,
        )

    def _aggregate_drafter_metrics(
        self, plosses, acces, loss_weights, grad_norm, lr, rank: int
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
            "train/loss_weighted": float(weighted_loss),
            "train/loss_raw_mean": float(raw_mean_loss),
            "train/avg_acc": float(avg_acc),
            "train/simulated_acc_len": float(simulated_acc_len),
            "train/grad_norm": float(grad_norm) if grad_norm is not None else 0.0,
            "train/lr": float(lr) if lr is not None else 0.0,
        }
        for i, p in enumerate(plosses_list):
            metrics[f"train/ploss_{i}"] = float(p)
        for i, a in enumerate(acces_list):
            metrics[f"train/acc_{i}"] = float(a)

        if rank == 0:
            print(
                f"[drafter] loss={metrics['train/loss_weighted']:.4f} "
                f"acc0={metrics['train/acc_0']:.4f} "
                f"acc_len={metrics['train/simulated_acc_len']:.2f} "
                f"grad={metrics['train/grad_norm']:.3f} "
                f"lr={metrics['train/lr']:.2e}"
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

    # ── Weight Sync ───────────────────────────────────────────

    def _iter_rollout_drafter_snapshot(self, randomize: bool, seed: int):
        snapshot = getattr(self, "_rollout_drafter_weight_sync_snapshot", None)
        if snapshot is None:
            raise RuntimeError("No rollout drafter snapshot cached on this rank.")

        generator = None
        if randomize:
            import torch.distributed as dist

            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed + rank)

        for name, tensor in snapshot:
            if randomize and tensor.is_floating_point():
                randomized = torch.empty_like(tensor)
                randomized.normal_(generator=generator)
                yield name, randomized
            else:
                yield name, tensor

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def update_rollout_drafter_weights_from_snapshot(self, mode: str = "restore", seed: int = 2026):
        """TEST_ONLY: push random/restored drafter weights through verl IPC.

        Self-caching: first call fetches the baseline drafter state_dict from
        vLLM workers via ``self.rollout.get_drafter_weights()`` and stashes it
        on this rank. Subsequent calls reuse the cache. Pushes through
        ``ServerAdapter.update_drafter_weights`` → server-side shared-name
        filter → ``drafter.model.load_weights``.
        """
        if self.rollout is None:
            return {"ok": False, "reason": "rollout is not initialized"}
        if mode not in {"random", "restore"}:
            raise ValueError(f"mode must be 'random' or 'restore', got {mode!r}")

        # Lazy one-shot cache of baseline drafter weights (CPU side).
        if getattr(self, "_rollout_drafter_weight_sync_snapshot", None) is None:
            reports = await self.rollout.get_drafter_weights()
            if isinstance(reports, list):
                rollout_rank = getattr(self.rollout, "rollout_rank", 0)
                report = reports[rollout_rank] if rollout_rank < len(reports) else None
            else:
                report = reports
            if not report or not report.get("ok", False):
                reason = report.get("reason") if report else "missing snapshot report"
                return {"ok": False, "reason": reason}
            self._rollout_drafter_weight_sync_snapshot = report["weights"]

        weights = self._iter_rollout_drafter_snapshot(randomize=(mode == "random"), seed=seed)
        await self.rollout.update_drafter_weights(weights)
        return {
            "ok": True,
            "mode": mode,
            "num_tensors": len(self._rollout_drafter_weight_sync_snapshot),
        }

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def update_weights(self, global_steps: int = None):
        """Sync actor + drafter weights to rollout.

        Extended from parent to also:
        1. Re-sync frozen modules (embed_tokens, lm_head) from actor → drafter
           (actor weights changed during update_actor())
        2. Sync drafter weights to rollout (for speculative decoding)

        Note: actor → HS collector sync is driven from the trainer via
        HSCollectorManager.update_weights, not here.
        """
        # Actor → rollout (inherited)
        await super().update_weights(global_steps=global_steps)

        # Actor → drafter frozen modules (embed_tokens, lm_head changed after training)
        if self.drafter is not None:
            self._sync_drafter_frozen_modules()

        if self.drafter is not None and self.rollout is not None:
            # Drafter → rollout (for speculative decoding at inference time)
            drafter_params, _ = self.drafter.engine.get_per_tensor_param()
            await self.rollout.update_drafter_weights(drafter_params)
