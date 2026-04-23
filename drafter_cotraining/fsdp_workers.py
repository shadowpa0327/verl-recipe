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
        if model_cfg.get("local_path"):
            self._init_drafter()
        else:
            logger.info(
                "drafter.model_config.local_path not set — skipping _init_drafter; "
                "update_drafter will run the fetch+collate smoke path only."
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
        from verl.workers.engine_workers import TrainingWorker, TrainingWorkerConfig

        drafter_cfg = self.config.drafter
        drafter_training_config = TrainingWorkerConfig(
            model_type="drafter_model",
            model_config=drafter_cfg.get("model_config", {}),
            engine_config=drafter_cfg.get("engine_config", {}),
            optimizer_config=drafter_cfg.get("optimizer_config", {}),
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
        """Receive per-rank shard of mooncake keys, fetch tensors, collate, log shapes.

        data is already split per DP rank by the drafter mesh dispatch fn.
        Each entry contains Mooncake keys — actual tensors fetched here.

        Dispatch: make_nd_compute_dataproto_dispatch_fn(mesh_name="drafter")
        splits DataProto.non_tensor_batch per rank via np.array_split.

        Smoke version: fetch → collate → print one summary line per rank.
        train_batch / remove_eagle3_tensors are intentionally not wired yet —
        they land in a follow-up once the padded shapes look right.
        """
        if len(data) == 0:
            return

        mooncake_keys = data.non_tensor_batch.get("mooncake_keys", [])
        shapes_list = data.non_tensor_batch.get("shapes", [])
        dtypes_list = data.non_tensor_batch.get("dtypes", [])
        if len(mooncake_keys) == 0:
            return

        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_initialized() else 0

        store = self._get_mooncake_store(data.meta_info.get("mooncake_cfg", {}), rank)
        if store is None:
            return DataProto(non_tensor_batch=data.non_tensor_batch)

        device = torch.device("cuda", torch.cuda.current_device())

        drafter_cfg = self.config.drafter if hasattr(self.config, "drafter") else {}
        batch_size = drafter_cfg.get("batch_size", len(mooncake_keys)) if hasattr(drafter_cfg, "get") else len(mooncake_keys)
        max_steps = drafter_cfg.get("max_steps", 1) if hasattr(drafter_cfg, "get") else 1

        from recipe.drafter_cotraining.eagle3_collator import Eagle3Collator
        collator = Eagle3Collator()

        for step in range(max_steps):
            start = step * batch_size
            end = min(start + batch_size, len(mooncake_keys))
            if start >= len(mooncake_keys):
                break

            features = []
            for i in range(start, end):
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
                feat = {
                    "input_ids": ids,
                    "hidden_states": hs,
                    "loss_mask": torch.ones_like(ids).long(),
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
            lhs_shape = (
                tuple(batch["last_hidden_states"].shape)
                if "last_hidden_states" in batch
                else None
            )
            print("="*100)
            print(
                f"[drafter rank {rank}] step {step + 1}/{max_steps}: "
                f"B={batch['input_ids'].shape[0]} T_pad={batch['input_ids'].shape[1]}  "
                f"input_ids={tuple(batch['input_ids'].shape)} "
                f"hidden_states={tuple(batch['hidden_states'].shape)} "
                f"last_hs={lhs_shape} "
                f"attn_mask={tuple(batch['attention_mask'].shape)} "
                f"loss_mask={tuple(batch['loss_mask'].shape)}"
            )
            print("="*100)

        # Mesh dispatch collector expects each rank to return a concatable
        # DataProto. Echo the input (drops meta_info that isn't per-sample).
        return DataProto(non_tensor_batch=data.non_tensor_batch)

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

    # ── Weight Sync ───────────────────────────────────────────

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
            # TODO: self.rollout.update_drafter_weights(drafter_params)
            # Requires rollout to support drafter weight updates
            pass
