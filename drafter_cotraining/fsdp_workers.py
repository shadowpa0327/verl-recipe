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
        self._drafter_skeleton = drafter_cfg.get("skeleton", False)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # 1-4. Actor, ref, rollout, checkpoint — all inherited
        super().init_model()

        if not self._drafter_enabled:
            return

        # 5. Build drafter training engine (unless skeleton mode)
        if not self._drafter_skeleton:
            self._init_drafter()

        # 6. Register drafter mesh (pure DP — every rank is unique).
        #    Registered even in skeleton mode so update_drafter() receives
        #    a mesh-dispatched per-rank DataProto shard.
        import torch.distributed as dist
        self._register_dispatch_collect_info(
            mesh_name="drafter",
            dp_rank=dist.get_rank(),  # world_rank = dp_rank (pure DP)
            is_collect=True,
        )

        logger.info(
            "Drafter co-training initialized (%s)",
            "skeleton" if self._drafter_skeleton else "drafter engine",
        )

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
        """Receive per-rank shard of mooncake keys, fetch tensors, (optionally) train.

        data is already split per DP rank by the drafter mesh dispatch fn.
        Each entry contains Mooncake keys — actual tensors fetched at train time.

        Dispatch: make_nd_compute_dataproto_dispatch_fn(mesh_name="drafter")
        splits DataProto.non_tensor_batch per rank via np.array_split.

        Skeleton mode (``drafter.skeleton=True``): fetches tensors and prints a
        per-rank summary. No train_batch. Used to verify the end-to-end path
        rollout → HS collector → controller → mesh dispatch → Mooncake read
        on the drafter workers without wiring a real EAGLE model.
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

        if self._drafter_skeleton:
            self._skeleton_fetch_and_report(
                rank=rank,
                mooncake_keys=mooncake_keys,
                shapes_list=shapes_list,
                dtypes_list=dtypes_list,
                mooncake_cfg=data.meta_info.get("mooncake_cfg", {}),
            )
            # mesh dispatch collector expects each rank to return a concatable
            # DataProto. Echo the input (drops meta_info that isn't per-sample).
            return DataProto(non_tensor_batch=data.non_tensor_batch)

        if self.drafter is None:
            return

        drafter_cfg = self.config.drafter
        max_steps = drafter_cfg.get("max_steps", 1)
        batch_size = drafter_cfg.get("batch_size", len(data))

        # Fetch tensors from Mooncake and train
        from recipe.drafter_cotraining.mooncake import EagleMooncakeStore

        for step in range(max_steps):
            start = step * batch_size
            end = min(start + batch_size, len(mooncake_keys))
            if start >= len(mooncake_keys):
                break

            batch_keys = mooncake_keys[start:end]
            batch_shapes = shapes_list[start:end]
            batch_dtypes = dtypes_list[start:end]

            # TODO: Fetch from Mooncake and run train_batch
            # For each key in batch_keys:
            #   tensors = mooncake_store.get(key, shapes, dtypes, device)
            #   self.drafter.train_batch(data=tensors)
            #   mooncake_store.remove_eagle3_tensors(key)

            logger.debug("update_drafter: step %d, keys %d-%d", step, start, end)

    def _skeleton_fetch_and_report(
        self,
        rank: int,
        mooncake_keys,
        shapes_list,
        dtypes_list,
        mooncake_cfg: dict,
    ):
        """Skeleton path — fetch each key on this rank and log a summary."""
        from recipe.drafter_cotraining.mooncake import EagleMooncakeStore
        from recipe.drafter_cotraining.mooncake.config import MooncakeConfig

        if not mooncake_cfg:
            logger.warning("[drafter rank %d] no mooncake_cfg in meta_info; skipping fetch", rank)
            return

        mc_cfg = MooncakeConfig(**mooncake_cfg)
        device = torch.device("cuda", torch.cuda.current_device())
        store = EagleMooncakeStore(mc_cfg)
        store.setup(device=device)

        n = len(mooncake_keys)
        print(f"[drafter rank {rank}] received {n} key(s); fetching via EagleMooncakeStore")

        try:
            for i in range(n):
                key = str(mooncake_keys[i])
                shapes = shapes_list[i] if isinstance(shapes_list[i], dict) else {}
                raw_dtypes = dtypes_list[i] if isinstance(dtypes_list[i], dict) else {}
                dtypes = {
                    k: (getattr(torch, v) if isinstance(v, str) and hasattr(torch, v) else v)
                    for k, v in raw_dtypes.items()
                }
                out = store.get(key=key, shapes=shapes, dtypes=dtypes, device=device)
                hs_sum = float(out.hidden_states.to(torch.float32).sum().item())
                lhs_info = (
                    "None"
                    if out.last_hidden_states is None
                    else f"{tuple(out.last_hidden_states.shape)}"
                )
                print(
                    f"[drafter rank {rank}] sample[{i}] key={key} "
                    f"hs={tuple(out.hidden_states.shape)} "
                    f"ids={tuple(out.input_ids.shape)} lhs={lhs_info} "
                    f"hs_sum={hs_sum:.3e}"
                )
        finally:
            store.close()

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
