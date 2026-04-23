"""
FSDPDrafterEngine — FSDP engine for EAGLE drafter model training.

Registered as model_type="drafter_model" in the EngineRegistry.

This engine manages the Eagle3Model (7-step TTT loop wrapping a draft model)
with FSDP for distributed training.

Relationship to Eagle3Model:
    FSDPDrafterEngine is verl infrastructure (FSDP, optimizer, device mgmt).
    Eagle3Model is the model (7-step TTT forward, Forward KL loss).
    The engine holds Eagle3Model as self.module — same pattern as
    FSDPEngine holding LlamaForCausalLM for the actor.

    FSDPDrafterEngine (verl)
      └── self.module = Eagle3Model (from torchspec, pure PyTorch)
            └── self.draft_model = LlamaForCausalLMEagle3
                  ├── embed_tokens  (frozen, copied from actor)
                  ├── fc            (trainable)
                  ├── midlayer      (trainable, FSDP-sharded)
                  ├── norm          (trainable)
                  └── lm_head       (trainable, draft model's own)

Frozen modules (embed_tokens, verifier_norm, target_lm_head_weight):
    Copied from actor weights (not live references — safe under FSDP).
    Must be re-synced after each update_actor() since the actor trains
    every RL step. Call sync_frozen_modules_from_actor() at init and
    after each actor update.

    Note: lm_head is NOT frozen. Per TorchSpec, the draft model has its
    own trainable lm_head (for draft logits in the loss kernel), separate
    from target_lm_head_weight (frozen, for target distribution).

Usage:
    engine = EngineRegistry.new("drafter_model", "fsdp", "cuda", ...)
    engine.sync_frozen_modules_from_actor(actor_embed, actor_norm)
"""

import copy
import logging
from typing import Optional

import torch
from tensordict import TensorDict

from verl.workers.engine.base import EngineRegistry
from verl.workers.engine.fsdp.transformer_impl import FSDPEngine

logger = logging.getLogger(__name__)


@EngineRegistry.register(model_type="drafter_model", backend=["fsdp", "fsdp2"], device=["cuda", "npu"])
class FSDPDrafterEngine(FSDPEngine):
    """
    FSDP engine for EAGLE drafter model.

    Tiny model (~2% of target params): fc projection + 1 decoder layer.
    Frozen modules (embed_tokens) copied from actor, re-synced after
    each update_actor(). lm_head is trainable (draft model's own).
    """

    def __init__(self, model_config, engine_config, optimizer_config, checkpoint_config):
        super().__init__(model_config, engine_config, optimizer_config, checkpoint_config)
        # Frozen modules from actor — set by sync_frozen_modules_from_actor()
        self._verifier_norm = None           # RMSNorm for pre-norm last_hs from vLLM
        self._target_lm_head_weight = None   # target lm_head weight for LazyTarget

    def initialize(self):
        """Load the EAGLE draft model, wrap in Eagle3Model, and set up FSDP.

        Creates:
            self.module = Eagle3Model(draft_model, length=7)
        Then parent's initialize() applies FSDP to self.module.
        """
        from recipe.drafter_cotraining.eagle3.draft.auto import AutoEagle3DraftModel, AutoDraftModelConfig
        from recipe.drafter_cotraining.eagle3.eagle3_model import Eagle3Model

        # 1. Load raw draft model (fc + midlayer + embed_tokens + lm_head + norm)
        draft_config = AutoDraftModelConfig.from_file(self.model_config.local_path)
        draft_model = AutoEagle3DraftModel.from_config(
            draft_config,
            torch_dtype=getattr(torch, self.model_config.dtype, torch.bfloat16),
        )

        # Freeze embedding (will be synced from actor)
        if hasattr(draft_model, "freeze_embedding"):
            draft_model.freeze_embedding()

        # 2. Wrap in Eagle3Model — adds 7-step TTT loop
        ttt_length = getattr(self.model_config, "ttt_length", 7)
        self.module = Eagle3Model(draft_model, length=ttt_length)

        trainable = sum(p.numel() for p in self.module.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.module.parameters() if not p.requires_grad)
        logger.info(
            "Eagle3Model loaded: %s trainable, %s frozen (%.1fM total)",
            f"{trainable:,}", f"{frozen:,}", (trainable + frozen) / 1e6,
        )

        # 3. Parent applies FSDP to self.module (Eagle3Model containing draft_model)
        super().initialize()

    # ------------------------------------------------------------------
    # Frozen module sync
    # ------------------------------------------------------------------

    def sync_frozen_modules_from_actor(
        self,
        actor_embed_tokens: torch.nn.Module,
        actor_lm_head: torch.nn.Module,
        actor_norm: Optional[torch.nn.Module] = None,
    ):
        """Copy frozen weights from the actor model into the drafter.

        Must be called:
        1. At init (after actor model is loaded)
        2. After each update_actor() (actor weights change with RL training)

        Modules synced (all frozen, requires_grad=False):
        - embed_tokens → self.module.draft_model.embed_tokens
        - target_lm_head_weight → self._target_lm_head_weight (separate, for target distribution)
        - verifier_norm → self._verifier_norm (separate, for pre-norm correction)

        Note: draft_model.lm_head is NOT synced — it is the draft model's
        own trainable parameter (produces draft logits in the loss kernel).
        actor_lm_head is only used for target_lm_head_weight.
        """
        draft_model = self._get_draft_model()

        # embed_tokens (used by draft_model.embed_input_ids())
        if hasattr(draft_model, "embed_tokens"):
            draft_model.embed_tokens.weight.data.copy_(actor_embed_tokens.weight.data)
            draft_model.embed_tokens.weight.requires_grad = False

        # target_lm_head_weight (used for target distribution in LazyTarget)
        # This is the actor's lm_head, NOT the draft model's lm_head.
        if self._target_lm_head_weight is None:
            self._target_lm_head_weight = actor_lm_head.weight.data.clone()
            self._target_lm_head_weight.requires_grad = False
        else:
            self._target_lm_head_weight.copy_(actor_lm_head.weight.data)

        # verifier_norm (RMSNorm applied to pre-norm last_hidden_states from vLLM)
        if actor_norm is not None:
            if self._verifier_norm is None:
                self._verifier_norm = copy.deepcopy(actor_norm)
                self._verifier_norm.requires_grad_(False)
            else:
                for p_dst, p_src in zip(self._verifier_norm.parameters(), actor_norm.parameters()):
                    p_dst.data.copy_(p_src.data)

        logger.info("Synced frozen modules from actor (embed_tokens, target_lm_head%s)",
                     ", verifier_norm" if actor_norm is not None else "")

    def _get_draft_model(self):
        """Get the underlying draft model from Eagle3Model wrapper.

        self.module is Eagle3Model which holds self.draft_model.
        With FSDP, self.module may be wrapped, so check .module too.
        """
        model = self.module
        # Unwrap FSDP if needed
        if hasattr(model, "module"):
            model = model.module
        # Eagle3Model holds draft_model
        if hasattr(model, "draft_model"):
            return model.draft_model
        return model

    # ------------------------------------------------------------------
    # Model inputs
    # ------------------------------------------------------------------

    def prepare_model_inputs(self, micro_batch: TensorDict):
        """Prepare inputs for Eagle3Model.forward().

        Handles verifier_norm and target construction before calling
        the 7-step TTT loop.

        Expects micro_batch to contain:
            input_ids:          [B, T]       — token IDs
            hidden_states:      [B, T, 3*D]  — 3 aux layer HS from Mooncake
            last_hidden_states: [B, T, D]    — final layer HS from Mooncake (pre-norm)
            attention_mask:     [B, T]       — 1=real, 0=pad
            loss_mask:          [B, T]       — which tokens contribute to loss

        Returns dict matching Eagle3Model.forward() signature:
            input_ids, attention_mask, target (LazyTarget), loss_mask, hidden_states
        """
        from recipe.drafter_cotraining.eagle3.eagle3_model import compute_lazy_target_padded

        input_ids = micro_batch["input_ids"]
        hidden_states = micro_batch["hidden_states"]
        attention_mask = micro_batch["attention_mask"]
        loss_mask = micro_batch["loss_mask"]
        last_hidden_states = micro_batch["last_hidden_states"]

        # Apply verifier_norm to pre-norm last_hidden_states from vLLM
        if self._verifier_norm is not None:
            with torch.no_grad():
                last_hidden_states = self._verifier_norm(last_hidden_states)

        # Build target for Forward KL loss
        eagle3 = self.module.module if hasattr(self.module, "module") else self.module
        target = compute_lazy_target_padded(
            last_hidden_states,
            self._target_lm_head_weight,
            eagle3.length,  # TTT length (7)
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target": target,
            "loss_mask": loss_mask,
            "hidden_states": hidden_states,
        }
