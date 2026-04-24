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
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from tensordict import TensorDict

from verl.base_config import BaseConfig
from verl.workers.engine.base import EngineRegistry
from verl.workers.engine.fsdp.transformer_impl import FSDPEngine

logger = logging.getLogger(__name__)


@dataclass
class DrafterModelConfig(BaseConfig):
    """Minimal model config for the drafter engine.

    The parent ``HFModelConfig`` is too heavy for us — it eagerly loads
    tokenizers, HF AutoConfig and hf_processor from a target-model path.
    The drafter is a standalone Eagle3 draft model loaded from a plain JSON
    file via ``AutoDraftModelConfig.from_file``, so we only need enough
    fields to satisfy ``FSDPEngine.__init__`` and ``_build_fsdp_module``.
    """

    _mutable_fields = {"local_path"}

    # Path to the draft-model JSON config (consumed by AutoDraftModelConfig.from_file).
    local_path: Optional[str] = None
    dtype: str = "bfloat16"
    ttt_length: int = 7

    # Path to the target (verifier) model — a HF repo id or local dir. We load
    # three frozen weights from here at drafter init:
    #   embed_tokens.weight      → draft_model.embed_tokens  (FSDP2-broadcast via fsdp2_load_full_state_dict)
    #   lm_head.weight           → self._target_lm_head_weight (dist.broadcast from rank 0)
    #   model.norm.weight        → self._verifier_norm.weight  (dist.broadcast from rank 0)
    # Matches TorchSpec's eagle3_trainer load pattern.
    target_model_path: Optional[str] = None

    # Fields accessed by FSDPEngine — stubbed so the parent __init__ works.
    use_remove_padding: bool = False
    lora_rank: int = 0
    lora_alpha: int = 16
    target_modules: Optional[Any] = None
    target_parameters: Optional[list[str]] = None
    exclude_modules: Optional[str] = None
    lora: dict[str, Any] = field(default_factory=dict)
    use_liger: bool = False
    use_fused_kernels: bool = False
    fused_kernel_options: dict = field(default_factory=dict)
    enable_gradient_checkpointing: bool = False
    enable_activation_offload: bool = False
    trust_remote_code: bool = False

    def get_processor(self):
        # The drafter has no tokenizer; FSDPCheckpointManager handles None.
        return None


def _load_tensors_from_model_path(model_path: str, keys: list[str]) -> dict:
    """Load a subset of weight tensors from an HF model directory or repo id.

    Handles both sharded checkpoints (``*.index.json`` + multiple safetensors
    files) and single-file checkpoints (``model.safetensors``). Falls back to
    ``snapshot_download`` if the path doesn't exist locally.
    """
    import glob
    import json
    import os

    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    if not os.path.exists(model_path):
        model_path = snapshot_download(repo_id=model_path)

    out: dict = {}

    index_files = glob.glob(os.path.join(model_path, "*.index.json"))
    if index_files:
        with open(index_files[0]) as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        files_to_keys: dict[str, list[str]] = {}
        for key in keys:
            files_to_keys.setdefault(weight_map[key], []).append(key)
        for fname, fkeys in files_to_keys.items():
            with safe_open(os.path.join(model_path, fname), framework="pt") as f:
                for key in fkeys:
                    out[key] = f.get_tensor(key)
        return out

    single_file = os.path.join(model_path, "model.safetensors")
    if os.path.exists(single_file):
        with safe_open(single_file, framework="pt") as f:
            for key in keys:
                out[key] = f.get_tensor(key)
        return out

    raise FileNotFoundError(
        f"No *.index.json or model.safetensors found under {model_path}"
    )


def build_drafter_subconfig(cfg, cls):
    """Build a drafter sub-config dataclass from a (possibly OmegaConf) dict.

    Bypasses ``omega_conf_to_dataclass`` (i.e. ``OmegaConf.structured(cls)``)
    which rejects some verl dataclasses whose type annotations don't match
    their ``None`` defaults — e.g. ``EngineConfig.max_token_len_per_gpu: int = None``
    raises ``ValidationError: Incompatible value 'None' for field of type 'int'``.
    """
    from dataclasses import fields

    from omegaconf import DictConfig, OmegaConf

    if isinstance(cfg, DictConfig):
        cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg = cfg or {}
    allowed = {f.name for f in fields(cls)}
    kwargs = {k: v for k, v in cfg.items() if k in allowed}
    return cls(**kwargs)


@EngineRegistry.register(model_type="drafter_model", backend=["fsdp", "fsdp2"], device=["cuda", "npu"])
class FSDPDrafterEngine(FSDPEngine):
    """
    FSDP engine for EAGLE drafter model.

    Tiny model (~2% of target params): fc projection + 1 decoder layer.
    Frozen modules (embed_tokens) copied from actor, re-synced after
    each update_actor(). lm_head is trainable (draft model's own).

    ``_build_module`` is overridden so the parent's standard
    ``_build_model_optimizer`` flow (build → FSDP-wrap → optimizer → LR
    scheduler → checkpoint manager) runs end-to-end; the drafter doesn't
    want the HF AutoModel path that ``FSDPEngine._build_module`` uses.
    """

    def __init__(self, model_config, engine_config, optimizer_config, checkpoint_config):
        super().__init__(model_config, engine_config, optimizer_config, checkpoint_config)
        # Frozen target-side weights — set by _load_target_frozen_weights()
        self._verifier_norm = None           # RMSNorm for pre-norm last_hs from vLLM
        self._target_lm_head_weight = None   # target lm_head weight for LazyTarget

    def _build_module(self):
        """Return the Eagle3-wrapped draft model (parent expects a plain nn.Module).

        On rank 0 we load the target model's ``embed_tokens.weight`` into the
        draft's ``embed_tokens`` before FSDP wraps. The parent's fsdp2 path
        captures ``full_state = module.state_dict()`` pre-wrap then calls
        ``fsdp2_load_full_state_dict(..., broadcast_from_rank0=True)`` so all
        ranks end up with the same weights post-wrap.
        """
        import torch.distributed as dist

        from recipe.drafter_cotraining.eagle3.draft.auto import AutoDraftModelConfig, AutoEagle3DraftModel
        from recipe.drafter_cotraining.eagle3.eagle3_model import Eagle3Model

        target_path = getattr(self.model_config, "target_model_path", None)
        local_path = self.model_config.local_path
        assert target_path, (
            "drafter.model_config.target_model_path must be set so the draft "
            "architecture can be auto-derived from the target model. "
            "(Provide a template at local_path only for vocab pruning or "
            "non-Llama architectures.)"
        )

        # Auto-derive draft architecture from the target's HF AutoConfig.
        # local_path is an optional template overlay — see auto.py.
        draft_config = AutoDraftModelConfig.from_target(target_path, template_path=local_path)
        draft_model = AutoEagle3DraftModel.from_config(
            draft_config,
            torch_dtype=getattr(torch, self.model_config.dtype, torch.bfloat16),
        )

        # Every rank loads embed_tokens from the target checkpoint. FSDP1 doesn't
        # do a rank-0-only load + broadcast like FSDP2's fsdp2_load_full_state_dict,
        # so the weights must be identical on each rank pre-wrap.
        target_path = getattr(self.model_config, "target_model_path", None)
        if target_path:
            draft_model.load_embedding(target_path, embedding_key="model.embed_tokens.weight")
            rank = dist.get_rank() if dist.is_initialized() else 0
            logger.info("[rank %d] Loaded draft.embed_tokens from %s", rank, target_path)

        if hasattr(draft_model, "freeze_embedding"):
            draft_model.freeze_embedding()

        ttt_length = int(getattr(self.model_config, "ttt_length", 7))
        module = Eagle3Model(draft_model, length=ttt_length)

        # apply_fsdp2 reads model._no_split_modules to pick the wrap targets.
        # Eagle3Model has exactly one LlamaDecoderLayer (as draft_model.midlayer);
        # sharding it across DP ranks handles the bulk of the params.
        module._no_split_modules = ["LlamaDecoderLayer"]
        # _select_fsdp2_wrap_targets reads model.config.tie_word_embeddings;
        # surface the draft config so FSDP2 setup doesn't blow up.
        module.config = draft_model.config

        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in module.parameters() if not p.requires_grad)
        logger.info(
            "Eagle3Model loaded: %s trainable, %s frozen (%.1fM total)",
            f"{trainable:,}", f"{frozen:,}", (trainable + frozen) / 1e6,
        )
        return module

    def initialize(self):
        """FSDPEngine.initialize() + target-model frozen-weight load.

        After the parent sets up ``self.module`` (FSDP-wrapped Eagle3), optimizer
        and LR scheduler, we load the target model's ``lm_head.weight`` and
        ``model.norm.weight`` on rank 0 and broadcast them to the other ranks.
        Both live on the engine (``_target_lm_head_weight``,``_verifier_norm``)
        — separate from the FSDP-wrapped draft model — so a plain
        ``dist.broadcast`` is enough.
        """
        super().initialize()
        target_path = getattr(self.model_config, "target_model_path", None)
        if target_path:
            self._load_target_frozen_weights(target_path)

    def _load_target_frozen_weights(self, target_model_path: str):
        """Load target's lm_head + model.norm into engine-owned frozen tensors.

        All ranks load from disk (cheap for a small weight count). Handles
        ``tie_word_embeddings=True`` models (e.g. Qwen3-4B) where
        ``lm_head.weight`` isn't stored separately — it aliases
        ``model.embed_tokens.weight``.
        """
        draft_model = self._get_draft_model()
        hidden_size = int(draft_model.config.hidden_size)
        rms_norm_eps = float(getattr(draft_model.config, "rms_norm_eps", 1e-6))
        dtype = getattr(torch, self.model_config.dtype, torch.bfloat16)
        device = torch.cuda.current_device()

        try:
            weights = _load_tensors_from_model_path(
                target_model_path, ["lm_head.weight", "model.norm.weight"]
            )
            lm_head_src = weights["lm_head.weight"]
        except KeyError:
            weights = _load_tensors_from_model_path(
                target_model_path,
                ["model.embed_tokens.weight", "model.norm.weight"],
            )
            lm_head_src = weights["model.embed_tokens.weight"]
            logger.info(
                "lm_head.weight absent (tie_word_embeddings=True); "
                "using model.embed_tokens.weight for target_lm_head_weight"
            )
        lm_head_w = lm_head_src.to(device=device, dtype=dtype)
        norm_w = weights["model.norm.weight"].to(device=device, dtype=dtype)

        lm_head_w.requires_grad = False
        self._target_lm_head_weight = lm_head_w

        from transformers.models.llama.modeling_llama import LlamaRMSNorm
        self._verifier_norm = LlamaRMSNorm(hidden_size, eps=rms_norm_eps).to(device=device, dtype=dtype)
        with torch.no_grad():
            self._verifier_norm.weight.copy_(norm_w)
        self._verifier_norm.requires_grad_(False)

        logger.info(
            "Loaded target-frozen weights from %s: lm_head=%s, verifier_norm=%s",
            target_model_path, tuple(lm_head_w.shape), tuple(norm_w.shape),
        )

    # ------------------------------------------------------------------
    # Frozen module sync (actor re-sync path — unused by the training smoke)
    # ------------------------------------------------------------------

    def sync_frozen_modules_from_actor(
        self,
        actor_embed_tokens: torch.nn.Module,
        actor_lm_head: torch.nn.Module,
        actor_norm: Optional[torch.nn.Module] = None,
    ):
        """No-op in the smoke (weights are loaded from ``target_model_path``).

        Kept on the class for forward-compat with the actor-updates path where
        the actor's weights drift every RL step and the drafter's frozen copies
        must be re-synced. Implementing that properly requires FSDP-aware
        gathering of the actor's (sharded) params — deferred follow-up.
        """
        if self._target_lm_head_weight is not None:
            return  # target-path init already ran
        logger.warning(
            "sync_frozen_modules_from_actor called but target_model_path was not "
            "set; drafter's frozen weights will remain at random init."
        )

    # ------------------------------------------------------------------
    # Checkpoint save — Eagle3-aware HF export alongside sharded state
    # ------------------------------------------------------------------

    def save_checkpoint(
        self,
        local_path: str,
        hdfs_path: Optional[str] = None,
        global_step: int = 0,
        max_ckpt_to_keep: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Sharded FSDP save + drafter-specific ``model.safetensors`` export.

        Verl's stock ``hf_model`` save path rebuilds via
        ``AutoModelForCausalLM.from_config`` — ``LlamaForCausalLMEagle3``
        isn't registered with HF's AutoModel, so that would silently
        instantiate a vanilla ``LlamaForCausalLM`` and drop every Eagle3
        tensor as unexpected. We write ``model.safetensors`` directly next
        to the ``config.json`` the parent already dumps from
        ``unwrap_model.config`` (which is the real HF ``LlamaConfig`` we
        set in ``_build_module``).

        Keep ``checkpoint_config.save_contents`` at the default
        ``["model", "optimizer", "extra"]`` for the drafter — do NOT
        include ``"hf_model"`` (that would trigger the broken path).
        """
        import os

        import torch.distributed as dist
        from safetensors.torch import save_file

        from verl.utils.fsdp_utils import get_fsdp_full_state_dict

        super().save_checkpoint(
            local_path=local_path,
            hdfs_path=hdfs_path,
            global_step=global_step,
            max_ckpt_to_keep=max_ckpt_to_keep,
            **kwargs,
        )

        full_state = get_fsdp_full_state_dict(
            self.module, offload_to_cpu=True, rank0_only=True
        )

        if dist.get_rank() == 0:
            prefix = "draft_model."
            extracted = {
                k[len(prefix):]: v
                for k, v in full_state.items()
                if k.startswith(prefix)
            }
            hf_path = os.path.join(local_path, "huggingface")
            os.makedirs(hf_path, exist_ok=True)
            save_file(extracted, os.path.join(hf_path, "model.safetensors"))
            logger.info(
                "Saved drafter HF export: %s/model.safetensors (%d tensors)",
                hf_path, len(extracted),
            )

        dist.barrier()

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
        from recipe.drafter_cotraining.eagle3.eagle3_model import compute_lazy_target_padded, padding

        input_ids = micro_batch["input_ids"]
        hidden_states = micro_batch["hidden_states"]
        attention_mask = micro_batch["attention_mask"]
        loss_mask = micro_batch["loss_mask"]
        last_hidden_states = micro_batch["last_hidden_states"]

        # Left-shift input_ids and last_hidden_states by 1 to match Eagle3's
        # inference semantics: at TTT step 0 position t the draft should take
        # aux[t] + token[t+1] (the token the verifier just emitted) and predict
        # token[t+2] = softmax(lm_head · verifier_hs[t+1]). Our Mooncake producer
        # captures at positions 0..T-1 (input_ids[t]=token[t], last_hs[t]=HS[t]),
        # so the training loop needs this shift — matches TorchSpec
        # `eagle3_trainer.py::_forward` (padding(..., left=False)).
        input_ids = padding(input_ids, left=False)
        last_hidden_states = padding(last_hidden_states, left=False)

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
