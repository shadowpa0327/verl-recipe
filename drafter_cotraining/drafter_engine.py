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
    engine = EngineRegistry.new("drafter_model", "fsdp2", "cuda", ...)
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
    attention_backend: str = "flex_attention"

    # Use the lazy forward-KL kernel (compute target probs inside the compiled
    # graph) instead of the default precomputed-bf16 path. Only meaningful when
    # vocab pruning is OFF — saves the resident ``(B, T+L, V_full) × 2 B``
    # tensor at the cost of a per-TTT-step ``(N_valid, V_full)`` softmax.
    # The lazy path defends against autograd-graph leakage with a triple
    # ``.detach()`` (factory + kernel) so it stays safe under FSDP2
    # micro-batch accumulation. See ``LazyTarget vs Precomputed Memory
    # Analysis.md`` for the memory crossover.
    enable_lazy_target: bool = False

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


@EngineRegistry.register(model_type="drafter_model", backend=["fsdp2"], device=["cuda", "npu"])
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
        self._target_lm_head_weight = None   # target lm_head weight for PrecomputedTarget builder

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
        attention_backend = getattr(self.model_config, "attention_backend", "flex_attention")
        draft_model = AutoEagle3DraftModel.from_config(
            draft_config,
            torch_dtype=getattr(torch, self.model_config.dtype, torch.bfloat16),
            attention_backend=attention_backend,
        )

        # Every rank loads embed_tokens from the target checkpoint. We do this
        # pre-wrap on every rank for simplicity and as a sanity layer; the
        # post-wrap fsdp2_load_full_state_dict broadcast in _build_fsdp_module
        # would also reconcile the weights from rank 0, but loading on every
        # rank costs almost nothing for an Embedding and avoids any window
        # where ranks disagree.
        target_path = getattr(self.model_config, "target_model_path", None)
        if target_path:
            draft_model.load_embedding(target_path, embedding_key="model.embed_tokens.weight")
            rank = dist.get_rank() if dist.is_initialized() else 0
            logger.info("[rank %d] Loaded draft.embed_tokens from %s", rank, target_path)

        if hasattr(draft_model, "freeze_embedding"):
            draft_model.freeze_embedding()

        ttt_length = int(getattr(self.model_config, "ttt_length", 7))
        module = Eagle3Model(
            draft_model,
            length=ttt_length,
            attention_backend=attention_backend,
        )

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

    def _build_fsdp_module(self, module):
        """Selective FSDP2 wrap for the drafter.

        Two `fully_shard()` calls, in order:

          1) `fully_shard(LlamaDecoderLayer)` — gives the midlayer its own FSDP
             sub-unit. Its 7 Linears (q/k/v/o_proj, gate/up/down_proj) and 2
             RMSNorms become DTensors managed by this sub-unit; gathered on
             pre-forward, resharded on post-forward (the FSDP2 default).

          2) `fully_shard(module)` — the root wrap. Per `fully_shard` semantics,
             every param NOT already in a sub-unit becomes managed by THIS
             unit. So `lm_head`, `norm`, `fc`, and `embed_tokens` all land in
             the root unit. They are still DTensors (FSDP2 always shards) —
             just OWNED by the root unit instead of their own sub-units.

        The whole point is `lm_head` ownership. Our compiled loss kernel reads
        `lm_head.weight` as an extracted tensor argument:

            logits = F.linear(norm_hs, lm_head_weight)   # NOT lm_head(input)

        bypassing `lm_head.forward()`. If `lm_head` had its own FSDP sub-unit
        (which is what verl's default `_select_fsdp2_wrap_targets` does when
        `tie_word_embeddings=False`, e.g. Qwen3-8B), THAT sub-unit's pre-forward
        hook would only fire on `lm_head(input)` — never on the kernel's
        extracted-tensor read. The kernel would then see a non-gathered
        (still-sharded) DTensor and either error with `Tensor × DTensor` or
        compute on a wrong shape.

        With `lm_head` in the root unit instead:
          - The root's pre-forward hook fires at the start of every
            `Eagle3Model.forward()` → gathers `lm_head.weight` to a Replicate
            DTensor (full shape on every rank).
          - PyTorch's auto-root-detection forces the root's effective
            `reshard_after_forward=False` (regardless of what we pass) → the
            gathered state survives through backward → optimizer step. No
            reshard between forward and backward.
          - PyTorch's DTensor dispatch handles `F.linear(plain_input,
            replicate_dtensor)` correctly by promoting the plain input to a
            Replicate DTensor → DTensor × DTensor matmul.

        See `verl/utils/fsdp_utils.py:734-766` (`set_reshard_after_forward`
        docstring) for the auto-root-detection guarantee, and
        `claude_docs/research/Eagle3-Co-Trained/FSDP2 Wrap — TorchSpec
        Cross-Reference.md` for the full design rationale and a comparison
        against TorchSpec's per-Linear granularity.

        FSDP2-only — FSDP1 is intentionally not supported on this engine. The
        @EngineRegistry.register backend list is `["fsdp2"]`, so an FSDP1
        config will fail at engine-resolution time. We force the strategy
        because (a) the FSDP2 selective wrap above is what makes the compiled
        loss kernel work, (b) FSDP1's `use_orig_params=True` workaround was a
        compatibility shim from before this engine existed, and (c) supporting
        both backends doubled the surface area of an already-load-bearing
        engine for no production benefit.
        """
        assert self.engine_config.strategy == "fsdp2", (
            f"FSDPDrafterEngine is FSDP2-only; got strategy={self.engine_config.strategy!r}. "
            "Set drafter.engine_config.strategy=fsdp2 in your config."
        )

        from torch.distributed.fsdp import (
            CPUOffloadPolicy,
            MixedPrecisionPolicy,
            fully_shard,
        )
        from verl.utils.fsdp_utils import fsdp2_load_full_state_dict

        param_dtype = getattr(torch, self.model_config.dtype, torch.bfloat16)
        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=torch.float32,
            cast_forward_inputs=True,
        )
        offload_policy = (
            CPUOffloadPolicy(pin_memory=True)
            if self.engine_config.offload_policy
            else None
        )
        fsdp_kwargs = {
            "mesh": self.device_mesh,
            "mp_policy": mp_policy,
            "offload_policy": offload_policy,
        }

        # Capture full state PRE-WRAP — fsdp2_load_full_state_dict broadcasts
        # from rank 0 onto every rank's sharded DTensors after wrap.
        full_state = module.state_dict()

        # Shard only LlamaDecoderLayer sub-units (TorchSpec-style).
        # Sub-units default to reshard_after_forward=True.
        sharded_count = 0
        for _name, sub in module.named_modules():
            if sub.__class__.__name__ == "LlamaDecoderLayer":
                fully_shard(sub, **fsdp_kwargs)
                sharded_count += 1
        logger.info(
            "FSDPDrafterEngine: sharded %d LlamaDecoderLayer sub-units (FSDP2 selective wrap)",
            sharded_count,
        )

        # Wrap root. Auto-detected as root → effective reshard_after_forward=False
        # → root params (lm_head, norm, fc, embed_tokens) stay gathered through bwd,
        # which is what the compiled loss kernel needs.
        fully_shard(module, **fsdp_kwargs)

        # Broadcast rank-0 full state onto all ranks' sharded DTensors.
        fsdp2_load_full_state_dict(module, full_state, self.device_mesh, offload_policy)
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

        Builds a PrecomputedTarget (bf16-stored target_p) outside the compile
        graph so the compiled loss kernel sees only fixed-shape inputs across
        micro-batches. Handles both pruning (t2d set) and no-pruning (t2d=None)
        configs uniformly.

        Expects micro_batch to contain:
            input_ids:          [B, T]       — token IDs
            hidden_states:      [B, T, 3*D]  — 3 aux layer HS from Mooncake
            last_hidden_states: [B, T, D]    — final layer HS from Mooncake (pre-norm)
            attention_mask:     [B, T]       — 1=real, 0=pad
            loss_mask:          [B, T]       — which tokens contribute to loss

        Returns dict matching Eagle3Model.forward() signature:
            input_ids, attention_mask, target (PrecomputedTarget), loss_mask, hidden_states
        """
        from recipe.drafter_cotraining.eagle3.eagle3_model import (
            compute_lazy_target_padded,
            compute_target_p_padded,
            padding,
        )

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

        eagle3 = self.module.module if hasattr(self.module, "module") else self.module

        # No vocab pruning today — t2d=None falls into the no-pruning branch.
        # When pruning is added later, surface t2d via _t2d_index.
        t2d = getattr(self, "_t2d_index", None)

        # Lazy target builder is only meaningful in the no-pruning regime; the
        # whole point of pruning is the V_draft projection, so keep the
        # precomputed path whenever t2d is set even if the flag is on.
        if getattr(self.model_config, "enable_lazy_target", False) and t2d is None:
            target = compute_lazy_target_padded(
                target_hidden_states=last_hidden_states,
                target_lm_head_weight=self._target_lm_head_weight,
                length=eagle3.length,  # TTT length (7)
            )
        else:
            target = compute_target_p_padded(
                target_hidden_states=last_hidden_states,
                target_lm_head_weight=self._target_lm_head_weight,
                loss_mask=loss_mask,
                length=eagle3.length,  # TTT length (7)
                t2d=t2d,
            )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target": target,
            "loss_mask": loss_mask,
            "hidden_states": hidden_states,
        }
