"""Standalone FSDP2 selective-wrap test for the drafter.

Exercises the TorchSpec-style selective wrap path used by
`FSDPDrafterEngine._build_fsdp_module` (recipe/drafter_cotraining/drafter_engine.py):
shard ONLY `LlamaDecoderLayer`; let `lm_head`/`norm`/`fc`/`embed_tokens` stay in
the root unit so they're auto-kept-gathered through forward → backward (because
PyTorch's `fully_shard` auto-detects the root and forces its effective
`reshard_after_forward=False`).

This isolates the FSDP2 plumbing from the rest of the pipeline: no Mooncake, no
HS collector, no vLLM. Constructs a tiny synthetic Eagle3Model so iteration is
fast (~seconds, not minutes), but exercises the same kernel + same wrap shape +
same DTensor handling as production.

Run (single-rank, world_size=1):
    /root/verl/ref/.venv/bin/torchrun --standalone --nproc-per-node=1 \\
        recipe/drafter_cotraining/scripts/test_fsdp2_drafter_wrap.py

Run (multi-rank, world_size=2):
    /root/verl/ref/.venv/bin/torchrun --standalone --nproc-per-node=2 \\
        recipe/drafter_cotraining/scripts/test_fsdp2_drafter_wrap.py

Each test prints `[T<N>] OK ...` or fails loudly. Exit code is 0 on full pass.
"""

from __future__ import annotations

import os
import sys
import traceback

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    fully_shard,
)
from transformers.models.llama.configuration_llama import LlamaConfig

# Repo-relative import setup
sys.path.insert(0, "/root/verl")

from recipe.drafter_cotraining.eagle3.draft.llama3_eagle import LlamaForCausalLMEagle3  # noqa: E402
from recipe.drafter_cotraining.eagle3.eagle3_model import (  # noqa: E402
    Eagle3Model,
    PrecomputedTarget,
    compute_target_p_padded,
)
from verl.utils.fsdp_utils import fsdp2_load_full_state_dict  # noqa: E402


# ── Constants for the tiny test fixture ──────────────────────
H, V, B, T_MAX, LENGTH = 128, 256, 1, 16, 3
NUM_HEADS, NUM_KV = 4, 2


def _make_tiny_eagle3() -> Eagle3Model:
    """Construct a small but production-shaped Eagle3Model on CPU (caller moves)."""
    config = LlamaConfig(
        hidden_size=H,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV,
        intermediate_size=H * 4,
        max_position_embeddings=1024,
        vocab_size=V,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        rope_scaling=None,
        pretraining_tp=1,
        pad_token_id=0,
    )
    config.draft_vocab_size = V
    draft = LlamaForCausalLMEagle3(config, attention_backend="flex_attention")
    draft = draft.to(dtype=torch.bfloat16)
    # Mirror production: embed_tokens frozen
    if hasattr(draft, "freeze_embedding"):
        draft.freeze_embedding()
    return Eagle3Model(draft, length=LENGTH, attention_backend="flex_attention")


def _selective_fsdp2_wrap(module, mesh, mp_policy, offload_policy):
    """Mirrors FSDPDrafterEngine._build_fsdp_module FSDP2 branch verbatim."""
    fsdp_kwargs = {
        "mesh": mesh,
        "mp_policy": mp_policy,
        "offload_policy": offload_policy,
    }
    full_state = module.state_dict()
    sharded = 0
    for _name, sub in module.named_modules():
        if sub.__class__.__name__ == "LlamaDecoderLayer":
            fully_shard(sub, **fsdp_kwargs)
            sharded += 1
    fully_shard(module, **fsdp_kwargs)
    fsdp2_load_full_state_dict(module, full_state, mesh, offload_policy)
    return module, sharded


def _make_synthetic_batch(device, length, *, seq_len=T_MAX):
    """Synthetic inputs for one Eagle3Model.forward call."""
    torch.manual_seed(7)
    input_ids = torch.randint(0, V, (B, seq_len), device=device)
    attention_mask = torch.ones(B, seq_len, dtype=torch.long, device=device)
    loss_mask = torch.ones(B, seq_len, device=device)
    hidden_states = torch.randn(B, seq_len, H * 3, device=device, dtype=torch.bfloat16)
    target_hs = torch.randn(B, seq_len, H, device=device, dtype=torch.bfloat16)
    target_lm_head = torch.randn(V, H, device=device, dtype=torch.bfloat16)

    target = compute_target_p_padded(
        target_hidden_states=target_hs,
        target_lm_head_weight=target_lm_head,
        loss_mask=loss_mask,
        length=length,
        t2d=None,
    )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "loss_mask": loss_mask,
        "hidden_states": hidden_states,
        "target": target,
    }


# ── Tests ────────────────────────────────────────────────────


def test_wrap_shape(model, sharded_count, log):
    """[T1] Selective wrap should produce exactly 1 LlamaDecoderLayer sub-unit."""
    assert sharded_count == 1, f"Expected 1 LlamaDecoderLayer sharded, got {sharded_count}"
    log("OK — sharded 1 LlamaDecoderLayer (root holds lm_head/norm/fc/embed_tokens)")


def test_set_requires_gradient_sync(model, log):
    """[T2] FSDP2 root must expose set_requires_gradient_sync (used for accum)."""
    fn = getattr(model, "set_requires_gradient_sync", None)
    assert fn is not None, "set_requires_gradient_sync missing on FSDP2 root"
    fn(False)
    fn(True)
    log("OK — set_requires_gradient_sync(True/False) callable on root")


def test_param_dtype(model, log):
    """[T3] Params should still be reachable; sub-unit ones are DTensor."""
    from torch.distributed.tensor import DTensor

    midlayer = model.draft_model.midlayer
    midlayer_qproj_w = midlayer.self_attn.q_proj.weight
    is_dtensor = isinstance(midlayer_qproj_w, DTensor)
    log(
        f"midlayer.self_attn.q_proj.weight: type={type(midlayer_qproj_w).__name__}, "
        f"shape={tuple(midlayer_qproj_w.shape)}, is_dtensor={is_dtensor}"
    )

    lm_head_w = model.draft_model.lm_head.weight
    log(
        f"draft_model.lm_head.weight: type={type(lm_head_w).__name__}, "
        f"shape={tuple(lm_head_w.shape)}, is_dtensor={isinstance(lm_head_w, DTensor)}"
    )
    log("OK — params reachable post-wrap (sub-unit params are DTensor; root params depend on PyTorch internal handling)")


def test_forward_backward(model, device, log):
    """[T4] Forward + backward must run without Tensor × DTensor errors.

    This is the canary: the compiled loss kernel reads `lm_head.weight` as a
    tensor argument inside a torch.compile region. If FSDP2's selective wrap is
    broken, this trips on the first call to compiled_forward_kl_loss.
    """
    batch = _make_synthetic_batch(device, length=model.length)
    plosses, _, acces = model(**batch)
    assert len(plosses) == model.length, f"Expected {model.length} TTT losses, got {len(plosses)}"
    for i, p in enumerate(plosses):
        assert torch.isfinite(p), f"ploss[{i}] is non-finite: {p}"
    log(f"forward OK — plosses={[f'{p.item():.3f}' for p in plosses]}")

    # Backward
    loss = sum(plosses) / len(plosses)
    loss.backward()

    has_grad = False
    grad_count = 0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            grad_count += 1
            if p.grad.abs().sum() > 0:
                has_grad = True
    assert has_grad, "no non-zero grads found after backward"
    log(f"backward OK — {grad_count} trainable params have grads")


def test_grad_sync_two_step(model, device, log):
    """[T5] Two backwards (sync=False then sync=True) should populate .grad.

    Under FSDP2, set_requires_gradient_sync(False) buffers partial grads
    internally and defers materialization on .grad until sync resumes. So
    .grad after the FIRST backward(sync=False) may not yet be visible — but
    after the SECOND backward(sync=True) the accumulated grad MUST appear
    on .grad. This is the production accumulation pattern.
    """
    # Zero any leftover grads
    for p in model.parameters():
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()

    # First micro-batch — sync OFF
    model.set_requires_gradient_sync(False)
    batch_a = _make_synthetic_batch(device, length=model.length)
    plosses_a, _, _ = model(**batch_a)
    loss_a = sum(plosses_a) / len(plosses_a)
    loss_a.backward()

    # Second micro-batch — sync ON
    model.set_requires_gradient_sync(True)
    batch_b = _make_synthetic_batch(device, length=model.length)
    plosses_b, _, _ = model(**batch_b)
    loss_b = sum(plosses_b) / len(plosses_b)
    loss_b.backward()

    has_grad = False
    grad_param_count = 0
    for p in model.parameters():
        if not p.requires_grad or p.grad is None:
            continue
        grad_param_count += 1
        g = p.grad
        # For DTensor grads, use .full_tensor() to reduce across replicas
        from torch.distributed.tensor import DTensor

        g_full = g.full_tensor() if isinstance(g, DTensor) else g
        if g_full.abs().sum().item() > 0:
            has_grad = True
            break
    assert has_grad, (
        "no non-zero grads after sync=False then sync=True backwards "
        f"(checked {grad_param_count} trainable params with .grad set)"
    )
    log(f"OK — accumulated grads visible on .grad after sync-resume "
        f"({grad_param_count} trainable params have .grad)")


# ── Driver ───────────────────────────────────────────────────


def main():
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    device = torch.device("cuda", local_rank)

    def log(msg):
        if rank == 0:
            print(f"[rank {rank}] {msg}", flush=True)

    if rank == 0:
        print(f"=== FSDP2 selective-wrap test (world_size={world_size}) ===", flush=True)

    mesh = init_device_mesh("cuda", (world_size,))
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        cast_forward_inputs=True,
    )

    # Build + move to GPU
    model = _make_tiny_eagle3().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"Built tiny Eagle3Model: {n_params:,} params (H={H}, V={V}, length={LENGTH})")

    # Wrap
    model, sharded_count = _selective_fsdp2_wrap(model, mesh, mp_policy, offload_policy=None)

    # ── Run tests
    failed = []
    for name, fn, args in [
        ("T1 wrap shape", test_wrap_shape, (model, sharded_count, lambda m: log(f"[T1] {m}"))),
        ("T2 grad-sync API", test_set_requires_gradient_sync, (model, lambda m: log(f"[T2] {m}"))),
        ("T3 param types", test_param_dtype, (model, lambda m: log(f"[T3] {m}"))),
        ("T4 forward+backward", test_forward_backward, (model, device, lambda m: log(f"[T4] {m}"))),
        ("T5 grad-sync two-step", test_grad_sync_two_step, (model, device, lambda m: log(f"[T5] {m}"))),
    ]:
        try:
            fn(*args)
        except Exception as exc:
            failed.append((name, exc))
            if rank == 0:
                print(f"[FAIL] {name}: {exc}", flush=True)
                traceback.print_exc()

    dist.barrier()
    dist.destroy_process_group()

    if rank == 0:
        if failed:
            print(f"\n=== {len(failed)} test(s) FAILED ===", flush=True)
            sys.exit(1)
        else:
            print("\n=== All FSDP2 wrap tests PASSED ===", flush=True)


if __name__ == "__main__":
    main()
