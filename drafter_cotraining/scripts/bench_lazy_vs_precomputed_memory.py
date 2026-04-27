"""Single-rank peak-memory benchmark: PrecomputedTarget vs LazyTarget.

Targets the kernel-level decision: what's the GPU memory delta between
``compute_target_p_padded`` + ``compiled_forward_kl_loss`` and
``compute_lazy_target_padded`` + ``compiled_forward_kl_loss_from_hs`` at
production Qwen3-8B shapes.

This is intentionally NOT a full Eagle3Model trace — the decoder body's
memory cost is identical between the two paths, so we isolate just the
target build + loss kernel + backward through it. For end-to-end peak,
run ``run_qwen3_8b_eagle3_pretrain.sh`` and look at ``nvidia-smi``.

Run:
    /root/verl/ref/.venv/bin/python recipe/drafter_cotraining/scripts/bench_lazy_vs_precomputed_memory.py
    /root/verl/ref/.venv/bin/python recipe/drafter_cotraining/scripts/bench_lazy_vs_precomputed_memory.py --rho 0.1
"""

from __future__ import annotations

import argparse
import gc
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/verl")

from recipe.drafter_cotraining.eagle3.eagle3_model import (  # noqa: E402
    compute_lazy_target_padded,
    compute_target_p_padded,
)
from recipe.drafter_cotraining.eagle3.ops.loss import (  # noqa: E402
    compiled_forward_kl_loss,
    compiled_forward_kl_loss_from_hs,
)


def _mb(b: int) -> float:
    return b / (1024 * 1024)


def _gb(b: int) -> float:
    return b / (1024 * 1024 * 1024)


def _reset():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def _build_inputs(B, T, H, V, rho, device, *, dtype=torch.bfloat16):
    """Allocate the production-shape tensors that BOTH paths share.

    Returns: prenorm_hs (grad-tracked), target_hs, target_lm_head_w (V, H),
             draft_norm_w, draft_lm_head_w (V, H), loss_mask (with sparsity rho).
    """
    torch.manual_seed(0)
    prenorm_hs = torch.randn(B, T, H, device=device, dtype=dtype, requires_grad=True)
    target_hs = torch.randn(B, T, H, device=device, dtype=dtype)
    target_lm_head_w = torch.randn(V, H, device=device, dtype=dtype)
    draft_norm_w = torch.randn(H, device=device, dtype=dtype, requires_grad=True)
    draft_lm_head_w = torch.randn(V, H, device=device, dtype=dtype, requires_grad=True)

    # Build a loss_mask with the requested density.
    mask_flat = torch.zeros(B * T, device=device)
    n_valid = max(1, int(B * T * rho))
    perm = torch.randperm(B * T, device=device)[:n_valid]
    mask_flat[perm] = 1.0
    loss_mask = mask_flat.reshape(B, T)
    return prenorm_hs, target_hs, target_lm_head_w, draft_norm_w, draft_lm_head_w, loss_mask


def _ttt_loop_precomputed(prenorm_hs_flat, target, draft_norm_w, draft_lm_head_w,
                          loss_mask, length, norm_eps):
    """Mirror Eagle3Model._calculate_loss for length TTT steps, precomputed path."""
    plosses = []
    mask = loss_mask
    for idx in range(length):
        valid_idx = mask.flatten().nonzero().squeeze(-1)
        if valid_idx.numel() == 0:
            continue
        torch._dynamo.mark_dynamic(valid_idx, 0)

        target_p_step = target.target_p_padded[:, idx : idx + mask.shape[1], :]
        tp_flat = target_p_step.reshape(-1, target_p_step.shape[-1])

        loss, _ = compiled_forward_kl_loss(
            prenorm_hs_flat, tp_flat, valid_idx,
            draft_norm_w, draft_lm_head_w, norm_eps,
        )
        plosses.append(loss)

        # Match Eagle3Model.forward: shift mask left for the next TTT step.
        mask = torch.cat([mask[:, 1:], torch.zeros_like(mask[:, -1:])], dim=1)

    loss_weights = [0.8 ** i for i in range(len(plosses))]
    return sum(w * p for w, p in zip(loss_weights, plosses))


def _ttt_loop_lazy(prenorm_hs_flat, target, draft_norm_w, draft_lm_head_w,
                   loss_mask, length, norm_eps):
    plosses = []
    mask = loss_mask
    D = target.lm_head_weight.shape[-1]
    for idx in range(length):
        valid_idx = mask.flatten().nonzero().squeeze(-1)
        if valid_idx.numel() == 0:
            continue
        torch._dynamo.mark_dynamic(valid_idx, 0)

        ths_step = target.hidden_states_padded[:, idx : idx + mask.shape[1], :]
        ths_flat = ths_step.reshape(-1, D)

        loss, _ = compiled_forward_kl_loss_from_hs(
            prenorm_hs_flat, ths_flat, valid_idx,
            draft_norm_w, draft_lm_head_w, target.lm_head_weight, norm_eps,
        )
        plosses.append(loss)

        mask = torch.cat([mask[:, 1:], torch.zeros_like(mask[:, -1:])], dim=1)

    loss_weights = [0.8 ** i for i in range(len(plosses))]
    return sum(w * p for w, p in zip(loss_weights, plosses))


def _bench_path(name, build_target_fn, ttt_loop_fn, inputs, length, norm_eps, rank):
    """Measure peak memory for one full target-build + 7-step TTT + backward."""
    prenorm_hs, target_hs, target_lm_head_w, draft_norm_w, draft_lm_head_w, loss_mask = inputs

    # Zero the draft grads to start clean.
    for p in (prenorm_hs, draft_norm_w, draft_lm_head_w):
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()

    _reset()
    base_alloc = torch.cuda.memory_allocated()
    print(f"[rank {rank}] {name}: baseline allocated = {_mb(base_alloc):.1f} MiB")

    # 1) Target build.
    target = build_target_fn(target_hs, target_lm_head_w, loss_mask, length)
    torch.cuda.synchronize()
    after_build_alloc = torch.cuda.memory_allocated()
    build_phase_peak = torch.cuda.max_memory_allocated()

    # 2) Forward through 7 TTT steps.
    torch.cuda.reset_peak_memory_stats()
    prenorm_hs_flat = prenorm_hs.reshape(-1, prenorm_hs.shape[-1])
    weighted = ttt_loop_fn(
        prenorm_hs_flat, target, draft_norm_w, draft_lm_head_w,
        loss_mask, length, norm_eps,
    )
    torch.cuda.synchronize()
    after_fwd_alloc = torch.cuda.memory_allocated()
    fwd_phase_peak = torch.cuda.max_memory_allocated()

    # 3) Backward.
    torch.cuda.reset_peak_memory_stats()
    weighted.backward()
    torch.cuda.synchronize()
    bwd_phase_peak = torch.cuda.max_memory_allocated()

    print(
        f"[rank {rank}] {name}:\n"
        f"  build  → resident_after_build = {_mb(after_build_alloc - base_alloc):8.1f} MiB    "
        f"build_phase_peak       = {_mb(build_phase_peak - base_alloc):8.1f} MiB\n"
        f"  fwd    → resident_after_fwd   = {_mb(after_fwd_alloc - base_alloc):8.1f} MiB    "
        f"fwd_phase_peak         = {_mb(fwd_phase_peak - base_alloc):8.1f} MiB\n"
        f"  bwd    → bwd_phase_peak       = {_mb(bwd_phase_peak - base_alloc):8.1f} MiB"
    )

    # Cleanup before next path.
    del target, weighted
    _reset()
    return {
        "build_resident": after_build_alloc - base_alloc,
        "build_peak": build_phase_peak - base_alloc,
        "fwd_resident": after_fwd_alloc - base_alloc,
        "fwd_peak": fwd_phase_peak - base_alloc,
        "bwd_peak": bwd_phase_peak - base_alloc,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=1, help="batch size (per-rank micro-batch)")
    parser.add_argument("--T", type=int, default=4096, help="sequence length (T_pad)")
    parser.add_argument("--H", type=int, default=4096, help="hidden size")
    parser.add_argument("--V", type=int, default=151936, help="vocab size (Qwen3-8B)")
    parser.add_argument("--length", type=int, default=7, help="TTT length")
    parser.add_argument("--rho", type=float, default=1.0,
                        help="loss-mask density [0,1] (1.0 = SFT-like full-coverage)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available", file=sys.stderr)
        sys.exit(1)

    rank = 0
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)

    print(
        f"=== Lazy vs Precomputed peak-memory bench ===\n"
        f"B={args.B}  T={args.T}  H={args.H}  V={args.V}  length={args.length}  rho={args.rho}\n"
    )

    inputs = _build_inputs(args.B, args.T, args.H, args.V, args.rho, device)
    norm_eps = 1e-6

    print(f"[rank {rank}] Tensor sizes after _build_inputs: "
          f"allocated = {_mb(torch.cuda.memory_allocated()):.1f} MiB\n")

    # ── Path A: PrecomputedTarget (bf16 storage)
    pc = _bench_path(
        "PrecomputedTarget",
        build_target_fn=lambda hs, w, m, L: compute_target_p_padded(
            target_hidden_states=hs, target_lm_head_weight=w,
            loss_mask=m, length=L, t2d=None,
        ),
        ttt_loop_fn=_ttt_loop_precomputed,
        inputs=inputs,
        length=args.length, norm_eps=norm_eps, rank=rank,
    )
    print()

    # ── Path B: LazyTarget
    lz = _bench_path(
        "LazyTarget",
        build_target_fn=lambda hs, w, m, L: compute_lazy_target_padded(
            target_hidden_states=hs, target_lm_head_weight=w, length=L,
        ),
        ttt_loop_fn=_ttt_loop_lazy,
        inputs=inputs,
        length=args.length, norm_eps=norm_eps, rank=rank,
    )
    print()

    # ── Comparison summary
    print(f"=== Summary (rho={args.rho}, V={args.V}, T={args.T}, B={args.B}) ===")
    print(f"{'phase':<22}{'precomputed':>16}{'lazy':>16}{'delta (lazy-pc)':>20}")
    for key, label in [
        ("build_resident", "build resident"),
        ("build_peak",     "build peak"),
        ("fwd_resident",   "fwd-end resident"),
        ("fwd_peak",       "fwd peak"),
        ("bwd_peak",       "bwd peak"),
    ]:
        pc_b, lz_b = pc[key], lz[key]
        delta = lz_b - pc_b
        print(f"{label:<22}{_mb(pc_b):>14.1f} MiB{_mb(lz_b):>14.1f} MiB{_mb(delta):>+18.1f} MiB")

    # The headline number: process-wide max during the full step.
    pc_overall = max(pc["build_peak"], pc["fwd_peak"], pc["bwd_peak"])
    lz_overall = max(lz["build_peak"], lz["fwd_peak"], lz["bwd_peak"])
    print(f"\n[rank {rank}] Step-overall peak (max of all phases):")
    print(f"  precomputed = {_mb(pc_overall):.1f} MiB")
    print(f"  lazy        = {_mb(lz_overall):.1f} MiB")
    print(f"  delta       = {_mb(lz_overall - pc_overall):+.1f} MiB "
          f"({100 * (lz_overall - pc_overall) / max(pc_overall, 1):+.1f}%)")


if __name__ == "__main__":
    main()
