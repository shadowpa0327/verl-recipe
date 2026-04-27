"""Standalone test: FSDP2 set_requires_gradient_sync preserves grad accumulation.

The question this test answers in isolation: under FSDP2's selective wrap, does
deferring grad-sync via `set_requires_gradient_sync(False)` on all-but-last
micro-batch produce the same accumulated `.grad` as always-sync?

Test design (avoids torch.compile recompilation issues by keeping ALL forwards
at the same shape):
  R1: 4 forwards each on B=1, scaling = mb_valid/total_valid, sync ON throughout.
  R2: 4 forwards each on B=1, same scaling, sync OFF for first 3, ON for last.

R1 and R2 must produce identical accumulated grads (within bf16 float noise) on
representative params (one root unit, one sub-unit). If they differ, the
sync-suppression path is broken.

The math correctness of the `mb_valid / total_valid` divisor (vs single-shot
semantics) is best validated by running the actual pretrain smoke at
`micro_batch_size_per_gpu=1` vs `=B_macro` and comparing loss curves —
torch.compile's shape-stability requirements make that hard to do cleanly
in a standalone test with B_per_call varying.

Run (must be world_size>=2 to actually exercise reduce-scatter; world_size=1
still verifies the local-grad math):

    /root/verl/ref/.venv/bin/torchrun --standalone --nproc-per-node=2 \\
        recipe/drafter_cotraining/scripts/test_fsdp2_micro_batch_accum.py
"""

from __future__ import annotations

import os
import sys
import traceback

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from transformers.models.llama.configuration_llama import LlamaConfig

sys.path.insert(0, "/root/verl")

from recipe.drafter_cotraining.eagle3.draft.llama3_eagle import LlamaForCausalLMEagle3  # noqa: E402
from recipe.drafter_cotraining.eagle3.eagle3_model import (  # noqa: E402
    Eagle3Model,
    compute_target_p_padded,
)
from verl.utils.fsdp_utils import fsdp2_load_full_state_dict  # noqa: E402


H, V, LENGTH = 128, 256, 3
NUM_HEADS, NUM_KV = 4, 2
SEQ_LEN = 32
N_MICRO = 4  # number of micro-batches; each has B=1 to keep kernel shape stable


def _make_tiny_eagle3():
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
    if hasattr(draft, "freeze_embedding"):
        draft.freeze_embedding()
    return Eagle3Model(draft, length=LENGTH, attention_backend="flex_attention")


def _selective_fsdp2_wrap(module, mesh, mp_policy):
    fsdp_kwargs = {"mesh": mesh, "mp_policy": mp_policy, "offload_policy": None}
    full_state = module.state_dict()
    for _name, sub in module.named_modules():
        if sub.__class__.__name__ == "LlamaDecoderLayer":
            fully_shard(sub, **fsdp_kwargs)
    fully_shard(module, **fsdp_kwargs)
    fsdp2_load_full_state_dict(module, full_state, mesh, None)
    return module


def _make_micro_batch(device, *, seed: int, n_valid: int):
    """Build one micro-batch (B=1) with `n_valid` loss-mask positions."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    input_ids = torch.randint(0, V, (1, SEQ_LEN), generator=g).to(device)
    attention_mask = torch.ones(1, SEQ_LEN, dtype=torch.long, device=device)
    loss_mask = torch.zeros(1, SEQ_LEN, device=device)
    loss_mask[0, :n_valid] = 1.0
    hidden_states = torch.randn(
        1, SEQ_LEN, H * 3, generator=g, dtype=torch.bfloat16
    ).to(device)
    target_hs = torch.randn(
        1, SEQ_LEN, H, generator=g, dtype=torch.bfloat16
    ).to(device)
    target_lm_head = torch.randn(V, H, generator=g, dtype=torch.bfloat16).to(device)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "loss_mask": loss_mask,
        "hidden_states": hidden_states,
        "target_hs": target_hs,
        "target_lm_head": target_lm_head,
        "n_valid": n_valid,
    }


def _zero_grads(model):
    for p in model.parameters():
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()


def _snapshot_grads(model, names):
    """Return {name: full_tensor.float().clone()}. Gathers DTensor grads."""
    from torch.distributed.tensor import DTensor

    snap = {}
    for full_name, p in model.named_parameters():
        if full_name in names and p.grad is not None:
            g = p.grad.detach()
            if isinstance(g, DTensor):
                g = g.full_tensor()
            snap[full_name] = g.clone().float()
    return snap


def _run_micro_batched(model, micro_batches, total_valid, weights, sync_pattern):
    """Run N forwards with per-mb scaling. sync_pattern[i] = is_last for mb i."""
    fn = getattr(model, "set_requires_gradient_sync", None)
    for mb_idx, mb in enumerate(micro_batches):
        if fn is not None:
            fn(sync_pattern[mb_idx])
        target = compute_target_p_padded(
            target_hidden_states=mb["target_hs"],
            target_lm_head_weight=mb["target_lm_head"],
            loss_mask=mb["loss_mask"],
            length=LENGTH,
            t2d=None,
        )
        plosses, _, _ = model(
            input_ids=mb["input_ids"],
            attention_mask=mb["attention_mask"],
            target=target,
            loss_mask=mb["loss_mask"],
            hidden_states=mb["hidden_states"],
        )
        scale = mb["n_valid"] / total_valid
        weighted = sum(w * p * scale for w, p in zip(weights, plosses))
        weighted.backward()
    if fn is not None:
        fn(True)  # leave engine in synced state


def _diff(a, b):
    return (a - b).abs().max().item()


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
        print(
            f"=== FSDP2 grad-sync suppression preserves accumulation "
            f"(world_size={world_size}) ===",
            flush=True,
        )

    mesh = init_device_mesh("cuda", (world_size,))
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        cast_forward_inputs=True,
    )

    torch.manual_seed(42)
    model = _make_tiny_eagle3().to(device)
    model = _selective_fsdp2_wrap(model, mesh, mp_policy)

    weights = [0.8 ** i for i in range(LENGTH)]

    track_names = {
        "draft_model.lm_head.weight",                      # root unit
        "draft_model.midlayer.self_attn.q_proj.weight",    # sub-unit (DTensor)
        "draft_model.norm.weight",                         # root unit
    }
    log(f"Tracking grads on: {sorted(track_names)}")

    # Build N_MICRO micro-batches with varying n_valid (so the test exercises
    # the production divisor math, not just uniform-N case).
    micro_batches = [
        _make_micro_batch(device, seed=11 + i, n_valid=8 + i * 2)
        for i in range(N_MICRO)
    ]
    total_valid = sum(mb["n_valid"] for mb in micro_batches)
    log(
        f"Micro-batches: {N_MICRO} × B=1, T={SEQ_LEN}, "
        f"n_valid per mb = {[mb['n_valid'] for mb in micro_batches]}, total_valid={total_valid}"
    )

    failures = []

    # ── R1: always-sync baseline. All micro-batches with sync=True.
    log("[R1] N_MICRO backwards, sync=True throughout (always reduce-scatter) ...")
    _zero_grads(model)
    try:
        _run_micro_batched(
            model,
            micro_batches=micro_batches,
            total_valid=total_valid,
            weights=weights,
            sync_pattern=[True] * N_MICRO,
        )
        grads_r1 = _snapshot_grads(model, track_names)
        for k in sorted(grads_r1):
            log(f"  R1 grad[{k}] norm={grads_r1[k].norm().item():.6f}")
    except Exception as exc:
        failures.append(f"R1 raised: {exc}")
        if rank == 0:
            traceback.print_exc()

    # ── R2: production sync pattern (off for all-but-last).
    log("[R2] N_MICRO backwards, sync=False for first N-1, sync=True for last ...")
    _zero_grads(model)
    try:
        _run_micro_batched(
            model,
            micro_batches=micro_batches,
            total_valid=total_valid,
            weights=weights,
            sync_pattern=[False] * (N_MICRO - 1) + [True],
        )
        grads_r2 = _snapshot_grads(model, track_names)
        for k in sorted(grads_r2):
            log(f"  R2 grad[{k}] norm={grads_r2[k].norm().item():.6f}")

        diffs = {
            k: _diff(grads_r1[k], grads_r2[k])
            for k in track_names
            if k in grads_r1 and k in grads_r2
        }
        max_diff = max(diffs.values()) if diffs else 0.0
        log(f"[R2] Max-abs grad diff vs R1: {max_diff:.6e}")
        for k, d in sorted(diffs.items()):
            log(f"  {k}: max_abs_diff={d:.6e}")
        # Sync-on/off should be bitwise identical for local accumulation; comm
        # is the only thing that changes. World_size=1 has no comm so should
        # be exact zero. World_size>1 may show tiny float noise from differing
        # reduce-scatter timing.
        tol = 1e-3 if world_size > 1 else 1e-6
        for k, d in diffs.items():
            if d > tol:
                failures.append(f"R2 grad mismatch on {k}: {d:.4e} > tol {tol:.0e}")
    except Exception as exc:
        failures.append(f"R2 raised: {exc}")
        if rank == 0:
            traceback.print_exc()

    dist.barrier()
    dist.destroy_process_group()

    if rank == 0:
        if failures:
            print("\n=== FAILURES ===", flush=True)
            for f in failures:
                print(f"  - {f}", flush=True)
            sys.exit(1)
        else:
            print(
                "\n=== Sync suppression preserves accumulated grads ===",
                flush=True,
            )


if __name__ == "__main__":
    main()
