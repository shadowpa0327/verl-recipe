"""Verify Eagle3 draft-model save/load round-trip equivalence.

Single-GPU smoke that mimics the FSDPDrafterEngine.save_checkpoint output
shape (config.json + model.safetensors under huggingface/), then reloads via
AutoEagle3DraftModel.from_pretrained and checks:

  1. state_dict keys + shapes + bitwise tensor equality
  2. forward-pass output equality on a fixed input

This is the prerequisite for wiring up actor_rollout_ref.drafter.model_path:
the load path used by training must be observably identical to
re-instantiating the freshly-built model in memory.

Usage:
  python -m recipe.drafter_cotraining.scripts.test_drafter_save_load_equivalence \
      --target-path /root/verl/Qwen3-8B \
      [--existing-ckpt /root/verl/checkpoints/.../drafter/huggingface]
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

import torch
from safetensors.torch import save_file

from recipe.drafter_cotraining.eagle3.draft.auto import (
    AutoDraftModelConfig,
    AutoEagle3DraftModel,
)


def _build_fresh(target_path: str, dtype: torch.dtype, attention_backend: str):
    draft_config = AutoDraftModelConfig.from_target(target_path, template_path=None)
    model = AutoEagle3DraftModel.from_config(
        draft_config, torch_dtype=dtype, attention_backend=attention_backend
    )
    return model, draft_config


def _save_engine_style(model, config, save_dir: str) -> None:
    """Mimic FSDPDrafterEngine.save_checkpoint's huggingface/ output layout."""
    os.makedirs(save_dir, exist_ok=True)
    config.save_pretrained(save_dir)
    state_dict = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(state_dict, os.path.join(save_dir, "model.safetensors"))


def _state_dict_equal(sd_a: dict, sd_b: dict) -> tuple[bool, list[str]]:
    issues: list[str] = []
    keys_a, keys_b = set(sd_a.keys()), set(sd_b.keys())
    if keys_a != keys_b:
        if keys_a - keys_b:
            issues.append(f"keys only in A: {sorted(keys_a - keys_b)}")
        if keys_b - keys_a:
            issues.append(f"keys only in B: {sorted(keys_b - keys_a)}")
    for k in sorted(keys_a & keys_b):
        a, b = sd_a[k], sd_b[k]
        if a.shape != b.shape:
            issues.append(f"{k}: shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
        elif a.dtype != b.dtype:
            issues.append(f"{k}: dtype mismatch {a.dtype} vs {b.dtype}")
        elif not torch.equal(a.cpu(), b.cpu()):
            max_abs = (a.cpu().float() - b.cpu().float()).abs().max().item()
            issues.append(f"{k}: tensors differ (max |Δ|={max_abs:.3e})")
    return len(issues) == 0, issues


@torch.no_grad()
def _forward_outputs(model, device: torch.device):
    """Drive the four exposed forward primitives on fixed inputs.

    Eagle3DraftModel doesn't expose a single ``forward`` — backbone() needs
    pre-built input/hidden embeddings. We exercise each entry point the
    drafter engine actually calls so any divergence after load surfaces.
    """
    model = model.to(device).eval()
    cfg = model.config
    target_hidden_size = getattr(cfg, "target_hidden_size", cfg.hidden_size)
    B, T = 2, 8
    g = torch.Generator(device="cpu").manual_seed(1234)
    input_ids = torch.randint(0, cfg.vocab_size, (B, T), generator=g, device="cpu").to(device)
    aux_hs = torch.randn(B, T, target_hidden_size * 3, generator=g, dtype=torch.float32).to(
        device=device, dtype=next(model.parameters()).dtype
    )
    last_hs = torch.randn(B, T, cfg.hidden_size, generator=g, dtype=torch.float32).to(
        device=device, dtype=next(model.parameters()).dtype
    )

    out = {}
    out["embed"] = model.embed_input_ids(input_ids).clone()
    out["proj"] = model.project_hidden_states(aux_hs).clone()
    out["logits"] = model.compute_logits(last_hs).clone()
    return out


def _outputs_equal(out_a: dict, out_b: dict) -> tuple[bool, list[str]]:
    issues: list[str] = []
    if set(out_a.keys()) != set(out_b.keys()):
        issues.append(f"keys differ: {set(out_a)} vs {set(out_b)}")
        return False, issues
    for k in out_a:
        a, b = out_a[k].cpu(), out_b[k].cpu()
        if a.shape != b.shape:
            issues.append(f"{k}: shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
            continue
        if not torch.equal(a, b):
            af, bf = a.float(), b.float()
            max_abs = (af - bf).abs().max().item()
            issues.append(f"{k}: tensors differ (max |Δ|={max_abs:.3e})")
    return len(issues) == 0, issues


def round_trip_test(target_path: str, attention_backend: str) -> int:
    print(f"\n=== Round-trip test: build → save → reload (target={target_path}) ===")
    dtype = torch.bfloat16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[1/4] Building fresh draft model from target config...")
    model_a, config = _build_fresh(target_path, dtype, attention_backend)

    sd_before = {k: v.detach().cpu().clone() for k, v in model_a.state_dict().items()}
    out_before = _forward_outputs(model_a, device)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = os.path.join(tmpdir, "huggingface")
        print(f"[2/4] Saving (engine-style) to {save_dir}...")
        # Move to CPU before saving so save_file gets contiguous CPU tensors
        # without forcing a copy of every parameter through CUDA again.
        model_a_cpu = model_a.cpu()
        _save_engine_style(model_a_cpu, config, save_dir)

        del model_a, model_a_cpu
        torch.cuda.empty_cache() if device.type == "cuda" else None

        print("[3/4] Reloading via AutoEagle3DraftModel.from_pretrained...")
        model_b = AutoEagle3DraftModel.from_pretrained(
            save_dir,
            torch_dtype=dtype,
            attention_backend=attention_backend,
        )

    sd_after = {k: v.detach().cpu().clone() for k, v in model_b.state_dict().items()}
    out_after = _forward_outputs(model_b, device)

    print("[4/4] Comparing state_dicts and forward outputs...")
    sd_ok, sd_issues = _state_dict_equal(sd_before, sd_after)
    if sd_ok:
        print(f"  state_dict: OK ({len(sd_before)} tensors match exactly)")
    else:
        print(f"  state_dict: FAIL ({len(sd_issues)} issues)")
        for x in sd_issues:
            print(f"    - {x}")

    out_ok, out_issues = _outputs_equal(out_before, out_after)
    if out_ok:
        print(f"  forward:    OK ({len(out_before)} primitives match exactly)")
    else:
        print(f"  forward:    FAIL ({len(out_issues)} issues)")
        for x in out_issues:
            print(f"    - {x}")

    return 0 if (sd_ok and out_ok) else 1


def existing_checkpoint_test(ckpt_dir: str, attention_backend: str) -> int:
    print(f"\n=== Existing-checkpoint test: load {ckpt_dir} ===")
    if not os.path.isdir(ckpt_dir):
        print(f"  SKIP: {ckpt_dir} is not a directory")
        return 0
    if not os.path.isfile(os.path.join(ckpt_dir, "config.json")):
        print(f"  SKIP: no config.json under {ckpt_dir}")
        return 0
    if not os.path.isfile(os.path.join(ckpt_dir, "model.safetensors")):
        print(f"  SKIP: no model.safetensors under {ckpt_dir}")
        return 0

    dtype = torch.bfloat16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[1/2] Loading via AutoEagle3DraftModel.from_pretrained...")
    model = AutoEagle3DraftModel.from_pretrained(
        ckpt_dir,
        torch_dtype=dtype,
        attention_backend=attention_backend,
    )

    print("[2/2] Comparing state_dict against on-disk safetensors...")
    from safetensors import safe_open

    on_disk = {}
    with safe_open(os.path.join(ckpt_dir, "model.safetensors"), framework="pt") as f:
        for k in f.keys():
            on_disk[k] = f.get_tensor(k)

    sd_in_mem = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    sd_ok, sd_issues = _state_dict_equal(on_disk, sd_in_mem)
    if sd_ok:
        print(f"  state_dict: OK ({len(on_disk)} tensors match exactly)")
    else:
        print(f"  state_dict: FAIL ({len(sd_issues)} issues)")
        for x in sd_issues:
            print(f"    - {x}")

    # Cheap forward sanity to confirm the loaded weights actually drive a graph.
    _ = _forward_outputs(model, device)
    print("  forward:    OK (executed without error)")

    return 0 if sd_ok else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target-path", default="/root/verl/Qwen3-8B",
        help="Target model path (for build-from-config in round-trip test).",
    )
    parser.add_argument(
        "--existing-ckpt", default=
        "/root/verl/checkpoints/draft_model_pretrain/pretrain/global_step_100/drafter/huggingface",
        help="Existing drafter HF export directory (for load-only test).",
    )
    parser.add_argument(
        "--attention-backend", default="flex_attention",
        choices=["flex_attention", "fa4"],
    )
    parser.add_argument("--skip-round-trip", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    rc = 0
    if not args.skip_round_trip:
        rc |= round_trip_test(args.target_path, args.attention_backend)
    if not args.skip_existing:
        rc |= existing_checkpoint_test(args.existing_ckpt, args.attention_backend)

    print("\n=== RESULT:", "PASS" if rc == 0 else "FAIL", "===")
    sys.exit(rc)


if __name__ == "__main__":
    main()
