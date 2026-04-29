#!/usr/bin/env python3
"""
Standalone single-controller simulation for the drafter co-training pipeline.

Exercises the full data flow WITHOUT Ray or GPU:

  1. DrafterDataController receives dummy rollout sequences (Level 1)
  2. Controller pulls raw prompts → simulates HS collection via Mooncake put()
  3. Controller receives SampleMeta (Level 2)
  4. Controller drains as DataProto → simulates mesh dispatch (Level 3)
  5. "Worker" fetches tensors from Mooncake via get() and verifies

This validates the controller ↔ Mooncake ↔ worker data lifecycle end-to-end
on CPU with TCP transport. No GPU, no Ray, no vLLM required.

Usage:
    python scripts/test_single_controller_hs.py

    # Larger test:
    python scripts/test_single_controller_hs.py --num-sequences 16 --dp-size 4

    # Use existing mooncake_master:
    python scripts/test_single_controller_hs.py --skip-master
"""

import argparse
import atexit
import os
import signal
import shutil
import subprocess
import sys
import time

import numpy as np
import torch


# ─── Mooncake master lifecycle ─────────────────────────────

def find_mooncake_master_bin():
    if "MOONCAKE_BUILD_DIR" in os.environ:
        return os.path.join(os.environ["MOONCAKE_BUILD_DIR"], "mooncake-store/src/mooncake_master")
    found = shutil.which("mooncake_master")
    if found:
        return found
    return os.path.expanduser("~/build/mooncake-store/src/mooncake_master")


def launch_master(port=50051, http_port=8090, lease_ttl_ms=2000):
    binary = find_mooncake_master_bin()
    if not os.path.exists(binary):
        print(f"ERROR: mooncake_master not found at {binary}")
        print("  Install mooncake or set MOONCAKE_BUILD_DIR")
        sys.exit(1)

    cmd = [
        binary,
        f"--port={port}",
        f"--http_metadata_server_port={http_port}",
        "--http_metadata_server_host=0.0.0.0",
        "--enable_http_metadata_server=true",
        f"--default_kv_lease_ttl={lease_ttl_ms}",
    ]
    print(f"  Launching: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def _kill():
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                proc.kill()

    atexit.register(_kill)
    time.sleep(2)

    if proc.poll() is not None:
        stderr = proc.stderr.read().decode() if proc.stderr else ""
        print(f"ERROR: mooncake_master exited with code {proc.returncode}")
        if stderr:
            print(f"  stderr: {stderr[:500]}")
        sys.exit(1)

    print(f"  mooncake_master running (PID {proc.pid})")
    return proc


# ─── Helpers ───────────────────────────────────────────────

def make_mooncake_config(host, master_port, metadata_port, seq_len, hidden_dim, lease_ttl):
    from recipe.drafter_cotraining.mooncake.config import MooncakeConfig
    return MooncakeConfig(
        local_hostname=host,
        metadata_server=f"http://{host}:{metadata_port}/metadata",
        master_server_address=f"{host}:{master_port}",
        global_segment_size=256 * 1024 * 1024,
        local_buffer_size=128 * 1024 * 1024,
        protocol="tcp",
        max_seq_len=seq_len,
        hidden_dim=hidden_dim,
        async_put_pool_size=1,
        kv_lease_ttl_s=lease_ttl,
    )


def make_dummy_sequence(seq_len, vocab_size=152064):
    """Simulate a rollout output: input_ids + attention_mask."""
    input_ids = torch.randint(0, vocab_size, (seq_len,), dtype=torch.int64)
    attention_mask = torch.ones(seq_len, dtype=torch.int64)
    prompt_len = seq_len // 2
    return input_ids, attention_mask, prompt_len


def make_eagle3_tensors(seq_len, hidden_dim, num_aux_layers=3):
    """Simulate what the KV connector would produce during prefill."""
    hidden_states = torch.randn(seq_len, hidden_dim * num_aux_layers, dtype=torch.bfloat16)
    input_ids = torch.randint(0, 152064, (seq_len,), dtype=torch.int64)
    last_hidden_states = torch.randn(seq_len, hidden_dim, dtype=torch.bfloat16)
    return hidden_states, input_ids, last_hidden_states


def fmt_bytes(n):
    if n >= 1024**2:
        return f"{n / 1024**2:.1f} MB"
    if n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} B"


# ─── Main simulation ──────────────────────────────────────

def run_simulation(args):
    from recipe.drafter_cotraining.data.controller import DrafterDataController, SequenceMeta, SampleMeta
    from recipe.drafter_cotraining.mooncake.eagle_store import EagleMooncakeStore

    mc_config = make_mooncake_config(
        host=args.master_host,
        master_port=args.master_port,
        metadata_port=args.metadata_port,
        seq_len=args.seq_len,
        hidden_dim=args.hidden_dim,
        lease_ttl=args.lease_ttl,
    )

    # ── Phase 0: Setup controller + Mooncake stores ──
    print(f"\n[Phase 0] Setting up controller (dp_size={args.dp_size}) and Mooncake stores")

    controller = DrafterDataController(dp_size=args.dp_size)

    # "Inference side" store — simulates what VllmHSCollector's KV connector writes
    writer_store = EagleMooncakeStore(mc_config)
    writer_store.setup()
    print("  Writer store (HS collector side): connected")

    # "Training side" store — simulates what update_drafter() reads
    reader_store = EagleMooncakeStore(mc_config)
    reader_store.setup()
    print("  Reader store (drafter worker side): connected")

    status = controller.get_status()
    print(f"  Controller status: {status}")

    # ── Phase 1: Simulate rollout → push_raw_prompts (Level 1) ──
    print(f"\n{'=' * 70}")
    print(f"[Phase 1] Simulating rollout: generating {args.num_sequences} sequences")
    print(f"{'=' * 70}")

    sequences = []
    for i in range(args.num_sequences):
        ids, mask, prompt_len = make_dummy_sequence(args.seq_len)
        seq_meta = SequenceMeta(
            input_ids=ids.numpy(),
            attention_mask=mask.numpy(),
            prompt_len=prompt_len,
            response_len=args.seq_len - prompt_len,
        )
        sequences.append(seq_meta)

    controller.push_raw_prompts(sequences)
    status = controller.get_status()
    print(f"  Pushed {len(sequences)} sequences → Level 1")
    print(f"  Controller: raw_prompts={status['raw_prompts']}, sample_pool={status['sample_pool']}")

    # ── Phase 2: Pull raw prompts → HS collection → push_samples (Level 2) ──
    print(f"\n{'=' * 70}")
    print(f"[Phase 2] HS Collection: pull prompts → Mooncake put → push samples")
    print(f"{'=' * 70}")

    raw_prompts = controller.pull_raw_prompts()
    print(f"  Pulled {len(raw_prompts)} raw prompts from Level 1")

    sample_metas = []
    total_put_ms = 0.0

    for i, seq_meta in enumerate(raw_prompts):
        key = f"rl_step0_seq{i:04d}"
        seq_len = len(seq_meta.input_ids)

        # Simulate HS collection: vLLM prefill → KV connector → Mooncake
        hs, ids, lhs = make_eagle3_tensors(seq_len, args.hidden_dim)

        t0 = time.time()
        metadata = writer_store.put(
            key=key,
            hidden_states=hs,
            input_ids=ids,
            last_hidden_states=lhs,
        )
        writer_store.flush()
        put_ms = (time.time() - t0) * 1000
        total_put_ms += put_ms

        sample = SampleMeta(
            mooncake_key=key,
            shapes=metadata["shapes"],
            dtypes=metadata["dtypes"],
            seq_len=seq_len,
            n_tokens=seq_len,
        )
        sample_metas.append(sample)

        if i < 3 or i == len(raw_prompts) - 1:
            hs_bytes = hs.numel() * hs.element_size()
            lhs_bytes = lhs.numel() * lhs.element_size()
            ids_bytes = ids.numel() * ids.element_size()
            total = hs_bytes + lhs_bytes + ids_bytes
            print(f"  [{i:3d}] key={key}  put {fmt_bytes(total)} in {put_ms:.1f}ms")
            if i == 2 and len(raw_prompts) > 4:
                print(f"  ... ({len(raw_prompts) - 4} more) ...")

    controller.push_samples(sample_metas)
    status = controller.get_status()
    print(f"\n  Pushed {len(sample_metas)} samples → Level 2")
    print(f"  Controller: raw_prompts={status['raw_prompts']}, sample_pool={status['sample_pool']}")
    print(f"  Total put time: {total_put_ms:.1f}ms ({total_put_ms / len(raw_prompts):.1f}ms avg)")

    # ── Phase 3: Drain as DataProto → simulate mesh dispatch (Level 3) ──
    print(f"\n{'=' * 70}")
    print(f"[Phase 3] Drain → DataProto → mesh dispatch (dp_size={args.dp_size})")
    print(f"{'=' * 70}")

    proto = controller.drain_as_dataproto()
    if proto is None:
        print("  ERROR: drain_as_dataproto returned None!")
        return False

    status = controller.get_status()
    print(f"  Drained pool → DataProto with {len(proto.non_tensor_batch['mooncake_keys'])} entries")
    print(f"  Controller after drain: raw_prompts={status['raw_prompts']}, sample_pool={status['sample_pool']}")

    # Simulate mesh dispatch: split non_tensor_batch per DP rank
    n_samples = len(proto.non_tensor_batch['mooncake_keys'])
    rank_indices = np.array_split(np.arange(n_samples), args.dp_size)

    for rank in range(args.dp_size):
        indices = rank_indices[rank]
        print(f"  Rank {rank}: {len(indices)} samples (indices {indices[0]}..{indices[-1]})")

    # ── Phase 4: Simulate worker-side Mooncake fetch + verify ──
    print(f"\n{'=' * 70}")
    print(f"[Phase 4] Worker-side: Mooncake get → verify → cleanup")
    print(f"{'=' * 70}")

    all_ok = True
    total_get_ms = 0.0

    for rank in range(args.dp_size):
        indices = rank_indices[rank]
        rank_keys = proto.non_tensor_batch['mooncake_keys'][indices]
        rank_shapes = proto.non_tensor_batch['shapes'][indices]
        rank_dtypes = proto.non_tensor_batch['dtypes'][indices]

        print(f"\n  ── Rank {rank} ({len(rank_keys)} samples) ──")

        for j, (key, shapes, dtypes) in enumerate(zip(rank_keys, rank_shapes, rank_dtypes)):
            t0 = time.time()
            result = reader_store.get(
                key=key,
                shapes=shapes,
                dtypes=dtypes,
                device=torch.device("cpu"),
            )
            get_ms = (time.time() - t0) * 1000
            total_get_ms += get_ms

            # Verify we got tensors with expected shapes
            hs_shape = result.hidden_states.shape
            ids_shape = result.input_ids.shape
            lhs_ok = result.last_hidden_states is not None

            expected_hs_shape = tuple(shapes["hidden_states"])
            expected_ids_shape = tuple(shapes["input_ids"])

            shape_match = (hs_shape == expected_hs_shape and ids_shape == expected_ids_shape)

            if not shape_match:
                print(f"    [{j}] key={key}  SHAPE MISMATCH!")
                print(f"         hs: got {hs_shape}, expected {expected_hs_shape}")
                print(f"         ids: got {ids_shape}, expected {expected_ids_shape}")
                all_ok = False
            elif j < 2 or j == len(rank_keys) - 1:
                hs_bytes = result.hidden_states.numel() * result.hidden_states.element_size()
                print(f"    [{j}] key={key}  get {fmt_bytes(hs_bytes)} in {get_ms:.1f}ms  "
                      f"hs={list(hs_shape)} ids={list(ids_shape)} lhs={'yes' if lhs_ok else 'NO'}  OK")
                if j == 1 and len(rank_keys) > 3:
                    print(f"    ... ({len(rank_keys) - 3} more) ...")

            # Cleanup from Mooncake
            reader_store.remove_eagle3_tensors(key, has_last_hidden_states=True)

    avg_get = total_get_ms / n_samples if n_samples else 0
    print(f"\n  Total get time: {total_get_ms:.1f}ms ({avg_get:.1f}ms avg)")

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print(f"  Single-Controller HS Pipeline Simulation")
    print(f"{'=' * 70}")
    print(f"  Sequences:  {args.num_sequences}")
    print(f"  DP size:    {args.dp_size}")
    print(f"  Seq len:    {args.seq_len}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Phases:")
    print(f"    1. push_raw_prompts   → {args.num_sequences} sequences into Level 1")
    print(f"    2. HS collection      → {len(sample_metas)} Mooncake puts ({total_put_ms:.0f}ms)")
    print(f"    3. drain_as_dataproto → DataProto split across {args.dp_size} ranks")
    print(f"    4. Mooncake get       → {n_samples} fetches ({total_get_ms:.0f}ms)")
    print(f"  Result: {'ALL PASSED' if all_ok else 'FAILURES DETECTED'}")
    print(f"{'=' * 70}")

    # Wait for deferred deletes
    wait = args.lease_ttl + 1.5
    print(f"\n  Waiting {wait:.1f}s for deferred deletes (TTL={args.lease_ttl}s)...")
    time.sleep(wait)

    writer_store.close()
    reader_store.close()

    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="Single-controller HS pipeline simulation (no GPU, no Ray, no vLLM)"
    )
    parser.add_argument("--master-host", default="localhost")
    parser.add_argument("--master-port", type=int, default=50051)
    parser.add_argument("--metadata-port", type=int, default=8090)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=896,
                        help="Hidden dimension (896 for Qwen2.5-0.5B)")
    parser.add_argument("--num-sequences", type=int, default=8,
                        help="Number of rollout sequences to simulate")
    parser.add_argument("--dp-size", type=int, default=2,
                        help="Data-parallel size for mesh dispatch simulation")
    parser.add_argument("--lease-ttl", type=float, default=2.0)
    parser.add_argument("--skip-master", action="store_true",
                        help="Don't launch mooncake_master (use existing)")
    args = parser.parse_args()

    print(f"{'=' * 70}")
    print(f"  Single-Controller HS Pipeline Simulation")
    print(f"  (DrafterDataController + Mooncake put/get, no GPU/Ray/vLLM)")
    print(f"{'=' * 70}")

    master_proc = None
    if not args.skip_master:
        print("\n[Setup] Launching mooncake_master...")
        lease_ttl_ms = int(args.lease_ttl * 1000)
        master_proc = launch_master(
            port=args.master_port,
            http_port=args.metadata_port,
            lease_ttl_ms=lease_ttl_ms,
        )
    else:
        print("\n[Setup] Skipping master launch (--skip-master)")

    try:
        ok = run_simulation(args)
    finally:
        if master_proc and master_proc.poll() is None:
            print("\n[Cleanup] Stopping mooncake_master...")
            time.sleep(0.5)
            master_proc.terminate()
            try:
                master_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                master_proc.kill()

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
