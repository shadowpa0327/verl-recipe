#!/usr/bin/env python3
"""
Toy example: Mooncake hidden-state put/get round-trip.

Auto-launches mooncake_master, creates synthetic Eagle3 hidden states,
writes them to the store, reads them back, and verifies bit-exact match.

No GPU required — runs entirely on CPU with TCP transport.
No Ray required — launches mooncake_master as a plain subprocess.

Usage:
    python scripts/test_vllm_hs_collector.py

    # Custom sizes:
    python scripts/test_vllm_hs_collector.py --seq-len 256 --hidden-dim 3584 --num-samples 5
"""

import argparse
import atexit
import os
import shutil
import signal
import subprocess
import sys
import time

import torch


# ─── Mooncake master lifecycle (no Ray) ──────────────────────

def find_mooncake_master_bin():
    if "MOONCAKE_BUILD_DIR" in os.environ:
        return os.path.join(os.environ["MOONCAKE_BUILD_DIR"], "mooncake-store/src/mooncake_master")
    found = shutil.which("mooncake_master")
    if found:
        return found
    return os.path.expanduser("~/build/mooncake-store/src/mooncake_master")


def launch_master(port=50051, http_port=8090, lease_ttl_ms=2000):
    """Launch mooncake_master as a subprocess. Returns the Popen handle."""
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


# ─── Store helpers ───────────────────────────────────────────

def make_config(master_host, master_port, metadata_port, seq_len, hidden_dim, lease_ttl=2.0):
    """Create MooncakeConfig for a CPU-only TCP test."""
    from recipe.drafter_cotraining.mooncake.config import MooncakeConfig

    return MooncakeConfig(
        local_hostname=master_host,
        metadata_server=f"http://{master_host}:{metadata_port}/metadata",
        master_server_address=f"{master_host}:{master_port}",
        global_segment_size=256 * 1024 * 1024,   # 256 MB (small for toy test)
        local_buffer_size=128 * 1024 * 1024,      # 128 MB
        protocol="tcp",
        max_seq_len=seq_len,
        hidden_dim=hidden_dim,
        async_put_pool_size=1,
        kv_lease_ttl_s=lease_ttl,  # must match master's --default_kv_lease_ttl
    )


def make_eagle3_tensors(seq_len, hidden_dim, num_aux_layers=3):
    """Create synthetic tensors mimicking Eagle3 hidden-state extraction.

    Returns the same structure that MooncakeHiddenStatesConnector would
    produce during a vLLM prefill:
      - hidden_states: (seq_len, hidden_dim * num_aux_layers)  bf16
      - input_ids:     (seq_len,)                              int64
      - last_hidden_states: (seq_len, hidden_dim)              bf16
    """
    hidden_states = torch.randn(seq_len, hidden_dim * num_aux_layers, dtype=torch.bfloat16)
    input_ids = torch.randint(0, 152064, (seq_len,), dtype=torch.int64)  # Qwen2.5 vocab
    last_hidden_states = torch.randn(seq_len, hidden_dim, dtype=torch.bfloat16)
    return hidden_states, input_ids, last_hidden_states


def fmt_bytes(n):
    if n >= 1024**2:
        return f"{n / 1024**2:.1f} MB"
    if n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} B"


def tensor_info(name, t):
    nbytes = t.numel() * t.element_size()
    return f"    {name:25s} {str(list(t.shape)):20s} {str(t.dtype):12s} {fmt_bytes(nbytes):>10s}"


def tensor_preview(name, t, n=5):
    flat = t.flatten().float()
    vals = flat[:n].tolist()
    preview = ", ".join(f"{v:+.4f}" for v in vals)
    return f"    {name:25s} first {n}: [{preview}, ...]"


# ─── Main test ───────────────────────────────────────────────

def run_test(args):
    from recipe.drafter_cotraining.mooncake.eagle_store import EagleMooncakeStore

    config = make_config(
        master_host=args.master_host,
        master_port=args.master_port,
        metadata_port=args.metadata_port,
        seq_len=args.seq_len,
        hidden_dim=args.hidden_dim,
        lease_ttl=args.lease_ttl,
    )

    # ── Connect two store clients (simulates inference → training) ──
    print("\n[2/4] Connecting store clients...")
    writer = EagleMooncakeStore(config)
    writer.setup()
    print("  Writer (inference side): connected")

    reader = EagleMooncakeStore(config)
    reader.setup()
    print("  Reader (training side):  connected")

    all_ok = True

    for i in range(args.num_samples):
        key = f"sample_{i:04d}"
        print(f"\n{'─' * 60}")
        print(f"  Sample {i+1}/{args.num_samples}  key={key}")
        print(f"{'─' * 60}")

        # ── Create tensors ──
        hs, ids, lhs = make_eagle3_tensors(args.seq_len, args.hidden_dim)

        print("\n  [PUT] Tensors to write:")
        print(tensor_info("hidden_states", hs))
        print(tensor_info("input_ids", ids))
        print(tensor_info("last_hidden_states", lhs))
        total = hs.numel() * hs.element_size() + ids.numel() * ids.element_size() + lhs.numel() * lhs.element_size()
        print(f"    {'total':25s} {'':20s} {'':12s} {fmt_bytes(total):>10s}")

        print(f"\n  [PUT] Tensor previews:")
        print(tensor_preview("hidden_states", hs))
        print(tensor_preview("input_ids", ids.float()))
        print(tensor_preview("last_hidden_states", lhs))

        # ── Write ──
        t0 = time.time()
        metadata = writer.put(key=key, hidden_states=hs, input_ids=ids, last_hidden_states=lhs)
        writer.flush()
        put_ms = (time.time() - t0) * 1000
        print(f"\n  [PUT] Done in {put_ms:.1f} ms")
        print(f"    returned shapes: {metadata['shapes']}")

        # ── Read ──
        t0 = time.time()
        result = reader.get(
            key=key,
            shapes=metadata["shapes"],
            dtypes=metadata["dtypes"],
            device=torch.device("cpu"),
        )
        get_ms = (time.time() - t0) * 1000
        print(f"\n  [GET] Done in {get_ms:.1f} ms")
        print(f"  [GET] Retrieved tensors:")
        print(tensor_info("hidden_states", result.hidden_states))
        print(tensor_info("input_ids", result.input_ids))
        if result.last_hidden_states is not None:
            print(tensor_info("last_hidden_states", result.last_hidden_states))

        print(f"\n  [GET] Retrieved previews:")
        print(tensor_preview("hidden_states", result.hidden_states))
        print(tensor_preview("input_ids", result.input_ids.float()))
        if result.last_hidden_states is not None:
            print(tensor_preview("last_hidden_states", result.last_hidden_states))

        # ── Verify ──
        hs_ok = torch.equal(hs, result.hidden_states)
        ids_ok = torch.equal(ids, result.input_ids)
        lhs_ok = result.last_hidden_states is not None and torch.equal(lhs, result.last_hidden_states)

        status = "PASS" if (hs_ok and ids_ok and lhs_ok) else "FAIL"
        print(f"\n  [VERIFY] hidden_states:      {'OK' if hs_ok else 'MISMATCH'}")
        print(f"  [VERIFY] input_ids:          {'OK' if ids_ok else 'MISMATCH'}")
        print(f"  [VERIFY] last_hidden_states: {'OK' if lhs_ok else 'MISMATCH'}")
        print(f"  [VERIFY] ── {status} ──")

        if not (hs_ok and ids_ok and lhs_ok):
            all_ok = False

        # ── Cleanup ──
        reader.remove_eagle3_tensors(key, has_last_hidden_states=True)

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print(f"  {args.num_samples} samples, seq_len={args.seq_len}, hidden_dim={args.hidden_dim}")
    print(f"  Result: {'ALL PASSED' if all_ok else 'SOME FAILED'}")
    print(f"{'=' * 60}")

    # Give deferred deletes time to run (must exceed TTL + buffer)
    wait = args.lease_ttl + 1.5
    print(f"\n  Waiting {wait:.1f}s for deferred deletes (lease TTL={args.lease_ttl}s)...")
    time.sleep(wait)

    print("  Closing stores...")
    writer.close()
    reader.close()

    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Mooncake hidden-state put/get toy example")
    parser.add_argument("--master-host", default="localhost")
    parser.add_argument("--master-port", type=int, default=50051)
    parser.add_argument("--metadata-port", type=int, default=8090)
    parser.add_argument("--seq-len", type=int, default=64, help="Sequence length per sample")
    parser.add_argument("--hidden-dim", type=int, default=896, help="Hidden dimension (e.g. 896 for Qwen2.5-0.5B)")
    parser.add_argument("--num-samples", type=int, default=3, help="Number of put/get cycles")
    parser.add_argument("--lease-ttl", type=float, default=2.0, help="Mooncake lease TTL in seconds")
    parser.add_argument("--skip-master", action="store_true", help="Don't launch mooncake_master (use existing)")
    args = parser.parse_args()

    print(f"{'=' * 60}")
    print(f"  Mooncake Put/Get Toy Example")
    print(f"{'=' * 60}")

    # ── Step 1: Launch master ──
    master_proc = None
    if not args.skip_master:
        print("\n[1/4] Launching mooncake_master...")
        lease_ttl_ms = int(args.lease_ttl * 1000)
        master_proc = launch_master(port=args.master_port, http_port=args.metadata_port, lease_ttl_ms=lease_ttl_ms)
    else:
        print("\n[1/4] Skipping master launch (--skip-master)")

    # ── Step 2-4: Run test ──
    try:
        ok = run_test(args)
    finally:
        # Close stores BEFORE killing master to avoid segfault in C++ destructors
        if master_proc and master_proc.poll() is None:
            print("\n[cleanup] Stopping mooncake_master...")
            time.sleep(0.5)  # let C++ destructors finish
            master_proc.terminate()
            try:
                master_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                master_proc.kill()

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
