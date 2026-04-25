#!/usr/bin/env python3
"""
Test Mooncake store: put / get / remove cycle with sample tensors.

Prerequisites:
  1. Install mooncake: pip install mooncake

The script launches its own mooncake_master subprocess by default
(``--launch-master``, on by default). Use ``--no-launch-master`` to point at
an already-running master.

Usage:
  # All-in-one (launches its own master subprocess on localhost):
  python scripts/test_mooncake_store.py

  # With an externally-running master:
  python scripts/test_mooncake_store.py --no-launch-master --master-host 10.0.0.1

  # Dry run (test imports + config only, no mooncake server needed):
  python scripts/test_mooncake_store.py --dry-run
"""

import argparse
import atexit
import os
import signal
import socket
import subprocess
import sys
import time


def _setup_ipv6_environment():
    """Set up environment for IPv6-only environments.

    Must be called BEFORE importing mooncake.
    Returns True if IPv6-only mode was configured.
    """
    # Check if we're in IPv6-only environment
    try:
        result = subprocess.run(["hostname", "-I"], capture_output=True, text=True)
        has_ipv4 = any("." in addr and ":" not in addr for addr in result.stdout.strip().split())
        if has_ipv4:
            return False
    except Exception:
        return False

    # Set MC_USE_IPV6 for the client
    os.environ["MC_USE_IPV6"] = "1"
    print("[IPv6] MC_USE_IPV6=1 enabled for client (P2PHANDSHAKE mode)")
    return True


# Set up IPv6 environment BEFORE any other imports
_IPV6_MODE = _setup_ipv6_environment()


def parse_args():
    parser = argparse.ArgumentParser(description="Test Mooncake store put/get/remove")
    parser.add_argument("--master-host", default="localhost", help="Mooncake master host")
    parser.add_argument("--master-port", type=int, default=50051, help="Mooncake master gRPC port")
    parser.add_argument("--metadata-port", type=int, default=8090, help="HTTP metadata port")
    parser.add_argument("--protocol", default="tcp", choices=["tcp", "rdma"])
    parser.add_argument("--device", default="cpu", help="Tensor device (cpu or cuda:0)")
    parser.add_argument("--seq-len", type=int, default=128, help="Test sequence length")
    parser.add_argument("--hidden-dim", type=int, default=4096, help="Hidden dimension")
    parser.add_argument("--dry-run", action="store_true", help="Test imports only, no server needed")
    parser.add_argument(
        "--launch-master",
        dest="launch_master",
        action="store_true",
        default=True,
        help="Launch a local mooncake_master subprocess (default: on)",
    )
    parser.add_argument(
        "--no-launch-master",
        dest="launch_master",
        action="store_false",
        help="Use an externally-running mooncake_master",
    )
    return parser.parse_args()


def _wait_for_port(host: str, port: int, timeout: float = 10.0) -> bool:
    deadline = time.time() + timeout
    # Use appropriate socket family for IPv6
    sock_family = socket.AF_INET6 if ":" in host else socket.AF_INET
    while time.time() < deadline:
        with socket.socket(sock_family, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            try:
                s.connect((host, port))
                return True
            except OSError:
                time.sleep(0.1)
    return False


def launch_local_master(args) -> "subprocess.Popen | None":
    """Spawn a mooncake_master subprocess on the configured ports.

    Lighter-weight than recipe.drafter_cotraining.mooncake.master.launch_mooncake_master
    (no Ray actor needed for a simple sanity test). Returns None if the
    binary is missing.
    """
    from recipe.drafter_cotraining.mooncake.master import (
        resolve_mooncake_master_bin,
        _is_ipv6_only_environment,
    )

    bin_path = resolve_mooncake_master_bin()
    if not os.path.exists(bin_path):
        print(f"  mooncake_master binary not found at {bin_path}; skipping launch")
        return None

    # Detect IPv6-only environment for P2PHANDSHAKE mode
    ipv6_only = _is_ipv6_only_environment()

    if ipv6_only:
        # In IPv6-only environments, use P2PHANDSHAKE mode:
        # - Disable HTTP metadata server (coro_http has IPv6 DNS resolution bug)
        # - Bind RPC to :: (IPv6 any address)
        cmd = [
            bin_path,
            f"--rpc_port={args.master_port}",
            "--rpc_address=::",
            "--enable_http_metadata_server=false",
            "--enable_metric_reporting=false",
        ]
        env = os.environ.copy()
        env["MC_USE_IPV6"] = "1"
        print(f"  Launching mooncake_master (IPv6-only/P2PHANDSHAKE mode): {' '.join(cmd)}")
    else:
        cmd = [
            bin_path,
            f"--rpc_port={args.master_port}",
            f"--http_metadata_server_port={args.metadata_port}",
            "--enable_http_metadata_server=true",
        ]
        env = os.environ.copy()
        print(f"  Launching mooncake_master: {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,  # so we can kill the whole process group
        env=env,
    )

    def _cleanup():
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=5)
            except Exception:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    pass

    atexit.register(_cleanup)

    # In IPv6-only environments, use ::1 instead of localhost
    connect_host = args.master_host
    if ipv6_only and args.master_host in ("localhost", "127.0.0.1"):
        connect_host = "::1"

    if not _wait_for_port(connect_host, args.master_port, timeout=30.0):
        print(f"  mooncake_master did not become reachable on {connect_host}:{args.master_port}")
        _cleanup()
        sys.exit(1)

    # Skip metadata port check in IPv6-only mode (P2PHANDSHAKE)
    if not ipv6_only:
        if not _wait_for_port(connect_host, args.metadata_port, timeout=30.0):
            print(f"  metadata server did not become reachable on {connect_host}:{args.metadata_port}")
            _cleanup()
            sys.exit(1)
        print(f"  mooncake_master ready on {connect_host}:{args.master_port} "
              f"(metadata :{args.metadata_port}), PID={proc.pid}\n")
    else:
        print(f"  mooncake_master ready on {connect_host}:{args.master_port} "
              f"(P2PHANDSHAKE mode), PID={proc.pid}\n")
        # Set metadata_server for P2PHANDSHAKE mode
        args.p2p_handshake = True

    # Update args.master_host so downstream code uses the correct address
    args.master_host = connect_host
    args.ipv6_only = ipv6_only
    return proc


def test_imports():
    """Step 1: Verify all mooncake modules import cleanly."""
    print("=" * 60)
    print("Step 1: Testing imports")
    print("=" * 60)

    from recipe.drafter_cotraining.mooncake.config import MooncakeConfig
    print(f"  MooncakeConfig: OK")

    from recipe.drafter_cotraining.mooncake.helpers import calculate_eagle3_buffer_size
    print(f"  calculate_eagle3_buffer_size: OK")

    from recipe.drafter_cotraining.mooncake.buffers import HostBuffer, HostBufferPool, AsyncPutManager
    print(f"  HostBuffer, HostBufferPool, AsyncPutManager: OK")

    from recipe.drafter_cotraining.mooncake.store import MooncakeHiddenStateStore
    print(f"  MooncakeHiddenStateStore: OK")

    from recipe.drafter_cotraining.mooncake.eagle_store import EagleMooncakeStore, Eagle3TargetOutput
    print(f"  EagleMooncakeStore, Eagle3TargetOutput: OK")

    print("\n  All imports passed.\n")
    return MooncakeConfig, EagleMooncakeStore, Eagle3TargetOutput, calculate_eagle3_buffer_size


def test_config(args, MooncakeConfig):
    """Step 2: Create and inspect config."""
    print("=" * 60)
    print("Step 2: Testing MooncakeConfig")
    print("=" * 60)

    # Use P2PHANDSHAKE for IPv6-only environments
    if getattr(args, "ipv6_only", False):
        config = MooncakeConfig(
            local_hostname=args.master_host if ":" in args.master_host else "::1",
            metadata_server="P2PHANDSHAKE",
            master_server_address=f"[{args.master_host}]:{args.master_port}" if ":" in args.master_host else f"{args.master_host}:{args.master_port}",
            global_segment_size=4294967296,
            local_buffer_size=536870912,
            host_buffer_size=8388608,
            protocol=args.protocol,
            device_name="",
            enable_gpu_direct=False,
            max_seq_len=args.seq_len,
            hidden_dim=args.hidden_dim,
        )
    else:
        config = MooncakeConfig.from_master_address(
            master_host=args.master_host,
            master_port=args.master_port,
            metadata_port=args.metadata_port,
            protocol=args.protocol,
            max_seq_len=args.seq_len,
            hidden_dim=args.hidden_dim,
        )

    print(f"  master_server_address: {config.master_server_address}")
    print(f"  metadata_server:       {config.metadata_server}")
    print(f"  protocol:              {config.protocol}")
    print(f"  global_segment_size:   {config.global_segment_size / 1024**3:.1f} GB")
    print(f"  local_buffer_size:     {config.local_buffer_size / 1024**2:.1f} MB")
    print(f"  host_buffer_size:      {config.host_buffer_size / 1024**2:.1f} MB")
    print(f"  async_put_pool_size:   {config.async_put_pool_size}")
    print(f"  kv_lease_ttl_s:        {config.kv_lease_ttl_s}")
    print()

    return config


def test_buffer_size(args, calculate_eagle3_buffer_size):
    """Step 3: Test buffer size calculation."""
    print("=" * 60)
    print("Step 3: Testing buffer size calculation")
    print("=" * 60)

    size = calculate_eagle3_buffer_size(
        max_seq_len=args.seq_len,
        batch_size=1,
        hidden_dim=args.hidden_dim,
        num_aux_layers=3,
    )
    print(f"  Buffer for 1 sample (seq={args.seq_len}, dim={args.hidden_dim}): {size / 1024**2:.1f} MB")

    size_8 = calculate_eagle3_buffer_size(
        max_seq_len=args.seq_len,
        batch_size=8,
        hidden_dim=args.hidden_dim,
        num_aux_layers=3,
    )
    print(f"  Buffer for 8 samples: {size_8 / 1024**2:.1f} MB")
    print()


def test_put_get_remove(args, config, EagleMooncakeStore, Eagle3TargetOutput):
    """Step 4: Full put/get/remove cycle."""
    import torch

    print("=" * 60)
    print("Step 4: Testing put / get / remove cycle")
    print("=" * 60)

    device = torch.device(args.device)

    # Create store and connect
    print(f"  Creating EagleMooncakeStore...")
    store = EagleMooncakeStore(config)

    print(f"  Connecting to Mooncake master at {config.master_server_address}...")
    store.setup(device=device if device.type == "cuda" else None)
    print(f"  Connected.\n")

    # Create sample tensors
    seq_len = args.seq_len
    hidden_dim = args.hidden_dim
    num_aux_layers = 3

    hidden_states = torch.randn(seq_len, hidden_dim * num_aux_layers, dtype=torch.bfloat16, device=device)
    input_ids = torch.randint(0, 32000, (seq_len,), dtype=torch.int64, device=device)
    last_hidden_states = torch.randn(seq_len, hidden_dim, dtype=torch.bfloat16, device=device)

    print(f"  Sample tensors created:")
    print(f"    hidden_states:      {hidden_states.shape} ({hidden_states.dtype})")
    print(f"    input_ids:          {input_ids.shape} ({input_ids.dtype})")
    print(f"    last_hidden_states: {last_hidden_states.shape} ({last_hidden_states.dtype})")
    print()

    # PUT
    key = "test_sample_001"
    print(f"  PUT key={key}...")
    t0 = time.time()
    metadata = store.put(
        key=key,
        hidden_states=hidden_states,
        input_ids=input_ids,
        last_hidden_states=last_hidden_states,
    )
    store.flush()
    t1 = time.time()
    print(f"  PUT completed in {(t1-t0)*1000:.1f} ms")
    print(f"  Metadata returned: shapes={metadata['shapes']}")
    print(f"                     dtypes={metadata['dtypes']}")
    print()

    # GET
    print(f"  GET key={key}...")
    t0 = time.time()
    output = store.get(
        key=key,
        shapes=metadata["shapes"],
        dtypes=metadata["dtypes"],
        device=device,
    )
    t1 = time.time()
    print(f"  GET completed in {(t1-t0)*1000:.1f} ms")
    print(f"  Retrieved: type={type(output).__name__}")
    print(f"    hidden_states:      {output.hidden_states.shape} ({output.hidden_states.dtype})")
    print(f"    input_ids:          {output.input_ids.shape} ({output.input_ids.dtype})")
    if output.last_hidden_states is not None:
        print(f"    last_hidden_states: {output.last_hidden_states.shape}")
    print()

    # Verify data integrity
    print(f"  Verifying data integrity...")
    hs_match = torch.allclose(hidden_states.cpu().float(), output.hidden_states.cpu().float(), atol=1e-2)
    ids_match = torch.equal(input_ids.cpu(), output.input_ids.cpu())
    print(f"    hidden_states match: {hs_match}")
    print(f"    input_ids match:     {ids_match}")
    if not hs_match:
        diff = (hidden_states.cpu().float() - output.hidden_states.cpu().float()).abs().max()
        print(f"    hidden_states max diff: {diff.item():.6f} (bf16 precision)")
    print()

    # REMOVE
    print(f"  REMOVE key={key}...")
    store.remove_eagle3_tensors(
        key=key,
        has_last_hidden_states=True,
        has_target=False,
    )
    print(f"  Removal force-deleted (batch_remove force=True).")
    print()

    # Cleanup
    store.close()
    print(f"  Store closed.\n")

    return hs_match and ids_match


def main():
    args = parse_args()

    MooncakeConfig, EagleMooncakeStore, Eagle3TargetOutput, calc_buf = test_imports()
    test_buffer_size(args, calc_buf)

    if args.dry_run:
        # For dry-run, create config with defaults
        config = test_config(args, MooncakeConfig)
        print("=" * 60)
        print("Dry run complete. All imports and config OK.")
        print("Run without --dry-run with a Mooncake master to test put/get/remove.")
        print("=" * 60)
        return

    if args.launch_master:
        print("=" * 60)
        print("Step 4a: Launching local mooncake_master subprocess")
        print("=" * 60)
        if launch_local_master(args) is None:
            print("Master not launched; pass --no-launch-master and start one yourself.")
            sys.exit(1)

    # Create config AFTER master is launched (so args.master_host is updated for IPv6)
    config = test_config(args, MooncakeConfig)
    success = test_put_get_remove(args, config, EagleMooncakeStore, Eagle3TargetOutput)

    print("=" * 60)
    if success:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
