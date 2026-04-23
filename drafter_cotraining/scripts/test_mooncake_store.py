#!/usr/bin/env python3
"""
Test Mooncake store: put / get / remove cycle with sample tensors.

Prerequisites:
  1. Install mooncake: pip install mooncake
  2. Start mooncake master (one of):
     a. mooncake_master --port=50051 --http_metadata_server_port=8090 --enable_http_metadata_server=true
     b. python -c "from recipe.drafter_cotraining.mooncake.master import resolve_mooncake_master_bin; print(resolve_mooncake_master_bin())"

Usage:
  # With a running mooncake master on localhost:
  python scripts/test_mooncake_store.py

  # With a remote master:
  python scripts/test_mooncake_store.py --master-host 10.0.0.1

  # Dry run (test imports + config only, no mooncake server needed):
  python scripts/test_mooncake_store.py --dry-run
"""

import argparse
import sys
import time


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
    return parser.parse_args()


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

    from recipe.drafter_cotraining.mooncake.deferred_delete import DeferredDeleteManager
    print(f"  DeferredDeleteManager: OK")

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
    print(f"  Removal queued (deferred delete after TTL).")
    print()

    # Cleanup
    store.close()
    print(f"  Store closed.\n")

    return hs_match and ids_match


def main():
    args = parse_args()

    MooncakeConfig, EagleMooncakeStore, Eagle3TargetOutput, calc_buf = test_imports()
    config = test_config(args, MooncakeConfig)
    test_buffer_size(args, calc_buf)

    if args.dry_run:
        print("=" * 60)
        print("Dry run complete. All imports and config OK.")
        print("Run without --dry-run with a Mooncake master to test put/get/remove.")
        print("=" * 60)
        return

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
