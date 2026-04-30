#!/usr/bin/env python3
"""Diagnose the Mooncake "store-not-ready" hypothesis.

Replicates the HS-collector / rollout co-existence: launches a mooncake_master,
spins up two EagleMooncakeStore clients (mimicking two vLLM workers), and
measures (a) when each segment becomes visible at the metadata server, and
(b) whether a put issued immediately after setup() succeeds.

Run:
    python scripts/diag_mooncake_init.py
"""

import argparse
import atexit
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.parse
import urllib.request

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--master-host", default="localhost")
    p.add_argument("--master-port", type=int, default=50051)
    p.add_argument("--metadata-port", type=int, default=8090)
    p.add_argument("--num-stores", type=int, default=2,
                   help="How many EagleMooncakeStore clients to spin up (mimic vLLM workers)")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--hidden-dim", type=int, default=4096)
    return p.parse_args()


# -- Master lifecycle (copied from test_mooncake_store launcher) --
def _wait_for_port(host, port, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            try:
                s.connect((host, port)); return True
            except OSError:
                time.sleep(0.1)
    return False


def launch_master(args):
    from recipe.drafter_cotraining.mooncake.master import resolve_mooncake_master_bin
    bin_path = resolve_mooncake_master_bin()
    cmd = [bin_path,
           f"--port={args.master_port}",
           f"--http_metadata_server_port={args.metadata_port}",
           "--enable_http_metadata_server=true"]
    print(f"[master] launching: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            preexec_fn=os.setsid)

    def cleanup():
        if proc.poll() is None:
            try: os.killpg(os.getpgid(proc.pid), signal.SIGTERM); proc.wait(timeout=5)
            except Exception:
                try: os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception: pass
    atexit.register(cleanup)

    if not _wait_for_port(args.master_host, args.master_port):
        sys.exit("master gRPC port never opened")
    if not _wait_for_port(args.master_host, args.metadata_port):
        sys.exit("master HTTP metadata port never opened")
    print(f"[master] ready on {args.master_host}:{args.master_port} (metadata :{args.metadata_port})\n")
    return proc


def poll_metadata_for_segment(args, segment_name, timeout=10.0):
    """Return ms-elapsed until http://host:metadata_port/metadata?key=mooncake/ram/<segment> stops returning 404.

    None if it never appears within `timeout`.
    """
    url = (
        f"http://{args.master_host}:{args.metadata_port}/metadata"
        f"?key={urllib.parse.quote('mooncake/ram/' + segment_name, safe='')}"
    )
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url, timeout=0.5) as r:
                if r.status == 200 and r.read():
                    return (time.time() - t0) * 1000
        except urllib.error.HTTPError as e:
            if e.code != 404:
                return None
        except Exception:
            pass
        time.sleep(0.005)  # 5 ms tick
    return None


def make_store(args, store_idx, device):
    from recipe.drafter_cotraining.mooncake.config import MooncakeConfig
    from recipe.drafter_cotraining.mooncake.eagle_store import EagleMooncakeStore
    config = MooncakeConfig.from_master_address(
        master_host=args.master_host,
        master_port=args.master_port,
        metadata_port=args.metadata_port,
        protocol="tcp",
        max_seq_len=args.seq_len,
        hidden_dim=args.hidden_dim,
    )
    print(f"[store {store_idx}] config: master={config.master_server_address} "
          f"local_hostname={config.local_hostname} "
          f"host_buffer_size={config.host_buffer_size}")
    return EagleMooncakeStore(config), config


def time_put(store, args, device, label):
    """Try a single batch_put_from of input_ids-sized data; return (ok, code, ms)."""
    import uuid
    if store._host_buffer_pool is None:
        return None, "no-pool", 0.0
    buf = store._host_buffer_pool.get_buffer()
    key = f"diag_{label}_{uuid.uuid4().hex[:8]}_ids"
    sz = args.seq_len * 8  # int64
    t0 = time.time()
    try:
        results = store._store.batch_put_from([key], [buf.ptr], [sz])
    except Exception as e:
        return False, f"raised: {e}", (time.time() - t0) * 1000
    ms = (time.time() - t0) * 1000
    code = results[0] if results else None
    if code == 0:
        try: store._store.remove(key)
        except Exception: pass
        return True, 0, ms
    return False, code, ms


def get_segment_name(store):
    """Construct the segment name Mooncake registers (best guess: `<local_hostname>:<port>`).

    Mooncake's segment key format observed in logs:
        mooncake/ram/<local_hostname>:<port>
    The port is auto-assigned at setup() time; the only way to read it back is
    from store internals. We don't actually need the exact name here — we just
    use this for a heuristic poll. If the poll never succeeds the test still
    catches that.
    """
    cfg = store.config
    hn = cfg.local_hostname
    return hn  # Just the hostname prefix; we'll rely on the put result for ground truth.


def main():
    args = parse_args()
    launch_master(args)
    device = torch.device("cpu")  # avoid GPU dependency for this diag

    print("=" * 70)
    print("Phase A — single store, immediate put after setup()")
    print("=" * 70)
    store0, cfg0 = make_store(args, 0, device)
    t_setup0 = time.time()
    store0.setup(device=None)
    print(f"[store 0] setup() returned in {(time.time()-t_setup0)*1000:.1f} ms")

    # Try puts at increasing delays after setup
    for delay_ms in [0, 50, 200, 500, 1000, 2000]:
        target_ts = t_setup0 + delay_ms / 1000.0
        sleep_for = max(0.0, target_ts - time.time())
        if sleep_for > 0:
            time.sleep(sleep_for)
        ok, code, ms = time_put(store0, args, device, f"phaseA_{delay_ms}ms")
        elapsed_total = (time.time() - t_setup0) * 1000
        print(f"  put @ ~{delay_ms:>4d} ms post-setup (actual {elapsed_total:5.0f} ms): "
              f"ok={ok}, code={code}, took={ms:.1f} ms")

    print()
    print("=" * 70)
    print(f"Phase B — {args.num_stores} stores; immediate put on each, no barrier")
    print("=" * 70)
    stores = [store0]
    for i in range(1, args.num_stores):
        s, _ = make_store(args, i, device)
        ts = time.time()
        s.setup(device=None)
        print(f"[store {i}] setup() in {(time.time()-ts)*1000:.1f} ms")
        ok, code, ms = time_put(s, args, device, f"phaseB_immediate_{i}")
        print(f"  immediate put on store {i}: ok={ok}, code={code}, took={ms:.1f} ms")
        stores.append(s)

    print()
    print("=" * 70)
    print("Phase C — warmup_rdma() effectiveness")
    print("=" * 70)
    storeC, _ = make_store(args, 99, device)
    storeC.setup(device=None)
    print("[store 99] calling warmup_rdma()...")
    tw = time.time()
    try:
        storeC.warmup_rdma()
        print(f"  warmup_rdma() ok in {(time.time()-tw)*1000:.1f} ms")
    except Exception as e:
        print(f"  warmup_rdma() FAILED: {e}")
    ok, code, ms = time_put(storeC, args, device, "phaseC_after_warmup")
    print(f"  put after warmup: ok={ok}, code={code}, took={ms:.1f} ms")

    print()
    print("=" * 70)
    print("Phase D — repeated put loop (does it stabilize after first failure?)")
    print("=" * 70)
    storeD, _ = make_store(args, 999, device)
    storeD.setup(device=None)
    fail_streak = 0
    first_ok_ms = None
    t_start = time.time()
    for i in range(50):
        ok, code, ms = time_put(storeD, args, device, f"phaseD_{i}")
        elapsed = (time.time() - t_start) * 1000
        if ok and first_ok_ms is None:
            first_ok_ms = elapsed
        if not ok:
            fail_streak += 1
            print(f"  put #{i:02d} @ {elapsed:6.0f} ms: code={code}")
        time.sleep(0.020)
    print(f"  first successful put at {first_ok_ms} ms (None = none succeeded), "
          f"failures={fail_streak}/50")

    print()
    print("=" * 70)
    print("Cleanup")
    print("=" * 70)
    for s in stores + [storeC, storeD]:
        try: s.close()
        except Exception: pass


if __name__ == "__main__":
    main()
