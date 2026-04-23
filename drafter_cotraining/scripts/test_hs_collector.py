#!/usr/bin/env python3
"""
Standalone test for HSCollectorManager.

Launches HSCollectorManager in standalone mode (no trainer, its own GPUs),
tokenizes prompts, pushes them through the manager's async server path,
extracts kv_transfer_params from the responses, and reads the hidden states
back from Mooncake via a separate reader store.

Proves end-to-end:
  TokenOutput.extra_fields["kv_transfer_params"] → Mooncake.get → hidden_states

Requires: 1 GPU, mooncake_master binary, a HuggingFace model.

Usage:
    python scripts/test_hs_collector.py
    python scripts/test_hs_collector.py --model Qwen/Qwen2.5-0.5B-Instruct --gpu 0
    python scripts/test_hs_collector.py --skip-master   # if master already running
"""

import argparse
import asyncio
import atexit
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from uuid import uuid4

import ray
import torch
from omegaconf import OmegaConf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_hs_collector")


# ─── Mooncake master lifecycle ────────────────────────────────────────


def find_mooncake_master_bin():
    if "MOONCAKE_BUILD_DIR" in os.environ:
        return os.path.join(os.environ["MOONCAKE_BUILD_DIR"], "mooncake-store/src/mooncake_master")
    found = shutil.which("mooncake_master")
    if found:
        return found
    return os.path.expanduser("~/build/mooncake-store/src/mooncake_master")


def launch_master(port, http_port, lease_ttl_ms):
    binary = find_mooncake_master_bin()
    if not os.path.exists(binary):
        print(f"ERROR: mooncake_master not found at {binary}")
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
        print(f"ERROR: mooncake_master exited with code {proc.returncode}\n  {stderr[:500]}")
        sys.exit(1)
    print(f"  mooncake_master running (PID {proc.pid})")
    return proc


# ─── Config helpers ───────────────────────────────────────────────────


def get_aux_layer_ids(model_path):
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    cfg = getattr(cfg, "text_config", cfg)
    n = cfg.num_hidden_layers
    aux = [1, max(1, n // 2 - 1), max(2, n - 4)]
    if (n - 1) not in aux:
        aux.append(n - 1)
    return sorted(set(aux)), cfg.hidden_size


def make_mooncake_config(host, master_port, metadata_port, max_seq_len, hidden_dim, lease_ttl):
    from recipe.drafter_cotraining.mooncake.config import MooncakeConfig

    return MooncakeConfig(
        local_hostname=host,
        metadata_server=f"http://{host}:{metadata_port}/metadata",
        master_server_address=f"{host}:{master_port}",
        global_segment_size=2 * 1024 * 1024 * 1024,
        local_buffer_size=512 * 1024 * 1024,
        protocol="tcp",
        max_seq_len=max_seq_len,
        hidden_dim=hidden_dim,
        async_put_pool_size=4,
        kv_lease_ttl_s=lease_ttl,
    )


def build_hs_collector_config(model_path, aux_layer_ids, max_seq_len, gpu_mem, n_gpus):
    """Minimal HSCollectorConfig-shaped DictConfig (standalone mode)."""
    return OmegaConf.create(
        {
            "enabled": True,
            "model_path": model_path,
            "n_gpus_per_node": n_gpus,
            "nnodes": 1,
            "inference": {
                "name": "vllm",
                "tensor_model_parallel_size": 1,
                "data_parallel_size": 1,
                "pipeline_model_parallel_size": 1,
                "max_model_len": max_seq_len,
                "gpu_memory_utilization": gpu_mem,
                "dtype": "bfloat16",
                "enforce_eager": False,
                "enable_chunked_prefill": False,
                "enable_prefix_caching": False,
                "engine_kwargs": {
                    "vllm": {
                        "trust_remote_code": True,
                        "distributed_executor_backend": "mp",
                        "disable_custom_all_reduce": True,
                        "kv_transfer_config": {
                            "kv_connector": "MooncakeHiddenStatesConnector",
                            "kv_connector_module_path": "recipe.drafter_cotraining.mooncake.hidden_states_connector",
                            "kv_role": "kv_producer",
                        },
                        "speculative_config": {
                            "method": "extract_hidden_states",
                            "num_speculative_tokens": 1,
                            "draft_model_config": {
                                "hf_config": {"eagle_aux_hidden_state_layer_ids": list(aux_layer_ids)},
                            },
                        },
                    },
                },
            },
        }
    )


# ─── Prompts ──────────────────────────────────────────────────────────

SAMPLE_PROMPTS = [
    "Explain gradient descent.",
    "What is the capital of France?",
    "Write a Python function that returns the Fibonacci sequence.",
    "Summarize the theory of relativity in two sentences.",
    "What are the main differences between TCP and UDP?",
    "Describe the water cycle.",
    "What is RLHF?",
    "Explain speculative decoding.",
]


# ─── Main test ────────────────────────────────────────────────────────


def run_test(args):
    from transformers import AutoTokenizer

    from recipe.drafter_cotraining.hs_collector import HSCollectorManager
    from recipe.drafter_cotraining.mooncake.eagle_store import EagleMooncakeStore

    aux_layers, hidden_size = get_aux_layer_ids(args.model)
    print(f"\n  Model={args.model} hidden={hidden_size} aux_layers={aux_layers}")

    mc_config = make_mooncake_config(
        host=args.master_host,
        master_port=args.master_port,
        metadata_port=args.metadata_port,
        max_seq_len=args.max_seq_len,
        hidden_dim=hidden_size,
        lease_ttl=args.lease_ttl,
    )
    mc_config.export_env()

    # Reader store (consumer side)
    reader_store = EagleMooncakeStore(mc_config)
    reader_store.setup()

    hs_config = build_hs_collector_config(
        model_path=args.model,
        aux_layer_ids=aux_layers,
        max_seq_len=args.max_seq_len,
        gpu_mem=args.gpu_mem,
        n_gpus=1,
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    print("\n[Phase 1] Launching HSCollectorManager (standalone)...")
    t0 = time.time()
    manager = HSCollectorManager(config=hs_config, resource_pool=None)
    print(f"  Ready in {time.time() - t0:.1f}s ({len(manager.rollout_replicas)} replicas)")

    print("\n[Phase 2] Tokenizing prompts...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts = [SAMPLE_PROMPTS[i % len(SAMPLE_PROMPTS)] for i in range(args.num_prompts)]
    tokenized = [tokenizer.encode(p, add_special_tokens=True) for p in prompts]
    for i, ids in enumerate(tokenized[:3]):
        print(f"  [{i}] {len(ids):4d} tokens  \"{prompts[i][:60]}...\"")

    print("\n[Phase 3] Calling server_manager.compute_hidden_states_single directly...")
    # Bypass the DataProto-shaped batch API; call single-sample directly
    # so we don't have to construct a full prompts/responses split.
    manager.wake_up()
    try:

        async def collect_all():
            return await asyncio.gather(
                *[manager.server_manager.compute_hidden_states_single(ids) for ids in tokenized]
            )

        t0 = time.time()
        results = asyncio.run(collect_all())
        gen_sec = time.time() - t0
    finally:
        manager.sleep()
    print(f"  Collected {len(results)} samples in {gen_sec:.2f}s")

    print("\n[Phase 4] Mooncake fetch + verify...")
    all_ok = True
    for i, (tokens, r) in enumerate(zip(tokenized, results, strict=True)):
        key = r["mooncake_key"]
        shapes = r["shapes"]
        dtypes = {
            k: getattr(torch, v) if isinstance(v, str) and hasattr(torch, v) else v for k, v in r["dtypes"].items()
        }
        if not key:
            print(f"  [{i}] FAIL: empty mooncake_key (connector didn't fire)")
            all_ok = False
            continue
        got = reader_store.get(key=key, shapes=shapes, dtypes=dtypes, device=torch.device("cpu"))
        hs_ok = got.hidden_states.shape == tuple(shapes["hidden_states"]) and got.hidden_states.abs().sum().item() > 0
        ids_ok = got.input_ids.tolist() == tokens
        status = "OK" if (hs_ok and ids_ok) else "FAIL"
        if not (hs_ok and ids_ok):
            all_ok = False
        if i < 3 or i == len(results) - 1 or not (hs_ok and ids_ok):
            print(f"  [{i}] key={key} hs={list(got.hidden_states.shape)} ids_match={ids_ok} {status}")
        reader_store.remove_eagle3_tensors(key, has_last_hidden_states=got.last_hidden_states is not None)

    time.sleep(args.lease_ttl + 1.0)
    reader_store.close()
    print(f"\n  Result: {'ALL PASSED' if all_ok else 'FAILURES DETECTED'}")
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Standalone HSCollectorManager test")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gpu-mem", type=float, default=0.5)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--master-host", default="localhost")
    parser.add_argument("--master-port", type=int, default=50051)
    parser.add_argument("--metadata-port", type=int, default=8090)
    parser.add_argument("--lease-ttl", type=float, default=5.0)
    parser.add_argument("--skip-master", action="store_true")
    args = parser.parse_args()

    print(f"{'=' * 70}")
    print(f"  HSCollectorManager standalone test")
    print(f"  model={args.model} gpu={args.gpu} prompts={args.num_prompts}")
    print(f"{'=' * 70}")

    master_proc = None
    if not args.skip_master:
        print("\n[Setup] Launching mooncake_master...")
        master_proc = launch_master(
            port=args.master_port,
            http_port=args.metadata_port,
            lease_ttl_ms=int(args.lease_ttl * 1000),
        )

    ray.init(ignore_reinit_error=True)
    ok = False
    try:
        ok = run_test(args)
    except KeyboardInterrupt:
        print("\n  Interrupted")
    except Exception:
        logger.exception("Test failed")
    finally:
        ray.shutdown()
        if master_proc and master_proc.poll() is None:
            time.sleep(0.5)
            master_proc.terminate()
            try:
                master_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                master_proc.kill()

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
