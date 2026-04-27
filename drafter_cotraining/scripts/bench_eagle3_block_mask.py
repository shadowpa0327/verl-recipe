"""Benchmark peak memory + wall time of `create_block_mask` for the EAGLE3 mask.

Diagnoses the cost of `compile_friendly_create_block_mask()` at the call site in
`recipe/drafter_cotraining/eagle3/draft/llama3_eagle.py:1434` (LlamaFlexAttention.forward).

Levers (per https://pytorch.org/blog/flexattention/#q-how-can-we-compute-blockmask-quicker):
  1. Broadcast over heads: H=None vs H=1 vs H=num_heads
  2. _compile=True on create_block_mask (independent of torch.compile of flex_attention)
  3. BLOCK_SIZE — quadratic effect on metadata size

Realistic EAGLE3 shapes (prompt_len=2048, response_len=2048, length=7 TTT steps):
  B  in {1, 2, 4, 8}            micro_batch_size_per_gpu (config default = 1)
  H  in {1 (broadcast), 32}     llama3_eagle.py passes H=1; draft model has 32 heads
  Q_LEN = 2048                  response length
  lck in 0..6                   TTT step index → KV_LEN = Q_LEN * (1 + lck)

Run examples:
  python bench_eagle3_block_mask.py --mode lck         # sweep KV_LEN
  python bench_eagle3_block_mask.py --mode batch       # sweep B
  python bench_eagle3_block_mask.py --mode head        # H=None vs H=1 vs H=32
  python bench_eagle3_block_mask.py --mode block       # BLOCK_SIZE sweep
  python bench_eagle3_block_mask.py --mode compile     # _compile=True vs False
  python bench_eagle3_block_mask.py --mode all         # everything
"""

from __future__ import annotations

import argparse
import gc
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch.nn.attention.flex_attention import create_block_mask, or_masks


def generate_eagle3_mask(seq_lengths: torch.Tensor, Q_LEN: int, KV_LEN: int, lck: int = 0):
    """Copied from recipe/drafter_cotraining/eagle3/ops/flex_attention.py."""

    def causal_mask(b, h, q_idx, kv_idx):
        c = q_idx >= kv_idx
        p = (kv_idx < seq_lengths[b]) & (q_idx < seq_lengths[b])
        return c & p

    def suffix_mask(b, h, q_idx, kv_idx):
        s = kv_idx >= Q_LEN
        p = kv_idx % Q_LEN < seq_lengths[b]
        d = (kv_idx - q_idx) % Q_LEN == 0
        return s & p & d

    mod = or_masks(causal_mask, suffix_mask)
    mod.__name__ = f"eagle3_Q{Q_LEN}_KV{KV_LEN}_lck{lck}"
    return mod


@dataclass
class Result:
    label: str
    B: int
    H_arg: object  # None or int
    Q_LEN: int
    KV_LEN: int
    block_size: int
    compile_flag: bool
    peak_mb: float = 0.0
    median_ms: float = 0.0
    sparsity: float = 0.0
    error: Optional[str] = None
    extra: dict = field(default_factory=dict)


def _bench_one(
    *,
    label: str,
    B: int,
    H_arg,
    Q_LEN: int,
    lck: int,
    block_size: int,
    compile_flag: bool,
    n_warmup: int = 1,
    n_iter: int = 3,
    device: str = "cuda",
) -> Result:
    KV_LEN = Q_LEN * (1 + lck)
    res = Result(label, B, H_arg, Q_LEN, KV_LEN, block_size, compile_flag)
    try:
        # Realistic seq_lengths: jitter inside [0.5*Q_LEN, Q_LEN] so mask actually
        # depends on b (mirrors what attention_mask.sum(-1) - lck looks like in training).
        torch.manual_seed(0)
        seq_lengths = torch.randint(Q_LEN // 2, Q_LEN + 1, (B,), device=device, dtype=torch.long)
        mask_mod = generate_eagle3_mask(seq_lengths, Q_LEN=Q_LEN, KV_LEN=KV_LEN, lck=lck)

        # Warmup (also pays compile cost so it does not bias timing).
        for _ in range(n_warmup):
            bm = create_block_mask(
                mask_mod,
                B=B,
                H=H_arg,
                Q_LEN=Q_LEN,
                KV_LEN=KV_LEN,
                device=device,
                BLOCK_SIZE=block_size,
                _compile=compile_flag,
            )
            del bm
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        times_ms = []
        peak = 0
        last_bm = None
        for _ in range(n_iter):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            bm = create_block_mask(
                mask_mod,
                B=B,
                H=H_arg,
                Q_LEN=Q_LEN,
                KV_LEN=KV_LEN,
                device=device,
                BLOCK_SIZE=block_size,
                _compile=compile_flag,
            )
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000)
            peak = max(peak, torch.cuda.max_memory_allocated())
            last_bm = bm

        times_ms.sort()
        res.median_ms = times_ms[len(times_ms) // 2]
        res.peak_mb = peak / (1024 * 1024)
        # Sparsity (fraction of blocks materialized): metadata size proxy.
        try:
            res.sparsity = float(last_bm.sparsity()) if last_bm is not None else 0.0
        except Exception:
            res.sparsity = -1.0

        # Capture metadata tensor sizes for diagnosis.
        if last_bm is not None:
            try:
                kvi = last_bm.kv_indices
                kvb = last_bm.kv_num_blocks
                res.extra["kv_indices_shape"] = tuple(kvi.shape)
                res.extra["kv_indices_mb"] = kvi.numel() * kvi.element_size() / (1024 * 1024)
                res.extra["kv_num_blocks_shape"] = tuple(kvb.shape)
                res.extra["kv_num_blocks_mb"] = kvb.numel() * kvb.element_size() / (1024 * 1024)
            except AttributeError:
                pass

        del last_bm
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:  # noqa: BLE001
        res.error = f"{type(e).__name__}: {e}"
    return res


def _print_table(rows: list[Result], title: str):
    print(f"\n=== {title} ===")
    hdr = (
        f"{'label':<28} {'B':>3} {'H':>5} {'Q_LEN':>6} {'KV_LEN':>7} "
        f"{'BLOCK':>5} {'cmp':>4} {'peak_MB':>9} {'time_ms':>9} {'sparse':>7} "
        f"{'kv_idx_MB':>10}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if r.error:
            print(f"{r.label:<28} ERROR: {r.error}")
            continue
        h = "None" if r.H_arg is None else str(r.H_arg)
        print(
            f"{r.label:<28} {r.B:>3} {h:>5} {r.Q_LEN:>6} {r.KV_LEN:>7} "
            f"{r.block_size:>5} {str(r.compile_flag):>4} "
            f"{r.peak_mb:>9.2f} {r.median_ms:>9.2f} {r.sparsity:>7.3f} "
            f"{r.extra.get('kv_indices_mb', float('nan')):>10.2f}"
        )


def mode_lck(args):
    """Sweep TTT step (KV_LEN = 2048 * (1+lck))."""
    rows = []
    for lck in args.lcks:
        rows.append(_bench_one(
            label=f"lck={lck}",
            B=args.B, H_arg=1, Q_LEN=args.Q_LEN, lck=lck,
            block_size=args.block_size, compile_flag=args.compile,
            n_warmup=args.warmup, n_iter=args.iters,
        ))
    _print_table(rows, f"KV_LEN sweep (B={args.B}, H=1, Q_LEN={args.Q_LEN}, "
                       f"BLOCK_SIZE={args.block_size}, _compile={args.compile})")
    return rows


def mode_batch(args):
    rows = []
    for B in args.batches:
        rows.append(_bench_one(
            label=f"B={B}",
            B=B, H_arg=1, Q_LEN=args.Q_LEN, lck=args.lck,
            block_size=args.block_size, compile_flag=args.compile,
            n_warmup=args.warmup, n_iter=args.iters,
        ))
    _print_table(rows, f"Batch sweep (H=1, Q_LEN={args.Q_LEN}, lck={args.lck}, "
                       f"BLOCK_SIZE={args.block_size}, _compile={args.compile})")
    return rows


def mode_head(args):
    """H=None (broadcast) vs H=1 (current code) vs H=32 (no broadcast)."""
    rows = []
    for H in [None, 1, 32]:
        rows.append(_bench_one(
            label=f"H={H}",
            B=args.B, H_arg=H, Q_LEN=args.Q_LEN, lck=args.lck,
            block_size=args.block_size, compile_flag=args.compile,
            n_warmup=args.warmup, n_iter=args.iters,
        ))
    _print_table(rows, f"H broadcast comparison (B={args.B}, Q_LEN={args.Q_LEN}, "
                       f"lck={args.lck}, BLOCK_SIZE={args.block_size}, _compile={args.compile})")
    return rows


def mode_block(args):
    rows = []
    for bs in args.block_sizes:
        rows.append(_bench_one(
            label=f"BS={bs}",
            B=args.B, H_arg=1, Q_LEN=args.Q_LEN, lck=args.lck,
            block_size=bs, compile_flag=args.compile,
            n_warmup=args.warmup, n_iter=args.iters,
        ))
    _print_table(rows, f"BLOCK_SIZE sweep (B={args.B}, H=1, Q_LEN={args.Q_LEN}, "
                       f"lck={args.lck}, _compile={args.compile})")
    return rows


def mode_compile(args):
    rows = []
    for cf in [False, True]:
        rows.append(_bench_one(
            label=f"_compile={cf}",
            B=args.B, H_arg=1, Q_LEN=args.Q_LEN, lck=args.lck,
            block_size=args.block_size, compile_flag=cf,
            n_warmup=args.warmup, n_iter=args.iters,
        ))
    _print_table(rows, f"_compile flag (B={args.B}, H=1, Q_LEN={args.Q_LEN}, "
                       f"lck={args.lck}, BLOCK_SIZE={args.block_size})")
    return rows


def mode_compile_grid(args):
    """lck × _compile cross product — does the win hold across the whole TTT loop?"""
    rows = []
    for lck in args.lcks:
        for cf in [False, True]:
            rows.append(_bench_one(
                label=f"lck={lck} _cmp={cf}",
                B=args.B, H_arg=1, Q_LEN=args.Q_LEN, lck=lck,
                block_size=args.block_size, compile_flag=cf,
                n_warmup=args.warmup, n_iter=args.iters,
            ))
    _print_table(rows, f"lck × _compile grid (B={args.B}, H=1, Q_LEN={args.Q_LEN}, "
                       f"BLOCK_SIZE={args.block_size})")

    # Side-by-side savings summary.
    print("\nSavings summary (lck → peak ratio False/True, time ratio False/True):")
    print(f"{'lck':>4} {'peak_F_MB':>10} {'peak_T_MB':>10} {'mem_x':>7} "
          f"{'time_F_ms':>10} {'time_T_ms':>10} {'time_x':>7}")
    by_lck = {}
    for r in rows:
        by_lck.setdefault((r.B, r.Q_LEN, r.KV_LEN), {})[r.compile_flag] = r
    for (B, Q, KV), pair in by_lck.items():
        f, t = pair[False], pair[True]
        lck = (KV // Q) - 1
        mem_x = f.peak_mb / max(t.peak_mb, 1e-6)
        time_x = f.median_ms / max(t.median_ms, 1e-6)
        print(f"{lck:>4} {f.peak_mb:>10.2f} {t.peak_mb:>10.2f} {mem_x:>7.1f}x "
              f"{f.median_ms:>10.2f} {t.median_ms:>10.2f} {time_x:>7.2f}x")
    return rows


def mode_compile_qlen(args):
    """Q_LEN sweep including the q_len<=128 branch — is the branch still useful?"""
    rows = []
    qs = args.qlens
    for q in qs:
        # Match the production lck behavior: KV_LEN = Q_LEN*(1+lck)
        for cf in [False, True]:
            rows.append(_bench_one(
                label=f"Q={q} _cmp={cf}",
                B=args.B, H_arg=1, Q_LEN=q, lck=args.lck,
                block_size=args.block_size, compile_flag=cf,
                n_warmup=args.warmup, n_iter=args.iters,
            ))
    _print_table(rows, f"Q_LEN × _compile (B={args.B}, H=1, lck={args.lck}, "
                       f"BLOCK_SIZE={args.block_size})")
    return rows


def mode_compile_amortize(args):
    """Per-call timing for the first N calls — quantify the compile-time tax."""
    print("\n=== _compile amortization (per-call time, ms) ===")
    Q, lck, B = args.Q_LEN, args.lck, args.B
    KV_LEN = Q * (1 + lck)
    n = args.amortize_n
    print(f"B={B}, H=1, Q_LEN={Q}, KV_LEN={KV_LEN}, BLOCK_SIZE={args.block_size}, n_calls={n}")
    print(f"{'call#':>6} {'_compile=False ms':>20} {'_compile=True ms':>20} "
          f"{'False peak MB':>15} {'True peak MB':>15}")

    torch.manual_seed(0)
    seq_lengths = torch.randint(Q // 2, Q + 1, (B,), device="cuda", dtype=torch.long)
    mask_mod = generate_eagle3_mask(seq_lengths, Q_LEN=Q, KV_LEN=KV_LEN, lck=lck)

    times_f, times_t = [], []
    peaks_f, peaks_t = [], []
    for cf, times, peaks in [(False, times_f, peaks_f), (True, times_t, peaks_t)]:
        gc.collect()
        torch.cuda.empty_cache()
        for _ in range(n):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            bm = create_block_mask(
                mask_mod, B=B, H=1, Q_LEN=Q, KV_LEN=KV_LEN,
                device="cuda", BLOCK_SIZE=args.block_size, _compile=cf,
            )
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
            peaks.append(torch.cuda.max_memory_allocated() / (1024 * 1024))
            del bm

    for i in range(n):
        print(f"{i:>6} {times_f[i]:>20.2f} {times_t[i]:>20.2f} "
              f"{peaks_f[i]:>15.2f} {peaks_t[i]:>15.2f}")

    # Steady-state median (skip first 2 to drop compile cost).
    if n > 2:
        ss_f = sorted(times_f[2:])[len(times_f[2:]) // 2]
        ss_t = sorted(times_t[2:])[len(times_t[2:]) // 2]
        print(f"\nSteady-state median (calls 3..{n}): "
              f"_compile=False={ss_f:.2f} ms, _compile=True={ss_t:.2f} ms, "
              f"speedup={ss_f / max(ss_t, 1e-6):.2f}x")
        compile_tax_ms = times_t[0] - ss_t
        print(f"First-call compile tax (_compile=True): {compile_tax_ms:.1f} ms "
              f"(amortizes after ~{int(compile_tax_ms / max(ss_f - ss_t, 1e-6))} calls)")


def mode_compile_recompile(args):
    """Cycle through several distinct shape signatures; measure recompile cost."""
    print("\n=== _compile recompile behavior on shape change ===")
    Q = args.Q_LEN
    B = args.B
    print(f"B={B}, H=1, Q_LEN={Q}, BLOCK_SIZE={args.block_size}")
    print(f"{'pass':>5} {'lck':>4} {'KV_LEN':>7} {'time_ms':>9} {'peak_MB':>9}")

    # Two passes: first pass compiles each lck, second pass should hit kernel cache.
    lcks = args.lcks
    seqs = {}
    for lck in lcks:
        seqs[lck] = (Q * (1 + lck))
        torch.manual_seed(lck)
    for pass_i in range(2):
        for lck in lcks:
            KV_LEN = Q * (1 + lck)
            torch.manual_seed(0)
            sl = torch.randint(Q // 2, Q + 1, (B,), device="cuda", dtype=torch.long)
            mm = generate_eagle3_mask(sl, Q_LEN=Q, KV_LEN=KV_LEN, lck=lck)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            bm = create_block_mask(
                mm, B=B, H=1, Q_LEN=Q, KV_LEN=KV_LEN,
                device="cuda", BLOCK_SIZE=args.block_size, _compile=True,
            )
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000
            peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
            print(f"{pass_i + 1:>5} {lck:>4} {KV_LEN:>7} {ms:>9.2f} {peak:>9.2f}")
            del bm


def _one_call_compile_true(B, Q, KV, block_size, device="cuda", seed=0):
    torch.manual_seed(seed)
    sl = torch.randint(Q // 2, Q + 1, (B,), device=device, dtype=torch.long)
    lck = (KV // Q) - 1
    mm = generate_eagle3_mask(sl, Q_LEN=Q, KV_LEN=KV, lck=lck)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    bm = create_block_mask(mm, B=B, H=1, Q_LEN=Q, KV_LEN=KV,
                           device=device, BLOCK_SIZE=block_size, _compile=True)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000
    peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
    del bm, mm
    return ms, peak


def mode_compile_chaos(args):
    """Stress _compile=True with truly dynamic shape changes.

    Three sub-tests:
      A. Shuffle the 7 TTT lck shapes across 3 passes (out-of-order replay).
      B. 30 random (B, Q_LEN, KV_LEN) tuples drawn from a wide grid.
      C. Same shape but cycle BLOCK_SIZE in {64, 128, 256}.
    """
    import random

    # ---- Sub-test A: shuffled lck order, 3 passes ----
    print("\n=== chaos A: shuffled lck order, 3 passes (B=1, Q=2048, BLOCK=128) ===")
    print(f"{'pass':>5} {'idx':>4} {'lck':>4} {'KV_LEN':>7} {'time_ms':>9} {'peak_MB':>9}")
    Q = args.Q_LEN
    rng = random.Random(0)
    cold_ms_total = 0.0
    pass_totals = []
    for pass_i in range(3):
        order = list(args.lcks)
        rng.shuffle(order)
        pass_ms = 0.0
        for idx, lck in enumerate(order):
            KV = Q * (1 + lck)
            ms, peak = _one_call_compile_true(args.B, Q, KV, args.block_size)
            print(f"{pass_i + 1:>5} {idx:>4} {lck:>4} {KV:>7} {ms:>9.2f} {peak:>9.2f}")
            pass_ms += ms
            if pass_i == 0:
                cold_ms_total += ms
        pass_totals.append(pass_ms)
    print(f"\nPass totals (ms): {[f'{p:.1f}' for p in pass_totals]}  "
          f"-> {pass_totals[0] / max(pass_totals[-1], 1e-6):.1f}x slower on cold pass")

    # ---- Sub-test B: 30 random shapes ----
    print("\n=== chaos B: 30 random (B, Q_LEN, KV_LEN) tuples ===")
    print(f"{'idx':>4} {'B':>3} {'Q_LEN':>6} {'KV_LEN':>7} {'time_ms':>9} {'peak_MB':>9}")
    rng = random.Random(42)
    # Q_LEN must be a multiple of BLOCK_SIZE (128).
    qs = [128 * k for k in range(2, 33)]   # 256 .. 4096
    Bs = [1, 2, 4, 8]
    multipliers = [1, 2, 3, 4, 5, 6, 7]
    seen = set()
    samples = []
    while len(samples) < 30:
        B = rng.choice(Bs)
        q = rng.choice(qs)
        m = rng.choice(multipliers)
        kv = q * m
        if (B, q, kv) in seen:
            continue
        seen.add((B, q, kv))
        samples.append((B, q, kv))
    times_b = []
    for i, (B, q, kv) in enumerate(samples):
        ms, peak = _one_call_compile_true(B, q, kv, args.block_size)
        times_b.append(ms)
        print(f"{i:>4} {B:>3} {q:>6} {kv:>7} {ms:>9.2f} {peak:>9.2f}")
    times_sorted = sorted(times_b)
    n_slow = sum(1 for x in times_b if x > 50)  # arbitrary "compile happened" threshold
    print(f"\n30 random shapes: median={times_sorted[15]:.2f} ms, "
          f"p90={times_sorted[27]:.2f} ms, max={max(times_b):.2f} ms, "
          f"slow (>50ms) calls={n_slow}/30")

    # ---- Sub-test C: cycle BLOCK_SIZE on the same shape ----
    print("\n=== chaos C: same shape (B=1, Q=2048, KV=14336), cycling BLOCK_SIZE ===")
    print(f"{'idx':>4} {'BLOCK':>6} {'time_ms':>9} {'peak_MB':>9}")
    block_cycle = [64, 128, 256, 64, 128, 256, 64, 128, 256]
    for i, bs in enumerate(block_cycle):
        ms, peak = _one_call_compile_true(1, 2048, 14336, bs)
        print(f"{i:>4} {bs:>6} {ms:>9.2f} {peak:>9.2f}")


def mode_compile_alt(args):
    """Compare `_compile=True` (deprecated) vs `torch.compile(create_block_mask)`.

    PyTorch 2.10 deprecates the `_compile=True` flag and recommends
    `torch.compile(create_block_mask)`. We test whether the recommendation is
    equivalent in (a) steady-state cost, (b) recompile behavior on shape change,
    and (c) recompile behavior when the mask_mod closure object is rebuilt
    (different seq_lengths tensor) on every call.
    """
    Q = args.Q_LEN
    bs = args.block_size
    B = args.B
    lcks = args.lcks

    # Cached singleton (matches the WrappedFlexAttention pattern in flex_attention.py).
    compiled_cbm = torch.compile(create_block_mask)

    def _run_flag(mm, B_, Q_, KV_):
        return create_block_mask(mm, B=B_, H=1, Q_LEN=Q_, KV_LEN=KV_,
                                  device="cuda", BLOCK_SIZE=bs, _compile=True)

    def _run_torchcompile(mm, B_, Q_, KV_):
        return compiled_cbm(mm, B=B_, H=1, Q_LEN=Q_, KV_LEN=KV_,
                             device="cuda", BLOCK_SIZE=bs)

    def _make_mm(seed, lck):
        torch.manual_seed(seed)
        sl = torch.randint(Q // 2, Q + 1, (B,), device="cuda", dtype=torch.long)
        KV = Q * (1 + lck)
        return generate_eagle3_mask(sl, Q_LEN=Q, KV_LEN=KV, lck=lck), KV

    def _measure(fn, mm, KV):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        bm = fn(mm, B, Q, KV)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
        del bm
        return ms, peak

    # ---- A: steady-state, same shape, 6 calls ----
    print("\n=== compile_alt A: steady-state same shape (B=%d, Q=%d, lck=6) ===" % (B, Q))
    print(f"{'call#':>5}  {'_compile=T ms':>15}  {'peak MB':>8} | "
          f"{'torch.compile ms':>17}  {'peak MB':>8}")
    mm, KV = _make_mm(0, 6)
    for i in range(6):
        ms_f, peak_f = _measure(_run_flag, mm, KV)
        ms_tc, peak_tc = _measure(_run_torchcompile, mm, KV)
        print(f"{i:>5}  {ms_f:>15.2f}  {peak_f:>8.2f} | {ms_tc:>17.2f}  {peak_tc:>8.2f}")

    # ---- B: 7 distinct shapes, single pass ----
    print("\n=== compile_alt B: 7 distinct lck shapes, single pass ===")
    print(f"{'lck':>4}  {'KV_LEN':>7}  {'_compile=T ms':>15}  {'peak MB':>8} | "
          f"{'torch.compile ms':>17}  {'peak MB':>8}")
    for lck in lcks:
        mm, KV = _make_mm(lck, lck)
        ms_f, peak_f = _measure(_run_flag, mm, KV)
        ms_tc, peak_tc = _measure(_run_torchcompile, mm, KV)
        print(f"{lck:>4}  {KV:>7}  {ms_f:>15.2f}  {peak_f:>8.2f} | {ms_tc:>17.2f}  {peak_tc:>8.2f}")

    # ---- C: closure churn — same shape, different seq_lengths tensor every call ----
    print("\n=== compile_alt C: closure churn (same shape, fresh seq_lengths each call) ===")
    print("This stresses whether torch.compile keys on the mask_mod closure object.")
    print(f"{'call#':>5}  {'_compile=T ms':>15}  {'peak MB':>8} | "
          f"{'torch.compile ms':>17}  {'peak MB':>8}")
    KV = Q * 7
    for i in range(6):
        # Fresh seq_lengths -> fresh mask_mod closure for each call.
        torch.manual_seed(100 + i)
        sl = torch.randint(Q // 2, Q + 1, (B,), device="cuda", dtype=torch.long)
        mm = generate_eagle3_mask(sl, Q_LEN=Q, KV_LEN=KV, lck=6)
        ms_f, peak_f = _measure(_run_flag, mm, KV)
        # Fresh closure for the torch.compile path too.
        torch.manual_seed(200 + i)
        sl2 = torch.randint(Q // 2, Q + 1, (B,), device="cuda", dtype=torch.long)
        mm2 = generate_eagle3_mask(sl2, Q_LEN=Q, KV_LEN=KV, lck=6)
        ms_tc, peak_tc = _measure(_run_torchcompile, mm2, KV)
        print(f"{i:>5}  {ms_f:>15.2f}  {peak_f:>8.2f} | {ms_tc:>17.2f}  {peak_tc:>8.2f}")


def mode_holistic(args):
    """Holistic memory picture: B × lck × _compile, plus full-TTT-loop peak.

    Sub-test 1: 2D grid of single-call peak across (B, lck, _compile).
    Sub-test 2: simulate one full TTT loop (lck=0..6 sequentially in one process)
                without resetting peak — captures the real per-layer peak the
                attention forward incurs for one microbatch.
    """
    Bs = args.batches
    lcks = args.lcks
    Q = args.Q_LEN
    bs = args.block_size

    # ---- 1: per-call grid ----
    print("\n=== holistic 1: per-call peak MB across B × lck × _compile "
          f"(Q={Q}, BLOCK={bs}) ===")
    cells: dict[tuple[int, int, bool], float] = {}
    times: dict[tuple[int, int, bool], float] = {}
    for B in Bs:
        for lck in lcks:
            for cf in [False, True]:
                r = _bench_one(
                    label=f"B={B} lck={lck} cmp={cf}",
                    B=B, H_arg=1, Q_LEN=Q, lck=lck,
                    block_size=bs, compile_flag=cf,
                    n_warmup=1, n_iter=2,
                )
                cells[(B, lck, cf)] = r.peak_mb
                times[(B, lck, cf)] = r.median_ms

    # Print as two pivot tables side by side.
    def _print_pivot(cf, label):
        print(f"\n{label}  (peak MB)")
        hdr = "B \\ lck   " + "  ".join(f"lck={l:>2}" for l in lcks)
        print(hdr)
        for B in Bs:
            row = [f"B={B:<3}    "]
            for lck in lcks:
                row.append(f"{cells[(B, lck, cf)]:>7.2f}")
            print("  ".join(row))
    _print_pivot(False, "_compile=False")
    _print_pivot(True, "_compile=True")

    # Savings ratio table
    print("\nSavings ratio (False / True):")
    print("B \\ lck   " + "  ".join(f"lck={l:>2}" for l in lcks))
    for B in Bs:
        row = [f"B={B:<3}    "]
        for lck in lcks:
            f = cells[(B, lck, False)]
            t = max(cells[(B, lck, True)], 1e-6)
            row.append(f"{f / t:>6.0f}x")
        print("  ".join(row))

    # ---- 2: full-TTT-loop peak ----
    print("\n=== holistic 2: peak MB across one full TTT loop (lck=0..6 sequential) ===")
    print(f"{'B':>3}  {'_cmp=F peak MB':>15}  {'_cmp=T peak MB':>15}  "
          f"{'_cmp=F sum_traffic':>18}  {'_cmp=T sum_traffic':>18}  {'headroom_freed_MB':>18}")
    for B in Bs:
        for cf in [False, True]:
            torch.manual_seed(0)
            sl = torch.randint(Q // 2, Q + 1, (B,), device="cuda", dtype=torch.long)
            # warm up the compile cache outside the measured window
            for lck in lcks:
                KV = Q * (1 + lck)
                mm = generate_eagle3_mask(sl, Q_LEN=Q, KV_LEN=KV, lck=lck)
                bm = create_block_mask(mm, B=B, H=1, Q_LEN=Q, KV_LEN=KV,
                                        device="cuda", BLOCK_SIZE=bs, _compile=cf)
                del bm, mm
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            # measured: full TTT loop
            traffic = 0.0
            for lck in lcks:
                KV = Q * (1 + lck)
                mm = generate_eagle3_mask(sl, Q_LEN=Q, KV_LEN=KV, lck=lck)
                bm = create_block_mask(mm, B=B, H=1, Q_LEN=Q, KV_LEN=KV,
                                        device="cuda", BLOCK_SIZE=bs, _compile=cf)
                traffic += cells[(B, lck, cf)]  # approximate: sum of per-call peaks
                del bm, mm
            torch.cuda.synchronize()
            loop_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
            if cf is False:
                f_loop_peak = loop_peak
                f_traffic = traffic
            else:
                t_loop_peak = loop_peak
                t_traffic = traffic
        headroom = f_loop_peak - t_loop_peak
        print(f"{B:>3}  {f_loop_peak:>15.2f}  {t_loop_peak:>15.2f}  "
              f"{f_traffic:>18.2f}  {t_traffic:>18.2f}  {headroom:>18.2f}")

    # ---- 3: per-step extrapolation ----
    print("\n=== holistic 3: per-training-step extrapolation ===")
    print("Per-step cost = N_drafter_layers × full TTT loop peak (one microbatch).")
    print("EAGLE3 drafter typically has 1 transformer block, but the call also")
    print("happens for the verifier-side feature concat — we model 1 and 4 layers.")
    for n_layers in [1, 4]:
        print(f"\nN_drafter_layers = {n_layers}:")
        print(f"{'B':>3}  {'_cmp=F MB':>10}  {'_cmp=T MB':>10}  {'savings_MB':>11}")
        for B in Bs:
            f = max(cells[(B, lck, False)] for lck in lcks) * n_layers
            t = max(cells[(B, lck, True)] for lck in lcks) * n_layers
            print(f"{B:>3}  {f:>10.2f}  {t:>10.2f}  {f - t:>11.2f}")


def mode_all(args):
    print("\n>>> Running full diagnostic suite\n")
    mode_lck(args)
    mode_batch(args)
    mode_head(args)
    mode_block(args)
    mode_compile(args)
    mode_compile_grid(args)
    mode_compile_qlen(args)
    mode_compile_amortize(args)
    mode_compile_recompile(args)
    mode_compile_chaos(args)
    mode_holistic(args)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["lck", "batch", "head", "block", "compile",
                                       "compile_grid", "compile_qlen",
                                       "compile_amortize", "compile_recompile",
                                       "compile_chaos", "compile_alt",
                                       "holistic", "all"],
                   default="all")
    p.add_argument("--B", type=int, default=1)
    p.add_argument("--Q_LEN", type=int, default=2048)
    p.add_argument("--lck", type=int, default=6, help="TTT step index; KV_LEN = Q_LEN*(1+lck)")
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--compile", action="store_true", help="pass _compile=True")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--lcks", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6])
    p.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--block-sizes", type=int, nargs="+", default=[64, 128, 256, 512])
    p.add_argument("--qlens", type=int, nargs="+",
                   default=[64, 128, 256, 512, 1024, 2048, 4096])
    p.add_argument("--amortize-n", type=int, default=10,
                   help="number of consecutive calls in compile_amortize mode")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")
    print(f"torch {torch.__version__}  device {torch.cuda.get_device_name(0)}")

    {"lck": mode_lck, "batch": mode_batch, "head": mode_head,
     "block": mode_block, "compile": mode_compile,
     "compile_grid": mode_compile_grid, "compile_qlen": mode_compile_qlen,
     "compile_amortize": mode_compile_amortize,
     "compile_recompile": mode_compile_recompile,
     "compile_chaos": mode_compile_chaos,
     "compile_alt": mode_compile_alt,
     "holistic": mode_holistic,
     "all": mode_all}[args.mode](args)


if __name__ == "__main__":
    main()
