#!/usr/bin/env python3
"""Per-step inference latency harness (M4, MLX).

StepShield's native timing metric is EIR (step counts), NOT wall-clock — this
harness supplies the separate ms-latency requirement. It drives the SAME
NemotronGuard code path used for the accuracy run (incremental KV cache,
prefill-only + 2-token readout), over real holdout step streams.

Modes reported:
  warm  — deployment pattern: per-trajectory KV cache resident; per-step cost =
          tokenize + prefill(new step tokens [+2 pending verdict tokens]) + readout.
  header — one-time per-trajectory header prefill (session setup).
  cold  — worst case: full re-prefill of header+history+step for a single
          decision (no cache reuse), sampled across trajectory positions.

Usage: python latency_bench.py [--model models/nemotron-mini-4b-q4]
       [--adapter adapters/] [--n-traj 40] [--json results/latency.json]
"""
import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx

BASE = Path(__file__).parent


def pct(sorted_v, q):
    return sorted_v[min(len(sorted_v) - 1, int(len(sorted_v) * q))]


def summarize(name, vals):
    v = sorted(vals)
    s = {"n": len(v), "p50": pct(v, 0.50), "p90": pct(v, 0.90),
         "p95": pct(v, 0.95), "p99": pct(v, 0.99),
         "mean": statistics.mean(v), "max": max(v)}
    print(f"{name:8s} n={s['n']:5d} p50={s['p50']:7.1f} p90={s['p90']:7.1f} "
          f"p95={s['p95']:7.1f} p99={s['p99']:7.1f} mean={s['mean']:7.1f} "
          f"max={s['max']:7.1f} (ms)")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/nemotron-mini-4b-q4")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--n-traj", type=int, default=40)
    ap.add_argument("--cold-samples", type=int, default=60)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(BASE))
    from nemotron_guard import NemotronGuard, render_header, render_step
    from run_holdout import load_holdout

    trajs = load_holdout(args.n_traj)
    print(f"Loaded {len(trajs)} holdout trajectories for latency measurement")

    det = NemotronGuard(model_path=str(BASE / args.model),
                        adapter_path=args.adapter, threshold=0.9)

    # ---- warm per-step + header ----
    step_lat, tok_counts = [], []
    det.header_latencies_ms = []
    for traj in trajs:
        asyncio.run(det.detect_trajectory(traj))
        step_lat.extend(det.last_step_latencies_ms)
        for s in traj["steps"]:
            tok_counts.append(len(det.tok.encode(render_step(s),
                                                 add_special_tokens=False)))
    warm = summarize("warm", step_lat)
    header = summarize("header", det.header_latencies_ms)
    tv = sorted(tok_counts)
    print(f"step tokens: p50={pct(tv,0.5)} p95={pct(tv,0.95)} max={max(tv)}")

    # ---- cold: single decision with full re-prefill ----
    from mlx_lm.models.cache import make_prompt_cache
    cold_lat = []
    k = 0
    for traj in trajs:
        steps = traj["steps"]
        for frac in (0.33, 0.66, 1.0):
            if k >= args.cold_samples:
                break
            idx = max(0, min(len(steps) - 1, int(len(steps) * frac) - 1))
            ids = det.tok.encode(render_header(traj.get("task", {})),
                                 add_special_tokens=False)
            for s in steps[:idx]:
                ids += det.tok.encode(render_step(s), add_special_tokens=False)
                ids += [det.ok_id] + det._nl
            ids += det.tok.encode(render_step(steps[idx]), add_special_tokens=False)
            t0 = time.perf_counter()
            det.cache = make_prompt_cache(det.model)
            last = det._feed(ids)
            two = last[det._sel]
            p = mx.softmax(two.astype(mx.float32), axis=-1)
            mx.eval(p)
            cold_lat.append((time.perf_counter() - t0) * 1000)
            k += 1
        if k >= args.cold_samples:
            break
    cold = summarize("cold", cold_lat)

    out = {"model": args.model, "adapter": args.adapter,
           "hardware": "Apple M4 Pro 48GB (MacBook Pro), MLX",
           "warm_per_step_ms": warm, "header_prefill_ms": header,
           "cold_full_prefill_ms": cold,
           "step_tokens": {"p50": pct(tv, 0.5), "p95": pct(tv, 0.95), "max": max(tv)}}
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(args.json, "w"), indent=2)
        print(f"saved {args.json}")


if __name__ == "__main__":
    main()
