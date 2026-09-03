#!/usr/bin/env python3
"""Per-step latency on GPU (torch, incremental KV cache, bf16 merged model).

Same protocol as the M4 bench: per-trajectory KV cache; per step measure
tokenize + forward(new step tokens [+2 pending verdict tokens]) + 2-token
softmax readout. lm_head applied only at the final position.

Usage: python3 gpu_latency_bench.py --model merged --data data/val.jsonl --n-traj 30
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache


def pct(v, q):
    return v[min(len(v) - 1, int(len(v) * q))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="merged")
    ap.add_argument("--data", default="data/val.jsonl")
    ap.add_argument("--n-traj", type=int, default=30)
    ap.add_argument("--json", default="gpu_latency.json")
    args = ap.parse_args()

    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(device)
    model.eval()
    ok_id = tok.encode(" OK", add_special_tokens=False)[0]
    bad_id = tok.encode(" BAD", add_special_tokens=False)[0]
    nl = tok.encode("\n", add_special_tokens=False)
    gpu_name = torch.cuda.get_device_name(0)
    print("GPU:", gpu_name)

    recs = [json.loads(l) for l in open(args.data)][: args.n_traj]

    @torch.no_grad()
    def feed(ids, cache):
        t = torch.tensor([ids], device=device)
        out = model.model(input_ids=t, past_key_values=cache, use_cache=True)
        h = out.last_hidden_state[:, -1:, :]
        return model.lm_head(h)[0, -1]

    # warmup
    c = DynamicCache()
    feed(tok.encode("warmup check", add_special_tokens=False), c)
    torch.cuda.synchronize()

    step_lat, header_lat, raw = [], [], []
    for rec in recs:
        cache = DynamicCache()
        hist = 0
        t0 = time.perf_counter()
        hids = tok.encode(rec["header"], add_special_tokens=False)
        feed(hids, cache)
        torch.cuda.synchronize()
        header_lat.append((time.perf_counter() - t0) * 1000)
        hist += len(hids)
        pending = []
        for s in rec["steps"]:
            t0 = time.perf_counter()
            ids = pending + tok.encode(s["text"], add_special_tokens=False)
            logits = feed(ids, cache)
            two = logits[[ok_id, bad_id]].float()
            p_bad = torch.softmax(two, dim=-1)[1]
            torch.cuda.synchronize()
            p = float(p_bad)
            ms = (time.perf_counter() - t0) * 1000
            step_lat.append(ms)
            raw.append({"new_tokens": len(ids), "history": hist, "ms": ms})
            hist += len(ids)
            pending = [ok_id] + nl

    def summ(name, vals):
        v = sorted(vals)
        s = {"n": len(v), "p50": pct(v, .5), "p90": pct(v, .9), "p95": pct(v, .95),
             "p99": pct(v, .99), "mean": statistics.mean(v), "max": max(v)}
        print(f"{name}: " + " ".join(f"{k}={x:.1f}" if isinstance(x, float) else f"{k}={x}"
                                     for k, x in s.items()))
        return s

    import numpy as np
    X = np.array([[1, r["new_tokens"], r["history"]] for r in raw], float)
    y = np.array([r["ms"] for r in raw])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2 = 1 - ((y - X @ coef) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print(f"fit: {coef[0]:.1f}ms + {coef[1]:.4f}ms/new_token + "
          f"{coef[2]:.6f}ms/history_token (R2={r2:.3f})")
    out = {"gpu": gpu_name, "model": args.model,
           "warm_per_step_ms": summ("warm", step_lat),
           "header_prefill_ms": summ("header", header_lat),
           "fit": {"intercept_ms": coef[0], "ms_per_new_token": coef[1],
                   "ms_per_history_token": coef[2], "r2": r2},
           "raw": raw}
    json.dump(out, open(args.json, "w"), indent=2)
    print("saved", args.json)


if __name__ == "__main__":
    main()
