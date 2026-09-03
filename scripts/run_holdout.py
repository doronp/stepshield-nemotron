#!/usr/bin/env python3
"""Generic StepShield holdout runner: merges scrubbed trajectories with the answer key
(same join as stepshield/scripts/run_hybridguard_holdout.py) and evaluates a detector
with the repo's own calculate_metrics. Adds wall-clock per-step latency capture.

Usage:
  python run_holdout.py --detector static|constraint|nemotron [--max N] [--output results.json]
"""
import argparse
import asyncio
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent  # repo root (script lives in scripts/)
REPO = BASE / "stepshield"
sys.path.insert(0, str(REPO / "benchmark"))

from metrics.timing_metrics import (  # noqa: E402
    TrajectoryResult, calculate_metrics, calculate_intervention_gap,
    is_early_detection, format_metrics_table,
)

SCRUBBED = REPO / "data" / "test_holdout" / "scrubbed"
ANSWER_KEY = REPO / "data" / "test_holdout" / "mapping" / "answer_key.jsonl"


def _import_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_holdout(max_n=None):
    answer_key = {}
    with open(ANSWER_KEY) as f:
        for line in f:
            e = json.loads(line)
            answer_key[e["id"]] = e
    trajectories = []
    for tf in sorted(SCRUBBED.glob("*.jsonl")):
        ak = answer_key.get(tf.stem)
        if not ak:
            continue
        traj = json.load(open(tf))
        trajectories.append({
            "trajectory_id": tf.stem,
            "trajectory_type": ak["trajectory_type"],
            "category": ak.get("category"),
            "severity": ak.get("severity"),
            "rogue_step": ak.get("rogue_step"),
            "steps": traj.get("steps", []),
            "task": traj.get("task", {}),
        })
        if max_n and len(trajectories) >= max_n:
            break
    return trajectories


def make_detector(name, args):
    if name == "static":
        from detectors.static_guard import StaticGuard
        return StaticGuard()
    if name == "constraint":
        from detectors.constraint_guard import ConstraintGuard
        return ConstraintGuard()
    if name == "nemotron":
        m = _import_module("nemotron_guard", Path(__file__).resolve().parent / "nemotron_guard.py")
        return m.NemotronGuard(model_path=args.model_path,
                               adapter_path=args.adapter,
                               threshold=args.threshold)
    raise ValueError(f"unknown detector {name}")


async def run(detector, trajectories):
    results = []
    step_latencies = []
    for i, traj in enumerate(trajectories):
        # per-step latency: wrap detect_step via detect_trajectory timing per step
        t0 = time.time()
        det = await detector.detect_trajectory(traj)
        if hasattr(detector, "last_step_latencies_ms"):
            step_latencies.extend(detector.last_step_latencies_ms)
        gap = calculate_intervention_gap(det.detection_step, det.ground_truth_step)
        early = is_early_detection(det.detection_step, det.total_steps,
                                   ground_truth_step=det.ground_truth_step, k=3)
        results.append(TrajectoryResult(
            trajectory_id=det.trajectory_id,
            trajectory_type=traj.get("trajectory_type", "unknown"),
            category=det.category or "UNK",
            total_steps=det.total_steps,
            ground_truth_step=det.ground_truth_step,
            detection_step=det.detection_step,
            detected=det.detected,
            intervention_gap=gap,
            is_early=early,
        ))
        el = time.time() - t0
        print(f"\r[{i+1}/{len(trajectories)}] {det.trajectory_id} "
              f"({traj['trajectory_type']}) det_step={det.detection_step} {el*1000:.0f}ms   ",
              end="", flush=True)
    print()
    return results, step_latencies


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", required=True)
    ap.add_argument("--max", type=int, default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    trajectories = load_holdout(args.max)
    print(f"Loaded {len(trajectories)} holdout trajectories "
          f"({sum(1 for t in trajectories if t['trajectory_type']=='rogue')} rogue)")

    detector = make_detector(args.detector, args)
    results, step_lat = asyncio.run(run(detector, trajectories))
    metrics = calculate_metrics(results)
    print(format_metrics_table(metrics, args.detector))

    out = {
        "detector": args.detector,
        "n_trajectories": len(results),
        "metrics": {
            "accuracy": metrics.accuracy, "precision": metrics.precision,
            "recall": metrics.recall, "f1": metrics.f1,
            "fpr": metrics.false_positives / max(1, metrics.false_positives + metrics.true_negatives),
            "eir_1": metrics.eir_1, "eir_3": metrics.eir_3, "eir_5": metrics.eir_5,
            "mean_gap": metrics.mean_intervention_gap,
            "tokens_saved": metrics.tokens_saved_mean,
        },
        "confusion": {"tp": metrics.true_positives, "fp": metrics.false_positives,
                      "tn": metrics.true_negatives, "fn": metrics.false_negatives},
        "trajectory_results": [vars(r) for r in results],
    }
    if step_lat:
        s = sorted(step_lat)
        out["step_latency_ms"] = {
            "n": len(s),
            "p50": s[len(s)//2],
            "p95": s[int(len(s)*0.95)],
            "p99": s[int(len(s)*0.99)],
            "mean": statistics.mean(s),
            "max": max(s),
        }
        print(f"Per-step latency ms: p50={out['step_latency_ms']['p50']:.1f} "
              f"p95={out['step_latency_ms']['p95']:.1f} mean={out['step_latency_ms']['mean']:.1f}")
    fpr = out["metrics"]["fpr"]
    print(f"FPR: {fpr*100:.1f}%")
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(args.output, "w"), indent=2)
        print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
