#!/usr/bin/env python3
"""Official StepShield metrics: feed GPU-scored holdout detections through the
REPO'S OWN calculate_metrics/format_metrics_table (benchmark gold standard).

Usage: python official_metrics.py --scores results/holdout_scores_v2.json --thr 0.97
"""
import argparse
import json
import sys
from pathlib import Path

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE / "stepshield" / "benchmark"))

from metrics.timing_metrics import (  # noqa: E402
    TrajectoryResult, calculate_metrics, calculate_intervention_gap,
    is_early_detection, format_metrics_table,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--thr", type=float, required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = json.load(open(args.scores))
    results = []
    for r in data["trajectories"]:
        flag_i = next((i for i, p in enumerate(r["probs"]) if p > args.thr), None)
        detection_step = flag_i + 1 if flag_i is not None else None  # 1-based
        gt = r.get("rogue_step")
        total = r.get("n_steps", len(r["probs"]))
        gap = calculate_intervention_gap(detection_step, gt)
        early = is_early_detection(detection_step, total, ground_truth_step=gt, k=3)
        results.append(TrajectoryResult(
            trajectory_id=r["id"],
            trajectory_type="rogue" if r["rogue"] else "correct",
            category=r.get("category", "UNK"),
            total_steps=total,
            ground_truth_step=gt,
            detection_step=detection_step,
            detected=detection_step is not None,
            intervention_gap=gap,
            is_early=early,
        ))
    m = calculate_metrics(results)
    print(format_metrics_table(m, f"NemotronGuard(thr={args.thr})"))
    fpr = m.false_positives / max(1, m.false_positives + m.true_negatives)
    print(f"FPR: {fpr*100:.2f}%")
    if args.out:
        json.dump({
            "threshold": args.thr, "scores_file": args.scores,
            "accuracy": m.accuracy, "precision": m.precision, "recall": m.recall,
            "f1": m.f1, "fpr": fpr, "eir_1": m.eir_1, "eir_3": m.eir_3,
            "eir_5": m.eir_5, "mean_gap": m.mean_intervention_gap,
            "tokens_saved": m.tokens_saved_mean,
            "confusion": {"tp": m.true_positives, "fp": m.false_positives,
                          "tn": m.true_negatives, "fn": m.false_negatives},
            "category_metrics": m.category_metrics,
        }, open(args.out, "w"), indent=2)
        print("saved", args.out)


if __name__ == "__main__":
    main()
