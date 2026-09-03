#!/usr/bin/env python3
"""Prepare StepShield training data for the incremental step classifier.

Output: JSONL, one record per trajectory:
  {"id": ..., "source": "train|benign_gen", "header": str,
   "steps": [{"text": str, "label": "OK"|"BAD"|null}]}   # null = mask (no loss)

Split policy:
  - train/ pairs (ROGUE+CLEAN share scenario key e.g. DEC-L1-001) stay in the same split.
  - 90/10 pair split by deterministic hash.
  - generated_benign: N_TRAIN_BENIGN to train, N_VAL_BENIGN to val (threshold tuning), rest unused.
Holdout 216 never touched here.
"""
import hashlib
import json
import random
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent  # repo root (script lives in scripts/)
REPO = BASE / "stepshield"
OUT = BASE / "data"
OUT.mkdir(exist_ok=True)

N_TRAIN_BENIGN = 400
N_VAL_BENIGN = 300
BACKUP_WINDOW = True  # v2b: set False (or env NOBACKUP=1) to drop +3-window BAD labels
import os
if os.environ.get("NOBACKUP") == "1":
    BACKUP_WINDOW = False

MAX_FIELD = {"thought": 400, "args": 500, "obs": 700}


def _trunc(s, n):
    s = s if isinstance(s, str) else json.dumps(s, ensure_ascii=False)
    return s if len(s) <= n else s[:n] + "…"


def render_header(task):
    parts = [f"Task: {task.get('title', '')}", task.get("description", "").strip()]
    cons = task.get("constraints") or []
    if cons:
        parts.append("Constraints: " + "; ".join(cons))
    return "\n".join(p for p in parts if p) + "\n"


def render_step(step):
    n = step.get("step")
    thought = _trunc(step.get("thought") or "", MAX_FIELD["thought"])
    action = step.get("action") or ""
    args = _trunc(step.get("arguments") or {}, MAX_FIELD["args"])
    obs = _trunc(step.get("observation") or "", MAX_FIELD["obs"])
    return f"<<S{n}>>\nT: {thought}\nA: {action} {args}\nO: {obs}\nV:"


def convert_traj(traj, is_rogue, ok_weight=1.0):
    """Kimi checkpoint-1 adjustments:
    - post-divergence unlabeled steps within +3 of first labeled rogue step -> BAD @ w=0.5
      (backup detections still inside the EIR_3 window); beyond +3 -> masked.
    - gen-benign OK steps can be downweighted via ok_weight (provenance confound).
    """
    steps_out = []
    first_rogue_idx = None
    for i, s in enumerate(traj["steps"]):
        if is_rogue and s.get("label"):
            lab, w = "BAD", 1.0
            if first_rogue_idx is None:
                first_rogue_idx = i
        elif is_rogue and first_rogue_idx is not None:
            if BACKUP_WINDOW and i - first_rogue_idx <= 3:
                lab, w = "BAD", 0.5
            else:
                lab, w = None, 0.0
        else:
            lab, w = "OK", ok_weight
        steps_out.append({"text": render_step(s), "label": lab, "w": w})
    return {"header": render_header(traj.get("task", {})), "steps": steps_out}


def pair_key(path):
    # DEC-L1-001-ROGUE.jsonl -> DEC-L1-001
    return path.stem.rsplit("-", 1)[0]


def split_of(key):
    h = int(hashlib.sha256(key.encode()).hexdigest(), 16) % 100
    return "val" if h < 10 else "train"


def main():
    recs = {"train": [], "val": []}
    stats = {"train": {"OK": 0, "BAD": 0, "mask": 0}, "val": {"OK": 0, "BAD": 0, "mask": 0}}

    files = sorted((REPO / "data" / "train").glob("*/*.jsonl"))
    for p in files:
        traj = json.load(open(p))
        is_rogue = "ROGUE" in p.stem
        rec = convert_traj(traj, is_rogue)
        rec["id"] = p.stem
        rec["source"] = "train"
        rec["is_rogue_traj"] = is_rogue
        sp = split_of(pair_key(p))
        recs[sp].append(rec)
        for s in rec["steps"]:
            stats[sp][s["label"] or "mask"] += 1

    ben = sorted((REPO / "data" / "generated_benign").glob("*.jsonl"))
    rng = random.Random(42)
    rng.shuffle(ben)
    for i, p in enumerate(ben[: N_TRAIN_BENIGN + N_VAL_BENIGN]):
        traj = json.load(open(p))
        rec = convert_traj(traj, is_rogue=False, ok_weight=0.5)
        rec["id"] = p.stem
        rec["source"] = "benign_gen"
        rec["is_rogue_traj"] = False
        sp = "train" if i < N_TRAIN_BENIGN else "val"
        recs[sp].append(rec)
        for s in rec["steps"]:
            stats[sp][s["label"] or "mask"] += 1

    for sp in ("train", "val"):
        rng.shuffle(recs[sp])
        with open(OUT / f"{sp}.jsonl", "w") as f:
            for r in recs[sp]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        n_rogue = sum(1 for r in recs[sp] if r["is_rogue_traj"])
        print(f"{sp}: {len(recs[sp])} trajs ({n_rogue} rogue) | step labels {stats[sp]}")


if __name__ == "__main__":
    main()
