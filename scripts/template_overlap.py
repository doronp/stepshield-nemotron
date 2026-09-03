#!/usr/bin/env python3
"""Recompute the train/holdout template-overlap measurement from public StepShield data.

No model, no weights, no GPU: this reads only the benchmark's own JSONL files and
recomputes the two numbers the release reports as its benchmark-level finding.

  (A) Task-header reuse.   For each of the 216 holdout trajectories, word-set Jaccard
      between its task `title + description` and that of every train-split trajectory;
      keep the maximum (its nearest train scenario). Report the median over holdout.

  (B) Step-stream novelty.  For each holdout trajectory, restrict to the train
      trajectories that tie for that maximum header Jaccard -- i.e. the *same template* --
      render both step streams exactly as the training pipeline renders them
      (mirrors `render_step` in scripts/prep_data.py) and take the maximum
      `difflib.SequenceMatcher(...).ratio()`. Report the median over holdout.

  (C) Exact step reuse.    How many rendered holdout steps appear verbatim anywhere in
      the train split.

Because (B) has no single canonical definition, the report also prints the same
statistic under three other reasonable choices (mean over same-template candidates,
minimum over them, and a seeded random train trajectory as a floor) so the headline
number can be read against its own sensitivity rather than taken on trust.

Requires the StepShield benchmark checked out (by default as `stepshield/` inside this
repo, matching the rest of the release). Standard library only.

    python scripts/template_overlap.py
    python scripts/template_overlap.py /path/to/stepshield/data --json results/template_overlap.json

Runtime: a few seconds; fully deterministic.
"""
import argparse
import difflib
import json
import random
import re
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent  # repo root (script lives in scripts/)
DEFAULT_DATA = REPO / "stepshield" / "data"

# Field truncation limits: identical to scripts/prep_data.py, so the strings compared
# here are the strings the model was actually trained/scored on.
MAX_FIELD = {"thought": 400, "args": 500, "obs": 700}

WORD_RE = re.compile(r"[a-z0-9]+")


def _trunc(s, n):
    s = s if isinstance(s, str) else json.dumps(s, ensure_ascii=False)
    return s if len(s) <= n else s[:n] + "…"


def render_step(step):
    """Mirror of render_step() in scripts/prep_data.py."""
    n = step.get("step")
    thought = _trunc(step.get("thought") or "", MAX_FIELD["thought"])
    action = step.get("action") or ""
    args = _trunc(step.get("arguments") or {}, MAX_FIELD["args"])
    obs = _trunc(step.get("observation") or "", MAX_FIELD["obs"])
    return f"<<S{n}>>\nT: {thought}\nA: {action} {args}\nO: {obs}\nV:"


def render_stream(traj):
    return "\n".join(render_step(s) for s in traj.get("steps", []))


def header_text(traj):
    task = traj.get("task", {}) or {}
    return (task.get("title", "") + " " + task.get("description", "")).strip()


def word_set(s):
    return set(WORD_RE.findall(s.lower()))


def jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def load_split(paths):
    out = []
    for p in paths:
        with open(p) as f:
            out.append((p.stem, json.load(f)))
    return out


def describe(name, values, width=34):
    return (f"{name:<{width}} median={statistics.median(values):.4f}  "
            f"mean={sum(values) / len(values):.4f}  "
            f"min={min(values):.4f}  max={max(values):.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir", nargs="?", default=str(DEFAULT_DATA),
                    help=f"StepShield data/ directory (default: {DEFAULT_DATA})")
    ap.add_argument("--train-glob", default="train/*/*.jsonl",
                    help="glob for train trajectories, relative to data_dir")
    ap.add_argument("--holdout-glob", default="test_holdout/scrubbed/*.jsonl",
                    help="glob for holdout trajectories, relative to data_dir")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for the random-pair floor (default: 0)")
    ap.add_argument("--json", dest="json_out", default=None,
                    help="also write the full result (incl. per-trajectory rows) to this path")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    train_paths = sorted(data_dir.glob(args.train_glob))
    hold_paths = sorted(data_dir.glob(args.holdout_glob))
    if not train_paths or not hold_paths:
        raise SystemExit(
            f"no data under {data_dir} "
            f"(train={len(train_paths)}, holdout={len(hold_paths)}). "
            "Clone the StepShield benchmark as stepshield/ inside this repo, "
            "or pass its data/ directory as an argument."
        )

    train = load_split(train_paths)
    hold = load_split(hold_paths)

    tr_words = [word_set(header_text(d)) for _, d in train]
    tr_headers = [header_text(d) for _, d in train]
    tr_streams = [render_stream(d) for _, d in train]
    train_header_set = set(tr_headers)
    train_step_set = {render_step(s) for _, d in train for s in d.get("steps", [])}

    print(f"data dir : {data_dir}")
    print(f"train    : {len(train)} trajectories, {len(train_header_set)} distinct task headers")
    print(f"holdout  : {len(hold)} trajectories, "
          f"{len({header_text(d) for _, d in hold})} distinct task headers")
    print()

    rng = random.Random(args.seed)
    rows = []
    for hid, d in hold:
        h_header = header_text(d)
        h_words = word_set(h_header)
        h_stream = render_stream(d)

        sims = [jaccard(h_words, tw) for tw in tr_words]
        best = max(sims)
        cand = [i for i, s in enumerate(sims) if s == best]

        ratios = [difflib.SequenceMatcher(None, h_stream, tr_streams[i]).ratio() for i in cand]
        j = rng.randrange(len(train))
        rand_ratio = difflib.SequenceMatcher(None, h_stream, tr_streams[j]).ratio()

        h_steps = [render_step(s) for s in d.get("steps", [])]
        rows.append({
            "holdout_id": hid,
            "header_jaccard_max": best,
            "header_exact_duplicate_in_train": h_header in train_header_set,
            "n_same_template_train_trajectories": len(cand),
            "nearest_train_id": train[cand[ratios.index(max(ratios))]][0],
            "step_stream_ratio_max_same_template": max(ratios),
            "step_stream_ratio_mean_same_template": sum(ratios) / len(ratios),
            "step_stream_ratio_min_same_template": min(ratios),
            "step_stream_ratio_random_train_pair": rand_ratio,
            "n_steps": len(h_steps),
            "n_steps_verbatim_in_train": sum(1 for s in h_steps if s in train_step_set),
        })

    jac_max = [r["header_jaccard_max"] for r in rows]
    step_max = [r["step_stream_ratio_max_same_template"] for r in rows]
    step_mean = [r["step_stream_ratio_mean_same_template"] for r in rows]
    step_min = [r["step_stream_ratio_min_same_template"] for r in rows]
    step_rand = [r["step_stream_ratio_random_train_pair"] for r in rows]

    n_exact = sum(1 for r in rows if r["header_exact_duplicate_in_train"])
    n_above_09 = sum(1 for v in jac_max if v > 0.9)
    steps_total = sum(r["n_steps"] for r in rows)
    steps_verbatim = sum(r["n_steps_verbatim_in_train"] for r in rows)

    print("(A) TASK-HEADER REUSE  -- word-set Jaccard(title+description) to nearest train scenario")
    print("    " + describe("max over train split", jac_max))
    print(f"    holdout trajectories whose task header is a VERBATIM train header : "
          f"{n_exact}/{len(rows)}")
    print(f"    holdout trajectories with header Jaccard > 0.9                    : "
          f"{n_above_09}/{len(rows)}")
    print()

    print("(B) STEP-STREAM NOVELTY  -- difflib.SequenceMatcher ratio on rendered step streams")
    print("    " + describe("max over same-template train", step_max) + "   <-- headline")
    print("    " + describe("mean over same-template train", step_mean))
    print("    " + describe("min over same-template train", step_min))
    print("    " + describe(f"random train pair (seed {args.seed})", step_rand))
    print()

    print("(C) EXACT STEP REUSE")
    print(f"    rendered holdout steps appearing verbatim in the train split: "
          f"{steps_verbatim}/{steps_total}")
    print()

    print("SUMMARY")
    print(f"    task-header Jaccard, median over holdout      : {statistics.median(jac_max):.4f}")
    print(f"    step-stream difflib ratio, median over holdout: {statistics.median(step_max):.4f}")
    print("    Read together: the holdout reuses the train split's task templates almost")
    print("    exactly, while the step sequences under those templates are new. A detector")
    print("    fine-tuned on the train split therefore has template familiarity that an")
    print("    untrained baseline does not; detectors trained on an external corpus are")
    print("    unaffected by this.")

    if args.json_out:
        out = {
            "data_dir": str(data_dir),
            "n_train": len(train),
            "n_holdout": len(hold),
            "n_distinct_train_headers": len(train_header_set),
            "header_jaccard": {
                "median": statistics.median(jac_max),
                "mean": sum(jac_max) / len(jac_max),
                "min": min(jac_max),
                "max": max(jac_max),
                "n_verbatim_header_duplicates": n_exact,
                "n_above_0_9": n_above_09,
            },
            "step_stream_difflib": {
                "max_over_same_template": {
                    "median": statistics.median(step_max),
                    "mean": sum(step_max) / len(step_max),
                    "min": min(step_max), "max": max(step_max)},
                "mean_over_same_template": {"median": statistics.median(step_mean)},
                "min_over_same_template": {"median": statistics.median(step_min)},
                "random_train_pair": {"median": statistics.median(step_rand), "seed": args.seed},
            },
            "exact_step_reuse": {"verbatim": steps_verbatim, "total": steps_total},
            "per_trajectory": rows,
        }
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
