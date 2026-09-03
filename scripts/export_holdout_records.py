#!/usr/bin/env python3
"""Export the 216 holdout trajectories as pre-rendered records for the GPU scorer.
Rendering identical to training (nemotron_guard.render_*)."""
import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
BASE = SCRIPTS.parent  # repo root
sys.path.insert(0, str(SCRIPTS))
from nemotron_guard import render_header, render_step  # noqa: E402
from run_holdout import load_holdout  # noqa: E402

out = BASE / "data" / "holdout_records.jsonl"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    for t in load_holdout():
        rec = {
            "id": t["trajectory_id"],
            "rogue": t["trajectory_type"] == "rogue",
            "source": "holdout",
            "rogue_step": t.get("rogue_step"),
            "n_steps": len(t["steps"]),
            "category": t.get("category"),
            "header": render_header(t.get("task", {})),
            "steps": [{"text": render_step(s)} for s in t["steps"]],
        }
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
print("wrote", out)
