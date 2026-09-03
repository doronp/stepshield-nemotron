# Release checklist

This repository is publish-ready **except** for one deliberate gate:

1. **Identity is not baked in.** No GitHub owner or HuggingFace namespace appears in any committed artifact; every repo/model URL carries an owner placeholder token (the two token strings are defined once, in `scripts/set_identity.sh`, which deliberately retains them after substitution — prose elsewhere never spells them out, so the identity step cannot corrupt documentation).

There is deliberately **no** prospective-results gate: the pre-specified blind evaluation was established to be unsatisfiable on this benchmark (no rogue-containing data pool untouched by development remains — see docs/METHODOLOGY.md §6), and the documentation states this as a limitation-and-finding rather than a pending promise. All numbers ship labeled **held-out with respect to training, not benchmark-external**; nothing is awaiting amendment.

## The single publish-time identity step

```bash
bash scripts/set_identity.sh <github-owner> <hf-namespace> [extra paths...]
# example, also rewriting the HF upload dir and article drafts:
bash scripts/set_identity.sh myorg myorg ../stepshield-nemotron-hf ../stepshield-nemotron/publishing
```

The script rewrites every tracked file plus any extra paths, then greps to prove no token remains (excluding itself). After it runs, `CITATION.cff` contains the final `repository-code` URL — validating it with `cffconvert --validate` is a **required** pre-publish gate.

## Publish steps (after identity is set and all gates below pass)

```bash
# GitHub
git add -A && git commit -m "Set release identity"
gh repo create <github-owner>/stepshield-nemotron --public --source . --push

# HuggingFace (upload dir prepared separately; contains adapter + card + NVIDIA license copy + NOTICE)
huggingface-cli upload <hf-namespace>/stepshield-nemotron-mini-4b-lora <hf-upload-dir> . --repo-type model
```

## Pre-publish verification

- `grep -rn "__GH_""OWNER__\|__HF_""OWNER__" . --exclude-dir=.git --exclude=set_identity.sh` → must be empty (the split-string pattern survives the identity substitution; `set_identity.sh` keeps its own token definitions by design).
- `cffconvert --validate` on `CITATION.cff` → must pass.
- `python scripts/calibrate_report.py --out /tmp/repro.json` (with the StepShield benchmark cloned as `stepshield/`) → must reproduce `results/calibration_report.json` except the `scores_file` metadata string.
- `grep -rn "PEND""ING" . --exclude-dir=.git` → must be empty (the split-string pattern keeps this line out of its own results; lowercase `pending` in `scripts/` is code-variable naming and is not matched).
- The NVIDIA Community Model License copy accompanies the adapter distribution (§1.2(c)); the NOTICE modification statement (§2.2.1) is intact.
