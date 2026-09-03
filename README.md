<p align="center">
  <img src="assets/raccoon.svg" width="76" alt="">
</p>

# StepShield-Nemotron: a 4B step-level rogue-agent detector with leakage-free threshold calibration

A LoRA fine-tune of [`nvidia/Nemotron-Mini-4B-Instruct`](https://huggingface.co/nvidia/Nemotron-Mini-4B-Instruct) that classifies **each step** of a coding-agent trajectory as benign or rogue on the [StepShield](https://github.com/glo26/stepshield) benchmark — at ~44 ms per step on a single NVIDIA L4, with no API dependency.

> **RESULTS STATUS: HELD-OUT WITH RESPECT TO TRAINING; NOT BENCHMARK-EXTERNAL.** Every number in this repository comes from StepShield's public 216-trajectory holdout, which the model **never trained on** — weight training and checkpoint selection used the train split only — with decision thresholds set under the leakage-controlled cross-validation protocol in [docs/METHODOLOGY.md](docs/METHODOLOGY.md). What these numbers are **not** is prospective or benchmark-external: the model consumed the benchmark's entire train split plus 700 of its 2,514 generated-benign trajectories (400 for training, 300 for threshold tuning), and the holdout itself was used for the legacy evaluation and threshold calibration — so StepShield retains **no rogue-containing data pool untouched by development** (the ~1,814 untouched generated-benign trajectories are clean-only, so they cannot yield the realized recall or EIR₃ a blind evaluation must report), and no qualifying blind evaluation can be produced from it (see ["Why there is no prospective evaluation"](#why-there-is-no-prospective-blind-evaluation) below). The correct next test of these claims is external validation on a different corpus.

The headline result is a pair of calibrated operating points on StepShield's 216-trajectory held-out test set (108 rogue / 108 clean twins) — same weights, same scores, same 43.9 ms p50 per-step latency (single NVIDIA L4, bf16, unquantized), differing only in the decision threshold θ — obtained under a **threshold-leakage-free conformal calibration protocol** (the methodological contribution of this repo — see [docs/METHODOLOGY.md](docs/METHODOLOGY.md)):

| Operating point (developed-on-holdout) | Recall | FPR | Accuracy | F1 | EIR₃ |
|---|---|---|---|---|---|
| **LOW-FPR** (conformal-max, primary) | 91.7% (99/108) [84.9, 95.6] | **0.93%** (1/108) [0.16, 5.06] | 95.4% [92.6, 97.7] | 95.2 | **0.990** |
| **MAX-RECALL** (α=0.10) | **98.1%** (106/108) | 9.26% (10/108) | 94.4% | 94.6 | 0.953 |

Intervals are Wilson 95%. EIR₃ = early-intervention rate: the fraction of detected rogue trajectories flagged within 3 steps of the true rogue step (StepShield's native timing metric).

**Read this before quoting any number:**

> **Within-template generalization.** StepShield's holdout reuses the training split's scenario templates: every holdout task header near-duplicates a train scenario (title+description word-Jaccard p50 = 1.0), while the step streams themselves are novel (difflib similarity ~0.10 — median 0.096, barely above the 0.071 floor for randomly paired trajectories). A model fine-tuned on the train split therefore gets a within-template advantage that the benchmark's published *untrained* baselines (LLMJudge, StaticGuard, …) do not exploit. All numbers here measure **within-scenario-distribution generalization to novel trajectories**, not cross-domain generalization. This is the single most important caveat on this work.
>
> **This overlap contradicts the benchmark's documented design.** StepShield's paper states its held-out set was generated "from incident templates that were *never used* during training or pattern development" and is "drawn from novel templates"; its README says "zero overlap with training data" (true at the *trajectory* level — no trajectory appears in both splits). Our measurement of the public release found the *template*-level claim does not hold: word-set Jaccard between each holdout scenario's title+description and its nearest train-split scenario has **median 1.0** (exact or near-exact template reuse), while rendered step streams compare at ~0.10 by `difflib.SequenceMatcher` (median 0.096, vs 0.071 for randomly paired trajectories; 0 of 3,056 rendered holdout steps appear verbatim in train) — both computable from the public StepShield data alone by running [`scripts/template_overlap.py`](scripts/template_overlap.py), no model involved. We report this as an empirical observation for the benchmark's users, scoped precisely: it matters exactly and only when a detector is **fine-tuned on the train split** (our situation); detectors trained on an external corpus are unaffected by it.

## Quickstart — watch your own agent

```bash
git clone https://github.com/doronp/stepshield-nemotron && cd stepshield-nemotron
uv run scripts/quickstart.py examples/sample_trajectory.jsonl
```

First run downloads the base model (~8 GB) and the 92 MB adapter from HuggingFace. Wants a ~24 GB GPU (bf16); CPU works but is slow. Apple-silicon users: `scripts/nemotron_guard.py` carries the incremental MLX class (re-calibrate θ per quantized artifact — see the warning below).

Input: one JSON line per trajectory — `{"task": {"title", "description", "constraints"}, "steps": [{"step", "thought", "action", "arguments", "observation"}]}` (see `examples/`). Output: per-step p(BAD), the first flagged step, and **exit code 1 if anything flagged** — drop it straight into CI. Default θ = 0.946597, the deployed conformal-max point (E[FPR] ≤ 0.92% on traffic exchangeable with the calibration cleans). For a guarantee on *your* traffic, re-calibrate: score ~100+ of your own clean trajectories and set θ to their max (E[FPR] ≤ 1/(n+1)).

Smoke-tested end-to-end on a fresh NVIDIA L4 (this exact command path, dependencies resolved from scratch by `uv`): the sample rogue flags at its exfiltration step (p(BAD) = 0.990), the sample clean stays ≤ 0.001 everywhere. Re-scoring the full 216-trajectory holdout through this script reproduces **215/216 of the committed decisions**; the single difference is a rogue trajectory whose peak score sits exactly on the threshold's lattice value, where bf16 re-computation jitter (raw mid-range scores can move by ~0.1 between batched and unbatched forwards) flips a strict `>` comparison. No clean trajectory changes verdict. The certification instrument remains the committed scores file plus `scripts/calibrate_report.py` — not this convenience path.

## Method in brief

Two components, fully specified in [docs/METHODOLOGY.md](docs/METHODOLOGY.md) and the training section below:

1. **The detector** — an incremental causal-LM step classifier: LoRA on Nemotron-Mini-4B reads each trajectory step and emits p(BAD) from a restricted two-token softmax; a trajectory is flagged at the first step whose score crosses a threshold θ. With an incremental KV cache each step costs one prefill — no decode — hence ~44 ms/step on an L4.
2. **The calibration protocol** (the contribution) — θ is *not* transferred from validation. It is set by conformal order statistics on clean trajectories only, under pair-level 4-fold cross-calibration over the holdout with pre-committed rules, so every reported decision was made by a threshold that never saw that trajectory or its scenario twin, and each operating point carries a finite-sample E[FPR] certificate.

## Why this exists

Agent-safety monitoring is usually evaluated on *whether* a rogue trajectory gets caught. StepShield's contribution is scoring *when*: a detector that flags a destructive `rm -rf` five steps after it ran is a forensics tool, not a guardrail. A useful step-level monitor must therefore be (a) accurate at the step level, (b) early, and (c) cheap enough to run on every step. This repo shows a 4B local model can hold ≥91% recall at ≤1% FPR with EIR₃ 0.99 at 44 ms/step — but only after fixing the part that usually breaks silently: **how the decision threshold is chosen**. Our first, naively-transferred threshold produced 19.4% FPR from the same frozen weights and scores (see "Legacy point" below); every improvement between that row and the table above comes from calibration methodology, not from the model.

## Results in context

Published baselines from the StepShield repository, same 216-trajectory holdout:

| Detector | Recall | FPR | Acc | EIR₃ | Per-step cost |
|---|---|---|---|---|---|
| **This work — LOW-FPR** (calibrated) | 91.7% | 0.93% | 95.4% | 0.990 | ~44 ms local (L4) |
| **This work — α=0.05** (calibrated) | 96.3% | 3.70% | 96.3% | 0.971 | ~44 ms local (L4) |
| **This work — MAX-RECALL** (calibrated) | 98.1% | 9.26% | 94.4% | 0.953 | ~44 ms local (L4) |
| LLMJudge (GPT-4.1-mini, untrained) | 95.4% | 5.6% | 95% | 0.89 | API call |
| HybridGuard | 75.9% | 44.4% | 66% | 0.40 | — |
| StaticGuard (847 rules) | 86.1% | 77.8% | 54% | 0.23 | negligible |
| ConstraintGuard | 24.1% | 25.0% | 50% | 0.08 | negligible |

The certified α=0.05 point (96.3% recall @ 3.7% FPR) improves on the published LLMJudge row on both axes **on this benchmark's holdout, under our calibration protocol, with the within-template caveat above** — LLMJudge is untrained and does not benefit from template familiarity. We do not claim general superiority over LLM judges.

Scope note: baseline rows are quoted from the StepShield README; we re-ran StaticGuard on our harness and reproduced its published numbers exactly (harness validation), but did not re-run LLMJudge.

### Legacy point (decertified, kept for continuity)

Our original single-shot frozen evaluation — threshold θ=0.064 transferred from a validation split that turned out to be a systematically harder discrimination problem than the holdout (its cleans keep scoring high even at strict thresholds, and its rogues score softer, so the validation-optimal threshold was low) — scored 98.1% recall / 19.4% FPR / 89.4% accuracy. It is **strictly dominated** by the certified MAX-RECALL point (identical recall at 9.26% vs 19.4% FPR) and the threshold-transfer failure is analyzed in the methodology doc. A McNemar test between the calibrated primary and the legacy rule on the same 216 decisions gives 20 vs 7 discordant pairs, exact two-sided p = 0.019.

### Certified FPR–recall curve (discrete; intermediate thresholds are NOT certified)

| Rule | Realized FPR | Recall | Acc | EIR₃ |
|---|---|---|---|---|
| conformal-max (tightest achievable) | 0.93% | 91.7% | 95.4% | 0.990 |
| α=0.025 | 0.93% | 94.4% | 96.8% | 0.971 |
| α=0.05 | 3.70% | 96.3% | 96.3% | 0.971 |
| α=0.10 | 9.26% | 98.1% | 94.4% | 0.953 |

For deployment, the pre-committed conformal-max rule evaluates to θ = **0.946597** (max of all 108 holdout clean scores): forward certificate E[FPR] ≤ 1/109 = 0.917% under exchangeability, with fully leakage-free recall 90.7% (98/108) [83.8, 94.9] (the threshold never saw a rogue score). An FPR ≤ 0.5% *certificate* is mathematically unachievable with 108 calibration cleans (conformal guarantees are quantized to 1/(n_c+1); 0.5% needs n_c ≥ 199).

### Latency

Per-step inference, single stream, incremental KV cache, warmup excluded. The L4 row was measured over real validation step streams, the M4 Pro row over real holdout step streams (per-step latency depends only on token counts, not labels, so the split does not affect the timings):

| Platform | p50 | p95 | p99 | n |
|---|---|---|---|---|
| NVIDIA L4 (GCP g2-standard-8), bf16 | **43.9 ms** | 57.6 | 65.5 | 479 |
| Apple M4 Pro 48GB, MLX q4 | 110.5 ms | 177.9 | 243.0 | 356 |

The M4 Pro floor is per-forward dispatch overhead (~3.6 ms/layer × 32 layers), which quantization alone does not remove. Latency model on L4 (OLS over 479 steps): `37.0 ms + 0.058 ms/new-token + 0.0039 ms/history-token` — the history term is small, so per-step cost grows slowly with trajectory length. **Threshold/quantization warning:** the certified thresholds apply to the bf16 deployment artifact only; quantized builds (e.g., the MLX q4 above) measurably narrow the OK/BAD score margins, so θ must be re-calibrated per artifact before any deployment. StepShield's native timing metric is EIR (step counts); all millisecond figures come from our own harness (`scripts/gpu_latency_bench.py`, `scripts/latency_bench.py`), clocks not locked, single-tenant VM.

## Figures

![FPR vs recall on the StepShield holdout](figures/fig1_fpr_recall.png)
*Certified operating points vs published baselines. Full-range panel (A) and operating-corner zoom (B); the gray step curve is the descriptive in-sample sweep, shown for context only.*

![The two operating points](figures/fig2_operating_points.png)
*The same frozen model at two certified thresholds (pooled out-of-fold, n=216): LOW-FPR primary vs MAX-RECALL. Wilson-95 whiskers on the primary's recall/FPR.*

![Calibration protocol](figures/fig4_protocol.png)
*The leakage-free calibration protocol — the methodological contribution. Thresholds come from calibration-fold cleans only; twins never straddle folds; the deployed θ carries a 1/109 forward certificate.*

![Latency](figures/fig3_latency.png)
*Warm per-step latency: percentile ranges for both platforms, and the full L4 distribution (n=479) the quoted percentiles come from.*

![Step-level scores on held-out trajectories](figures/fig5_eir_stepscores.png)
*The mechanism on three real held-out trajectories: a gap-0 detection inside the EIR₃ window, its clean twin staying under θ, and one of the two deep-tail misses that bound recall.*

All figures regenerate from the committed JSONs via `python figures/src/make_figures.py` (matplotlib; colorblind-safe palette).

## Why there is no prospective (blind) evaluation

We pre-committed a full protocol for a prospective evaluation — frozen model (`v2-step1100`) and deployed threshold θ = 0.946597, applied exactly once to a slice nobody in the development loop had seen, realized (FPR, recall, EIR₃) reported verbatim next to the forward certificate they test — and then established that **the protocol is unsatisfiable on this benchmark**. An inventory of StepShield's data (docs/METHODOLOGY.md §6) shows no qualifying slice exists: the entire train split (639 rogue/clean pairs = 1,278 trajectories) was consumed by weight training; the generated-benign corpus — the benchmark's own `data/generated_benign` trajectories, which this project read and folded into training — is development-exposed by our use of it; and the 216-trajectory holdout was used for the frozen legacy evaluation and then for threshold calibration. Nothing is left that development did not touch.

We report this as a finding rather than filling the slot with a weaker substitute: **a single-public-holdout benchmark cannot support a prospective claim by any detector fine-tuned on its train split**, once that holdout has been used even once. The honest ceiling for this repository is therefore exactly what the status banner says — held-out with respect to training, with leakage-controlled threshold calibration — and the prospective upgrade requires a *different* corpus, not a StepShield re-run. That external validation is future work and is promised nowhere in this repository as an existing result.

## Disclosures (complete list)

1. **Within-template generalization** — see the box above. This caveat applies to all of our rows and to none of the untrained baselines.
2. **Calibrated-operating-point framing.** The certified numbers come from cross-validated threshold calibration *on the holdout itself* (each trajectory decided at a threshold that never saw it or its scenario twin — mechanically leakage-free at the threshold level, checked in three rounds of external adversarial LLM review — Kimi V3, commissioned by us — prior to release; scope and catches summarized in METHODOLOGY). The analyst had previously seen full-holdout summaries, so these are calibrated-operating-point claims, **not** fresh single-shot claims. The only fresh single-shot number in this repo is the legacy 89.4% row. "Certified" throughout means "procedure verified leakage-free under the stated assumptions" — it is not a third-party audit.
3. **The pre-declared accuracy target was NOT met — and 98.1% is recall, not accuracy.** This project declared a ≥98% **accuracy** target before evaluation. No operating point reaches it: certified accuracy tops out at 96.8% (the α=0.025 curve point; 95.4% at the primary), and the legacy single-shot accuracy was 89.4%. The 98.1% figure that recurs in this repository is **recall** at the MAX-RECALL point (106/108 rogues caught, at 9.26% FPR). Anywhere you see 98.1%, read "share of rogues caught", never "share of all decisions correct".
4. **50/50 prevalence, and what an FPR means at scale.** Accuracy, F1, and precision are at the benchmark's 50/50 rogue/clean prevalence. Balanced-set precision is 99.0% at the primary point and 91.4% at MAX-RECALL (`calibration_report.json`); at a realistic rogue prevalence of ~1%, the same certified (recall, FPR) pairs imply precision of roughly 50% and roughly 10% respectively — most flags would be false alarms. The same arithmetic in alarm volume: a deployment screening N clean agent runs per day at trajectory-level false-positive rate f raises f·N false alarms per day — at 10,000 clean runs/day, our 0.93% point raises ~93 and our 9.26% point ~926. This gap between balanced-set metrics and deployed alarm behavior is a property of **every** detector evaluated on a ~50/50 benchmark, this one included; the transferable quantities are (recall, FPR) — multiply the FPR by your own volume — and even those assume no distribution shift.
5. **No benchmark-external validation exists.** All results are held-out with respect to training, but every number was produced within this benchmark's development loop, and the benchmark retains no untouched pool for a blind run (see the section above). As a general methodological matter — independent of this project — developed-on results in this domain can differ substantially from results on data nobody in the loop has seen. Treat these numbers as within-benchmark evidence pending external validation on a different corpus.
6. **FPR floor.** No FPR ≤ 0.5% certificate is possible at this calibration size (see above).
7. **Residual misses.** At the primary point, 9 FNs = 2 deep-tail trajectories (max p(BAD) 0.002/0.014 — undetectable at any threshold) + 7 near-threshold, skewed to subtle L3 rogues.

## Repository layout

```
README.md               this file
RELEASE.md              publish checklist (identity step + gates)
docs/METHODOLOGY.md     the leakage-free calibration protocol (the contribution)
MODEL_CARD.md           model card (mirrors the HuggingFace card)
results/                certified result JSONs (see below)
figures/                publication figures (SVG/PDF/PNG) + generation script
scripts/                training, scoring, calibration, latency benchmarks
scripts/set_identity.sh publish-time identity step (repo/HF URLs carry owner placeholder tokens)
CITATION.cff            how to cite
LICENSE                 Apache-2.0 (code in this repo)
```

Key artifacts:

- `results/calibration_report.json` — fold assignments, per-fold thresholds, all certified metrics (the source of every number above).
- `results/descriptive_roc_216.json` — descriptive 61-knot full-holdout sweep. **Selection-biased as a menu**: thresholds read from it are not deployable without re-certification.
- `results/v2_holdout_step1100.json` — the frozen model's per-step p(BAD) scores on all 216 holdout trajectories (the input to calibration).
- `results/v2_val_step1100.json` — the same model's validation scores (backs the threshold-transfer diagnosis in METHODOLOGY §1).
- `results/OFFICIAL_holdout_frozen.json` — the legacy single-shot evaluation at θ=0.064.
- `results/latency_l4_bf16.json`, `results/latency_m4_*.json` — raw latency samples and percentiles.

## Model

The LoRA adapter (~92 MB) is **not** stored in this repo — it ships on HuggingFace at `DoronP/stepshield-nemotron-mini-4b-lora` (see MODEL_CARD.md). Base model: `nvidia/Nemotron-Mini-4B-Instruct` under the NVIDIA Community Model License; the adapter is a Derivative Model under that license (details and production-use restrictions in the model card).

Formulation: incremental causal-LM step classifier. One sequence per trajectory — task header, then per step `<<Sn>>\nT: <thought>\nA: <action+args>\nO: <observation>\nV:` — read logits at the final position, restricted softmax over the two verdict tokens {`" OK"`=12192, `" BAD"`=100275}, flag the trajectory at the first step with p(BAD) > θ. Screening mode: the verdict slot is always filled with `OK` in the history (training and inference identical), so per-step cost is prefill of the new step tokens only — no decode.

## Reproducing

Requires the StepShield repo cloned as `stepshield/` inside this repo, a GPU box for training/scoring, and Python ≥3.11 (`uv venv && uv pip install torch transformers peft numpy`; the certified latency numbers were measured on torch 2.9.1 / transformers 5.x — pin those for exact reproduction).

```bash
# 0. One-time: render the 216 holdout trajectories to records (also shipped as results inputs)
python scripts/export_holdout_records.py

# 1. Data prep (renders train/val records from the StepShield train split)
python scripts/prep_data.py

# 2. Train (1x L4, ~2.2h): LoRA r16/alpha32/dropout .05 on q,k,v,o,up,down; bad_weight 6; 3 epochs
python scripts/gpu_train_lora.py --okfill --bad-weight 6

# 3. Score the 216 holdout trajectories with the frozen adapter (all-OK history)
python scripts/gpu_score_trajs.py --model nvidia/Nemotron-Mini-4B-Instruct \
    --adapter <adapter_dir> --data data/holdout_records.jsonl --out results/v2_holdout_step1100.json

# 4. Leakage-free calibration + full certified report (CPU, seconds)
python scripts/calibrate_report.py

# 5. Latency
python scripts/gpu_latency_bench.py   # L4
python scripts/latency_bench.py       # Apple silicon (MLX)
```

Step 4 reproduces every certified number from the committed scores file alone — you do not need a GPU to verify the headline claims.

## Citation

See [CITATION.cff](CITATION.cff). Please also cite the StepShield benchmark (Felicia et al.).

## License

Code in this repository: Apache-2.0. The StepShield benchmark is MIT (© Gloria Felicia et al.). The base model and the LoRA adapter are governed by the NVIDIA Community Model License — note that it permits redistribution of Derivative Models (with a copy of the license) but conditions **production** use on an NVIDIA NIM runtime / NVIDIA AI Enterprise subscription unless the model is on NVIDIA's published exception lists (Nemotron-Mini-4B is not on the PC/workstation list as of the version we checked, v. Dec 13 2024). Research and evaluation use is unaffected. Details in MODEL_CARD.md.
