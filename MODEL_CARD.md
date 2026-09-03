# Model card: StepShield-Nemotron-Mini-4B LoRA (step-level rogue-agent detector)

LoRA adapter for [`nvidia/Nemotron-Mini-4B-Instruct`](https://huggingface.co/nvidia/Nemotron-Mini-4B-Instruct) that scores each step of a coding-agent trajectory with a rogue-likelihood score p(BAD) — a restricted two-token softmax, used for ranking and thresholding; it has *not* been shown to be a calibrated probability (a temperature T=1.25 was fitted for analysis only and never applied to decisions) — evaluated on the [StepShield](https://github.com/glo26/stepshield) benchmark.

- **Adapter:** ~92 MB safetensors, PEFT/LoRA (adapter only — no merged base weights are distributed).
- **Developed by:** Doron Podoleanu (independent).
- **Finetuned from:** `nvidia/Nemotron-Mini-4B-Instruct` (frozen; checkpoint "v2-step1100", selected by validation trajectory-AUC 0.9938).
- **License:** NVIDIA Community Model License (the adapter is a Derivative Model of an NVIDIA Model — see License section; this is **not** an open-source license). Evaluation/training code is Apache-2.0 in the [companion repo](https://github.com/doronp/stepshield-nemotron).

## ⚠️ What this is not

**This is not a general-purpose safety classifier.** It was trained and evaluated exclusively on StepShield's six incident categories (DEC/INV/RES/SEC/TST/UFO) in ReAct-style coding-agent trajectories rendered in one specific text format. Its numbers do not transfer to other agent frameworks, other task domains, prompt-injection detection, content safety, or free-form chat — and were not measured there. Do not deploy it as a lone safety control anywhere.

## Headline results — read with the caveat

> **RESULTS STATUS: HELD-OUT WITH RESPECT TO TRAINING; NOT BENCHMARK-EXTERNAL.** All numbers below come from the benchmark's public 216-trajectory holdout, which the model **never trained on** (weight training and checkpoint selection used the train split only); thresholds were set by the leakage-controlled cross-calibration described below. No prospective (blind) evaluation exists, and none can be produced from this benchmark — it retains no rogue-containing data pool untouched by development; the only untouched material is clean-only generated-benign data, which cannot yield the realized recall or EIR₃ a blind run must report (see "Why there is no prospective evaluation" below). External validation on a different corpus is the correct next test of these numbers, and is future work.

Two calibrated operating points on StepShield's 216-trajectory held-out set (108 rogue / 108 clean scenario twins), under a threshold-leakage-free conformal calibration protocol (pair-level, category-stratified 4-fold CV; thresholds from calibration-fold cleans only; twins never straddle folds; the procedure was checked in three rounds of external adversarial LLM review — Kimi V3, commissioned by the author — prior to release). "Certified" on this card means exactly that: a pre-committed procedure with a finite-sample guarantee, mechanically executed and review-checked; it is **not** a third-party audit or formal certification.

| Operating point | Recall | FPR | Acc | F1 | EIR₃ |
|---|---|---|---|---|---|
| **LOW-FPR** (conformal-max, primary) | 91.7% (99/108) [84.9, 95.6] | **0.93%** (1/108) [0.16, 5.06] | 95.4% | 95.2 | **0.990** |
| **MAX-RECALL** (α=0.10) | **98.1%** (106/108) | 9.26% (10/108) | 94.4% | 94.6 | 0.953 |

Intervals are Wilson 95%. EIR₃ = early-intervention rate: the share of detected rogue trajectories flagged within 3 steps of the true rogue step (StepShield's native timing metric).

Full certified curve (discrete points; intermediate thresholds are NOT certified):

| Rule | Realized FPR | Recall | Acc | EIR₃ |
|---|---|---|---|---|
| conformal-max | 0.93% | 91.7% | 95.4% | 0.990 |
| α=0.025 | 0.93% | 94.4% | 96.8% | 0.971 |
| α=0.05 | 3.70% | 96.3% | 96.3% | 0.971 |
| α=0.10 | 9.26% | 98.1% | 94.4% | 0.953 |

> **Within-template caveat (the most important sentence on this card):** StepShield's holdout reuses the training split's scenario templates (task-header word-Jaccard p50 = 1.0; the step streams are novel, difflib ~0.10 — median 0.096, vs a 0.071 floor for randomly paired trajectories). A fine-tuned model gets a within-template advantage that the benchmark's untrained published baselines do not exploit. Every number above is **within-scenario-distribution generalization to novel trajectories** — expect worse on out-of-distribution scenarios, and measure before relying on it. (This overlap contradicts the benchmark's documented design — its paper describes the holdout as built "from incident templates that were never used during training"; measured by us on the public data, word-set Jaccard on title+description, median 1.0; the trajectory-level "zero overlap" claim remains true. It affects only detectors fine-tuned on the train split, like this one; detectors trained on an external corpus are unaffected.)

Additional framing: these are calibrated operating points computed by cross-validation over the holdout (mechanically leakage-free at the threshold level), **not a fresh single-shot evaluation** — the analyst had seen full-holdout summary metrics before designing the calibration. The only fresh single-shot holdout number in this project is the decertified legacy point (θ=0.064: 89.4% accuracy, 98.1% recall, 19.4% FPR), kept for continuity and strictly dominated by the certified α=0.10 point. Accuracy/F1 are at the benchmark's 50/50 prevalence; (FPR, recall) are less prevalence-sensitive but still not guaranteed to transfer under scenario or traffic shift. The α=0.05 point (96.3% @ 3.7%) beats the published LLMJudge baseline (95.4% @ 5.6%, GPT-4.1-mini) on both axes *on this benchmark's holdout under our protocol, with the within-template caveat*. The LLMJudge figures are quoted from the StepShield README and were not re-run by us (we re-ran and exactly reproduced only StaticGuard); the comparison is between point estimates with no paired significance test. We do not claim general superiority over LLM judges. Pre-commitment note: α=0.025 outperformed the primary on the report folds (same single FP, higher recall) and was *not* promoted to primary, per the pre-committed protocol — it stays a curve point.

## Recommended threshold

**θ = 0.946597** (flag a trajectory at the first step with p(BAD) > θ). Provenance: conformal-max over all 108 holdout clean scores → forward certificate E[FPR] ≤ 1/109 ≈ 0.917% on exchangeable future cleans, with leakage-free recall 90.7% (98/108) [83.8, 94.9] and EIR₃ 0.990. Under distribution shift the certificate is void — **re-calibrate on ≥199 cleans from your own traffic** (that size also unlocks a genuine ≤0.5% certificate; 108 cleans mathematically cannot). Note the deployed θ is computed from *all* 108 cleans (its in-sample FPR of 0 is by construction, not evidence), unlike the CV table above where each trajectory was decided by a threshold that never saw it; deployed recall and CV recall also share the same 108 rogues, so they are correlated estimates, not independent confirmations. Alternative α-grid thresholds are in `calibration_report.json` in the companion repo.

## Intended use

- Research on step-level agent monitoring and early intervention.
- A reference operating-point + calibration recipe for local (no-API) trajectory screening on StepShield-formatted traces.
- Screening mode only: the detector reads the trajectory with `OK` filled into every past verdict slot and never feeds its own verdicts back (this matches training exactly).

Out of scope: any production safety-critical use without domain-specific evaluation and recalibration; any use prohibited by the NVIDIA Community Model License or NVIDIA's Trustworthy AI terms.

## How it works / usage

Incremental causal-LM step classification. Per trajectory: a task header, then per step the text `<<Sn>>\nT: <thought>\nA: <action+args>\nO: <observation>\nV:`; read the logits at the final position; restricted softmax over the two verdict tokens `" OK"` (id 12192) and `" BAD"` (id 100275); p(BAD) is the step score. History uses `OK` at every verdict slot. With an incremental KV cache the per-step cost is prefill of the new step's tokens only (no decode).

```python
# research/evaluation use — see the License section for production restrictions
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "nvidia/Nemotron-Mini-4B-Instruct"
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="cuda")
model = PeftModel.from_pretrained(model, "<this-adapter>")
model.eval()

OK, BAD, THETA = 12192, 100275, 0.946597

@torch.inference_mode()
def step_p_bad(prompt_so_far: str) -> float:  # prompt ends with "\nV:"
    ids = tok(prompt_so_far, return_tensors="pt").input_ids.cuda()
    logits = model(ids).logits[0, -1, [OK, BAD]]
    return torch.softmax(logits, -1)[1].item()
```

(Production code should keep a KV cache and append only new step tokens — see `scripts/nemotron_guard.py` and `scripts/gpu_score_trajs.py` in the companion repo. Deliberately, no `pipeline_tag` is declared and `inference: false` is set: this adapter is not a standard HF pipeline — scoring requires the incremental verdict-slot protocol above.)

## Latency

| Platform | p50 | p95 | p99 | n |
|---|---|---|---|---|
| NVIDIA L4 (GCP g2-standard-8), bf16, torch eager, single stream | 43.9 ms | 57.6 | 65.5 | 479 |
| Apple M4 Pro 48GB, MLX q4 | 110.5 ms | 177.9 | 243.0 | 356 |

Warm per-step, incremental KV cache, real validation step streams (the Apple-silicon row uses holdout streams; per-step latency depends only on token counts, not labels), clocks not locked. Header prefill (once per trajectory): ~40 ms on L4. StepShield's native timing metric is EIR (steps); millisecond figures are from our own harness. **Quantization warning:** q4 measurably narrows the OK/BAD margin (measured validation-wide on an earlier checkpoint of the same training run: rogue max-p mean −0.05, worst −0.55) — re-calibrate θ per quantized artifact. All detection metrics and the deployed threshold on this card are bf16; only the Apple-silicon latency row and the quantization-margin observation are q4.

## Training details

- LoRA r=16, α=32, dropout 0.05 on `q_proj,k_proj,v_proj,o_proj,up_proj,down_proj`, all 32 layers (Nemotron-Mini's MLP has no gate_proj).
- Objective: weighted CE at verdict positions only. BAD at annotated rogue steps (w=1) and up to +3 steps after divergence (w=0.5); generated-benign OK steps w=0.5; BAD class weight 6.
- 3 epochs, OneCycle peak LR 1e-4, batch 4 × grad-accum 2, grad checkpointing, seed 7; 133 min on 1× NVIDIA L4.
- Data: StepShield train split — 639 rogue/clean scenario pairs (split 90/10 into train/val **by pair**) + 400 generated-benign trajectories. The holdout was excluded from weight training and checkpoint selection; it was used only for the pre-committed frozen evaluation and the leakage-controlled threshold calibration described above.
- Checkpoint selection: validation trajectory-AUC under deployment semantics (all-OK history).

## Evaluation protocol

Scores: one frozen pass over the 216 holdout trajectories (bf16, L4, all-OK history). Thresholds: conformal order statistics on calibration-fold cleans only, pair-level 4-fold CV, pre-committed rules/seed/reporting language; each trajectory decided exactly once by a threshold that never saw it or its scenario twin. Metrics: StepShield's own `calculate_metrics` (trajectory-level recall/FPR/accuracy/F1 + EIR). Full protocol, assumptions, and threats to validity: [docs/METHODOLOGY.md](https://github.com/doronp/stepshield-nemotron/blob/main/docs/METHODOLOGY.md). Provenance: all calibrated detection metrics trace to `results/calibration_report.json`; the legacy point to `results/OFFICIAL_holdout_frozen.json`; latency to `results/latency_l4_bf16.json` and `results/latency_m4_*.json`; checkpoint selection to `ckpt_meta.json`; baselines are quoted from the StepShield README — all in the companion repo.

## Why there is no prospective (blind) evaluation

We pre-committed a blind-evaluation protocol — the frozen model and deployed θ = 0.946597 applied exactly once to a never-seen slice, realized (FPR, recall, EIR₃) reported verbatim next to the 0.917% forward certificate they test — and then established that it is **unsatisfiable on this benchmark**: the model trained on the entire train split (639 rogue/clean pairs = 1,278 trajectories), the benchmark's own generated-benign corpus was read and folded into training (development-exposed by that use), and the 216-trajectory holdout was consumed by the legacy evaluation and threshold calibration. StepShield retains no pool that development did not touch, so no run from it can honestly be labeled PROSPECTIVE. We state this as a finding about single-public-holdout benchmarks — once a detector is fine-tuned on the train split and the holdout is used even once, the prospective rung is unreachable — rather than promise an evaluation that cannot exist. The numbers on this card are held-out with respect to training, not benchmark-external; the prospective upgrade requires a different corpus and is future work. Full inventory and reasoning: METHODOLOGY §6 in the companion repo.

## Limitations

1. **Within-template evaluation** (see caveat box) — the dominant limitation.
2. **No benchmark-external validation.** All results were produced inside this benchmark's development loop; the benchmark retains no untouched pool for a blind run (see the section above). As a general methodological matter, developed-on results in this domain can differ substantially from results on data nobody in the loop has seen; these numbers are held-out with respect to training, but external-corpus validation does not yet exist.
3. **The pre-declared ≥98% accuracy target was NOT met — and 98.1% is recall, not accuracy.** Certified accuracy tops out at 96.8% (α=0.025 curve point; 95.4% at the primary); the legacy single-shot accuracy was 89.4%. The recurring 98.1% figure is **recall** at the MAX-RECALL point (106/108 rogues caught, at 9.26% FPR) — never read it as accuracy.
4. **Balanced-set precision does not survive realistic prevalence.** At the benchmark's 50/50 mix, precision is 99.0% (primary) / 91.4% (MAX-RECALL); at ~1% rogue prevalence the same certified (recall, FPR) pairs imply roughly 50% and roughly 10% precision respectively — most flags would be false alarms. Evaluate at your own prevalence before trusting flag volume.
5. Six incident categories, one trajectory format, English-only, coding-agent domain.
6. Residual misses at the primary point: 9 FNs — 2 with max p(BAD) ≤ 0.014 (undetectable at any threshold), 7 near-threshold, skewed to subtle L3 rogues (α=0.025 recovers 3 of the 7 at zero additional FPs).
7. FPR certificate is marginal (expectation over exchangeable cleans), not a per-run cap; the single realized FP corresponds to 0.93% pooled.
8. Apple-silicon MLX q4 runs ~110 ms p50, roughly 2.5× the L4 p50 — consistent with the observed ~3.6 ms/layer per-forward dispatch floor on that runtime (q4/q8/bf16 latencies were near-identical there).
9. Robustness to adversarial evasion (an agent phrasing rogue steps to look benign) was not evaluated.
10. Context window: the base model's limit is 4,096 tokens; rendered StepShield trajectories (≤20 steps) fit within it. For longer traces, do not silently truncate — reject them or define and evaluate an explicit windowing policy first.
11. Operational guidance: route flags to human review or an evaluated downstream policy; do not let a flag trigger irreversible automated action. Pin the base-model and tokenizer revisions in deployment — verdict-token ids (12192/100275) are tokenizer-specific.

## License and attribution (NVIDIA Community Model License)

- Base model: `nvidia/Nemotron-Mini-4B-Instruct`, © NVIDIA Corporation, licensed under the **NVIDIA Community Model License** (v. Aug 23, 2024; copy distributed with the adapter as required by §1.2(c)).
- This adapter is a **Derivative Model** under §1.2.3 of that license. Modification notice (§2.2.1): *the distributed weights are LoRA adapter deltas trained by Doron Podoleanu on StepShield data; they modify the behavior of the base NVIDIA Model and are not original NVIDIA weights.*
- The adapter weights are licensed under the same NVIDIA Community Model License — **not** Apache/MIT (§2.1.4 forbids subjecting NVIDIA Models to open-source licenses). Only the companion repo's code is Apache-2.0.
- **Production-use restriction (§1.2): read before deploying.** The license grants production use only via an NVIDIA NIM runtime under an NVIDIA AI Enterprise subscription, except for models on NVIDIA's published PC/workstation and gaming exception lists. As of the list version we checked (v. Dec 13, 2024), **Nemotron-Mini-4B is not on the PC/workstation exception list**. Research and evaluation use is fine; for production, consult the current NVIDIA lists and licensing.
- Outputs restriction (§2.1.2): the license prohibits using the model or its outputs to develop or improve non-NVIDIA-derivative AI models without NVIDIA's written approval.
- "Nemotron" is used solely to identify the base model; no NVIDIA trademark license is granted or implied, and NVIDIA does not endorse this work.
- Use must be consistent with [NVIDIA's Trustworthy AI terms](https://www.nvidia.com/en-us/agreements/trustworthy-ai/terms/).
- Benchmark: StepShield, MIT License, © 2026 Gloria Felicia et al.
