# Methodology: leakage-free threshold calibration for step-level detectors

This document specifies the calibration protocol behind the certified operating points in this repository. The model, its training, and the scoring pass are conventional; the part that changed the reported FPR by a factor of ~20 at nearly constant recall was **how the decision threshold was chosen**. We believe the protocol is the reusable contribution: it is cheap, distribution-free, and directly applicable to any score-thresholded detector evaluated on a paired benchmark.

Results status: everything below is **held-out with respect to training, and not benchmark-external**. The holdout never contributed to weight training or checkpoint selection; it was touched once by the frozen legacy evaluation and then used, under the leakage-controlled protocol of §2, for threshold calibration. No prospective (blind) evaluation exists, and §6 documents why none can be produced from this benchmark: we pre-specified the blind protocol and found it unsatisfiable — StepShield retains no rogue-containing data pool untouched by development (the untouched remainder is clean-only; §6). External validation on a different corpus is the correct future test of these numbers.

Terminology: throughout, "certified" means the threshold-selection procedure carries an explicit finite-sample guarantee under stated assumptions and was verified mechanically leakage-free in external adversarial LLM review (three rounds, Kimi V3, commissioned by the author). It does not mean externally audited.

## 1. The failure that motivated it

The standard recipe — tune θ on a validation split, freeze, evaluate once on the holdout — failed in a way that is worth internalizing:

- **Setup.** Frozen LoRA classifier emits per-step p(BAD); trajectory is flagged at the first step with p(BAD) > θ. θ was frozen at 0.064 by a pre-registered rule on validation (geometric midpoint of the widest high-accuracy plateau; the rule was amended once, from a zero-FP rule to the accuracy-plateau rule, after observing validation behaviour — amendment documented in the build log and disclosed in the paper's pre-registration appendix), before any holdout metric was computed.
- **Result.** Validation balanced accuracy 90.0% → holdout 89.4%, **but** with FPR 19.4% (21/108 false positives). A post-hoc sweep (report-only) showed θ≈0.9 would have delivered 97.2% accuracy at 0.9% FPR from the *same scores*.
- **Diagnosis.** The validation split (train-split provenance) is a systematically *harder discrimination problem* than the holdout, on both sides of the boundary. On validation, raising the threshold was a genuinely bad trade: at θ=0.90 the paired-clean FPR is still 5.0% (the val cleans' hard tail reaches 0.99) while recall collapses from 93.3% to 75.0% — so the pre-registered accuracy-plateau rule correctly settled low *for validation*. The holdout has the opposite shape: only 1/108 cleans scores above 0.9 and recall at θ≈0.9 stays at 95.4%, so a high threshold there is nearly free — and the transferred low θ paid 19.4% FPR for nothing. (All figures computable from the shipped score files `results/v2_holdout_step1100.json` and `results/v2_val_step1100.json`.) Recall transferred fine; the FPR did not.

Two properties make this failure mode dangerous: it is invisible in validation metrics, and it is *asymmetric* — a threshold that transfers badly usually wrecks exactly one error rate. Any pipeline that ships a fixed θ chosen on data from a different generator than deployment traffic inherits it.

The naive fix — re-tune θ on the holdout and report the best row — is threshold-level leakage: the 97.2% figure above is selection-biased and we do not claim it. The protocol below gets the benefit of holdout-provenance calibration *without* letting any decision be made by a threshold that saw that trajectory.

## 2. Protocol

**Data.** The StepShield holdout: 108 scenario pairs, each a rogue trajectory and its benign twin generated from the same scenario (216 trajectories). Model weights and per-step scores are frozen inputs (`results/v2_holdout_step1100.json`); nothing in this protocol touches the model.

**Split unit = the pair, never the trajectory.** All splits operate on the 108 pairs, category-stratified, seed 7. A clean trajectory and its rogue twin never straddle a calibration/report boundary — twins share a scenario template, so splitting them would leak template-specific score structure from calibration into report. We verified all 108 pair templates are mutually distinct (no cross-pair duplication), which is what the exchangeability assumption below rests on.

**4-fold cross-calibration.** For each fold k: thresholds are computed on the *other three folds'* trajectories (the calibration set), then applied to fold k (the report set). Pooling the four report folds yields exactly one decision per trajectory, each made by a threshold that never saw that trajectory or its twin. A unit test asserts the no-straddle and decide-once invariants; it must pass before any metric is read.

**Thresholds from calibration CLEANS only, by conformal order statistics.** For a target false-positive rate α, with n_c calibration clean scores s₍₁₎ ≤ … ≤ s₍n_c₎:

- *order-statistic rule:* θ = s₍k₎ with k = ⌈(n_c+1)(1−α)⌉;
- *conformal-max rule* (α → tightest): θ = s₍n_c₎, giving E[FPR] ≤ 1/(n_c+1) under exchangeability of clean scores.

Rogue scores are never consulted, so the recall estimate at the resulting θ is untouched by selection. The guarantee is distribution-free and *marginal* (an expectation over splits, not a per-split bound). Guarantees are quantized to multiples of 1/(n_c+1): the category-stratified split yields 78–84 calibration cleans per fold, so the tightest per-fold certificates are 1.18–1.27%; a 0.5% certificate would need n_c ≥ 199 cleans. We state this rather than implying arbitrarily low FPR is reachable.

**Pre-commitment.** Before evaluating any report fold we fixed, in writing: the primary rule (conformal-max), the α grid {0.025, 0.05, 0.10}, the seed, the pooled-micro headline aggregation, the realized-FPR reporting language, and the rule that no grid point gets promoted to primary after the fact. That last clause bound: α=0.025 outperformed the primary on the report folds (same single FP, +2.8pp recall) and is reported as a curve point, not as the headline.

**Deployed threshold.** The shipped θ = 0.946597 is the conformal-max over *all* 108 cleans: forward certificate E[FPR] ≤ 1/109 = 0.917% on future exchangeable cleans. Its in-sample FPR is 0 *by construction* and is not evidence. Its recall (90.7%, 98/108) is fully leakage-free — the threshold computation never saw a rogue score. Note the CV recall (91.7%) and deployed recall are computed on the same 108 rogues: correlated estimates, not independent confirmations; since θ_deployed ≥ every fold threshold, CV recall upper-bounds deployed recall in expectation, and the observed 91.7 vs 90.7 is that consistency check passing.

**What probability calibration cannot do.** Temperature scaling and isotonic regression were fit for comparison (T=1.25). Both are monotone score transforms: they relabel the threshold axis but produce the identical ROC, so they cannot change any achievable (FPR, recall) pair. They were never applied to any decision. Quantile/conformal calibration is the correct tool when the quantity you need to control is an error *rate*, not a probability.

## 3. Results under the protocol

Pooled out-of-fold, 216 decisions (source of truth: `results/calibration_report.json`):

| Rule | Fold thresholds | Realized FPR | Recall | Acc | EIR₃ |
|---|---|---|---|---|---|
| conformal-max (primary) | .9466 / .8808 / .9466 / .9466 | 0.93% (1/108) | 91.7% (99/108) | 95.4% | 0.990 |
| α=0.025 | .9466 / .8808 / .8808 / .8808 | 0.93% | 94.4% | 96.8% | 0.971 |
| α=0.05 | .8808 / .8176 / .8520 / .8808 | 3.70% | 96.3% | 96.3% | 0.971 |
| α=0.10 | .5622 / .3486 / .3486 / .5622 | 9.26% | 98.1% | 94.4% | 0.953 |

Realized FPR sits at or below the nominal target at every point (conformal conservatism, expected). The per-fold forward certificates for the primary evaluate to E[FPR] ≤ 1.27% (worst fold); the realized 1 FP fell in one of the 30-clean report folds (1/30 ≈ 3.3% in isolation — priced in by the marginal 1/(n_c+1) bound; the guarantee is an expectation, not a per-fold cap). A secondary single 50/50 split (54/54 pairs) gives 88.9% recall at 0.00% realized FPR under the same rules — consistent, lower-powered.

The certified α=0.10 point **strictly dominates** the legacy transferred point (98.1% recall at 9.26% vs 19.4% FPR) — identified in external review, not by us. McNemar between the calibrated primary and the legacy rule on the same 216 decisions: 20 vs 7 discordant, exact two-sided p = 0.019. The correction is real but mixed: −18.5pp FPR, −6.5pp recall, +6.0pp accuracy, EIR₃ 0.896 → 0.990.

## 4. Assumptions, threats to validity, and honest labels

1. **Exchangeability.** The conformal guarantee needs future cleans exchangeable with the calibration cleans. Within the benchmark this holds by construction (verified: no duplicate templates across pairs). In deployment it will not hold exactly; θ should be re-calibrated on ~200+ cleans from actual traffic (which also unlocks a real ≤0.5% certificate).
2. **Analyst prior exposure.** The analyst had seen full-holdout summary metrics (from the legacy evaluation) before designing the calibration. The *mechanical procedure* is leakage-free — no decision threshold saw its own trajectory or twin — but the framing is honestly a **calibrated-operating-point claim**, not a fresh single-shot claim. The two are distinguished everywhere in this repo. A fresh untouched holdout would upgrade the claim; the benchmark provides only one, and §6 shows no substitute pool exists within it.
3. **Within-template generalization.** Independent of calibration: holdout scenarios reuse train templates (task-header word-Jaccard p50 = 1.0; step streams novel, difflib ~0.10, median 0.096). Fine-tuned rows carry an advantage untrained baselines do not. All our numbers are within-scenario-distribution. Note this contradicts the benchmark's documented design: StepShield's paper states the held-out set was generated "from incident templates that were never used during training or pattern development" and is "drawn from novel templates" (its README's "zero overlap with training data" is true at the trajectory level; the template-level novelty claim is what our measurement contradicts). The measurement: word-set Jaccard between each holdout scenario's title+description and its nearest train scenario (median 1.0) and `difflib.SequenceMatcher` over rendered step streams (~0.10; median 0.096, vs 0.071 for randomly paired trajectories, and 0 of 3,056 rendered holdout steps appear verbatim in train), computable from the public data alone by running `scripts/template_overlap.py`. The observation matters exactly and only for detectors fine-tuned on the train split; detectors trained on an external corpus are unaffected.
4. **Curve discipline.** The certified curve is the four discrete points; interpolations are not certified. The descriptive 61-knot sweep (`results/descriptive_roc_216.json`) is in-sample and selection-biased as a threshold menu — plot it, do not deploy from it.
5. **Review scope.** Kimi V3 review rounds 1–3 (design; line-by-line re-derivation of every number, which caught one transcription error in a prompt and one per-fold-certificate arithmetic slip, both reconciled; evidence-bundle verification) concluded "certified: leakage-free" *for the threshold-selection procedure*. The reviewer is an LLM commissioned and prompted by the author — external to the analysis, not independent in the institutional sense; the certification claim is scoped to the mechanical protocol and the committed artifacts, nothing broader.
6. **No benchmark-external validation.** As a general methodological matter in detection work — not specific to this project — results produced inside a development loop, however carefully held out, can differ substantially from results on data nobody in the loop has seen. Our numbers are held-out with respect to training with leakage-controlled calibration, but no benchmark-external or prospective validation exists (§6 explains why none can be produced here). Weight them accordingly until an external-corpus evaluation is run.

## 5. Applying this elsewhere

The protocol needs: (a) a frozen scorer, (b) a paired or clustered eval set with the cluster structure known, (c) ~100+ cleans. Then: split by cluster; calibrate on cleans only via order statistics; pre-commit rule, grid, seed, and language; report pooled out-of-fold metrics plus the deployed conformal-max θ with its 1/(n_c+1) certificate. Total compute on top of scoring: seconds. `scripts/calibrate_report.py` is self-contained and reproduces every number in this document from the committed scores file.

## 6. The blind protocol we specified — and why it cannot be executed on this benchmark

The strongest upgrade available to the claims in this document would be an evaluation on a slice nobody in the development loop has ever seen. We pre-committed a full protocol for it:

1. **Frozen inputs.** Model weights (`v2-step1100`) and the deployed threshold θ = 0.946597 as committed in this repository — no retraining, no re-calibration, no per-slice tuning of any kind.
2. **Single pass.** The blind slice is scored once, under the same screening-mode semantics (all-OK history, bf16); the first-crossing rule decides each trajectory.
3. **Reporting.** Realized FPR, recall, and EIR₃ on the blind slice are reported verbatim as **PROSPECTIVE**, next to the forward certificate they test (E[FPR] ≤ 0.917% under exchangeability). If the blind slice's clean distribution is not exchangeable with the holdout cleans, a certificate miss is an informative outcome and is reported as such, not explained away.
4. **No promotion.** Whatever the blind numbers are, the held-out numbers above keep their label; the two sets are never mixed in one table without labels.

**Outcome: the protocol is unsatisfiable on StepShield.** Executing step 2 requires a qualifying slice, and an inventory of the benchmark's data shows none exists:

- the **train split** — 639 rogue/clean scenario pairs, i.e. 1,278 trajectories — was consumed in full by weight training (with a 90/10 pair-level train/val carve-out, both sides of which drove training and checkpoint selection);
- the **generated-benign corpus** is the benchmark's own generated-benign data (2,514 trajectories in `stepshield/data/generated_benign`, read by `scripts/prep_data.py`), of which this project consumed 700 — 400 as training material and 300 as validation threshold-tuning material, development-exposed by that use. The remaining ~1,814 were never touched by development, but they are clean-only by construction: they can yield a realized FPR, not the realized recall or EIR₃ the pre-committed protocol requires;
- the **216-trajectory holdout** was used once for the frozen legacy evaluation and then for the threshold calibration of §2.

No rogue-containing pool remains that development did not touch — the benchmark's only rogue-containing held-out pool is the 216-trajectory holdout, already spent — so no evaluation run from this benchmark can satisfy the pre-committed protocol, and none can be labeled PROSPECTIVE without the label being false. We regard establishing this as a useful result in its own right, not a gap in ours: **a benchmark with a single public holdout cannot support a prospective claim by any detector fine-tuned on its train split, once that holdout has been used even once** — and the time to check for an untouched pool is *before* development starts, because development spends the pool irreversibly. Benchmark authors can dissolve the problem by shipping a second, sequestered holdout (ideally cross-template; see §4, item 3) or a scenario generator; consumers of any single-holdout benchmark should assume the prospective rung is unreachable and plan external validation from day one.

The consequence for this document's claims is stated in the results-status note at the top: everything here is held-out with respect to training under leakage-controlled calibration, nothing here is benchmark-external, and the prospective upgrade must come from a different corpus. That external validation is future work; no part of this repository presents it as an existing result.
