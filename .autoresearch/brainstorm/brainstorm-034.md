# Brainstorm EXP-034
**Created**: 2026-06-09
**Goal**: goals/improve-cifar10-test-accuracy.md

## Web Search & Literature Review

- (Carried from brainstorm-033 — no new search needed; this is a direct refinement of the EXP-033 augmentation-cooldown result.) **YOLOX "close mosaic"** (disable strong aug for the final epochs → clean-distribution fine-tune) and the **openreview "Tradeoffs in Data Augmentation" study** (turning off detrimental aug recovers clean-distribution accuracy) are the grounding evidence. The new evidence is internal: EXP-033's own trajectory.
- knowledge/papers/trivialaugment.md (TA = aug ceiling, EXP-012).

## Experimental History Review

- **Current best = baseline 96.22% (EXP-012, commit 6c417a4)**; bar = 96.32. 34 experiments done; ~25 axes mapped, but the **augmentation-SCHEDULE axis is OPEN** (introduced by EXP-033).
- **EXP-033 (augmentation cooldown @ COOLDOWN_FRAC=0.15) = no-improvement, 96.10 (−0.12pp)** — but a near-miss with a clear refinement signal. Implementation (2nd CPU transform `train_tf_clean` + epoch-boundary `train_set.transform` swap + `aug_cooled` flag gating Cutout) WORKED: cooldown fired once at ep77/frac 0.85, throughput-neutral (90 ep, dt 8ms, params 4,299,866). Trajectory: full-aug to ep76=95.43 → clean-data tail climbed to **ep86=96.10 (best)** → declined to 96.01 by ep90. (reports/exp-report-033.md, logs/exp-log-033.md)
- **KEY RE-ANALYSIS of EXP-033**: `best_test_acc` is the MAX over epochs, so the post-ep86 decline did NOT cost the metric — 96.10 was already the captured peak. Therefore "shorten the window to avoid the decline" is NOT the lever (the decline is free). The real lever: the cooldown STARTED TOO EARLY (frac 0.85, ep77). Cutting strong aug at ep77 sacrificed ~5 epochs of productive strong-aug training during the mid-LR phase; the clean fine-tune lift (+0.67pp, 95.43→96.10) started from a base (95.43) that was lower than where a longer strong-aug schedule would have been. The full-aug baseline (EXP-012) reached 96.22 training with strong aug essentially throughout.
- **Untried within the open axis**: a LATER cooldown start (shorter window) so the clean fine-tune lifts from a HIGHER pre-cooldown base; and a MILDER cooldown (drop only TA, keep Cutout). Closed/avoid: re-running 15% (done); any global aug add/reduce (convergence-bound meta-pattern); compute-adders (epoch wall).

## Candidate Ideas

### 1. Later/shorter augmentation cooldown (COOLDOWN_FRAC 0.15 → 0.10)
**Summary**: Identical mechanism to EXP-033, but start the cooldown later: `COOLDOWN_FRAC = 0.10` (cooldown begins at frac 0.90 ≈ ep~81 instead of frac 0.85 ≈ ep77), giving ~9 clean-data epochs (matching the observed ~9-10-epoch climb-to-peak duration in EXP-033) while preserving ~5 more epochs of strong-aug training in the productive mid-LR phase. The clean fine-tune then lifts from a higher pre-cooldown base.

**Reasoning**: EXP-033 proved the cooldown mechanism produces a real, repeatable clean-data climb (+0.67pp over ~9 epochs) and is throughput-neutral. Its only shortfall was starting too early — sacrificing strong-aug training that the baseline shows is productive right up to the end. A later start keeps the strong-aug benefit longer, then applies the SAME clean alignment on top of a higher base. Rough extrapolation: EXP-033's full-aug model was 95.43 at frac 0.85; at frac 0.90 a full-aug model would be higher (~95.6-95.8, more annealed), and a comparable clean lift would land ~96.3-96.5 — plausibly clearing the 96.32 bar. The ~9-epoch window is chosen to match the observed time for the clean climb to peak (so the fine-tune completes within the budget). Compute-neutral, hierarchy/recipe-preserving, NOT a global regularizer change — sidesteps all closed-axis traps, exactly as EXP-033.

**Sources**: reports/exp-report-033.md (trajectory ep76-90), logs/exp-log-033.md; goal-learnings § Failed Approaches (EXP-033 entry, LIVE LEAD); YOLOX close-mosaic; train.py L28 (COOLDOWN_FRAC).

**Estimated Effort**: low — one-line constant change (0.15 → 0.10) on top of the (discarded) EXP-033 diff, which must be re-applied. ~15 lines total, identical to EXP-033 except the constant.

**Risk Assessment**: (a) Fewer clean epochs (~9 vs ~14) may produce a smaller lift if the climb needs the full duration → no-improvement. (b) The +0.67 lift may not be additive on a higher base (diminishing returns near the ceiling). (c) Run-to-run epoch variance (±0.2pp) could mask the effect. Worst case is a bounded no-improvement; no crash/invalid path. Safe.

### 2. Milder cooldown: drop only TrivialAugment, KEEP Cutout (at COOLDOWN_FRAC 0.10)
**Summary**: Same later-start window (0.10), but in the cooldown phase disable only TrivialAugment and KEEP Cutout(16) running. Tests whether retaining the label-preserving occlusion regularizer prevents any clean-tail over-sharpening while still removing the strong photometric/geometric distribution shift (TA's solarize/posterize/shear) that drives the train-test gap.

**Reasoning**: TA is the strong DISTRIBUTION-SHIFTING aug (its ops move pixel statistics far from the clean test images); Cutout is a mild, label-preserving occlusion that does NOT shift the global distribution. The train-test-gap mechanism is mostly about TA, so dropping only TA may capture most of the alignment benefit while Cutout's regularization keeps the clean tail from saturating. Isolates which aug is responsible for the cooldown effect.

**Sources**: knowledge/papers/trivialaugment.md; goal-learnings (Cutout-16 interior optimum EXP-013/021); reports/exp-report-033.md (Unexplored Avenues).

**Estimated Effort**: low — drop the Cutout gate, keep TA-only swap.

**Risk Assessment**: Keeping Cutout may retain enough distribution shift (zeroed patches are themselves off-distribution vs clean test) to blunt the alignment benefit → smaller lift. Confounds the EXP-033 comparison (two variables differ: window AND which augs). Medium-low value vs candidate 1.

### 3. BN-recalibration-only tail (freeze weights, recompute BN running stats on clean data)
**Summary**: Instead of training on clean data in the tail, after the normal full-aug run, run a few hundred forward-only passes over clean (un-augmented) images in train() mode to recompute BN running mean/var on the clean distribution (cf. SWA's `update_bn`), then evaluate. Isolates the BN-distribution-alignment component from the weight-fine-tuning component.

**Reasoning**: Part of the cooldown's hypothesized benefit is BN running stats re-aligning to the clean distribution. This tests that component alone, cheaply, without giving up any training epochs.

**Sources**: SWA update_bn (knowledge/papers/swa.md); reports/exp-report-033.md (Unexplored Avenues).

**Estimated Effort**: medium — a forward-only BN-update loop at the end, careful not to call evaluate() more than once/epoch.

**Risk Assessment**: BN-only recalibration on a net already trained with augmented BN stats often gives a tiny effect (the augmented stats are close to clean for mean/var); likely a small null. Lower expected impact than candidate 1.

## Idea Evaluation

**Candidate 3 (BN-only)** isolates a clean mechanism but has the lowest expected impact — BN mean/var under TA are not wildly off the clean distribution (Normalize already centers the data), so the recalibration delta is likely small. Useful as a diagnostic, not a bar-clearer.

**Candidate 2 (drop-only-TA)** is a reasonable variable-isolation experiment but changes TWO things vs EXP-033 (window AND aug set), muddying attribution, and keeping Cutout may retain enough off-distribution signal to blunt the benefit. Better as a follow-up once the window is tuned.

**Candidate 1 (later/shorter cooldown, 0.10)** is the strongest: it is the single highest-evidence refinement directly implied by EXP-033's own data, changes exactly ONE variable (the start fraction) for clean attribution, has the clearest mechanism (preserve productive strong-aug training → clean fine-tune from a higher base), the highest expected impact (the rough extrapolation lands near/above the 96.32 bar), the lowest effort (one-constant change), and the safest failure mode. It keeps the augmentation-schedule axis — the only open axis — moving with a principled next point.

## Chosen Idea
**Selected**: Later/shorter augmentation cooldown — COOLDOWN_FRAC 0.15 → 0.10 (cooldown starts at frac 0.90, ~9 clean-data epochs), same mechanism as EXP-033 otherwise.

**Why this idea**: It is the principled one-variable refinement that EXP-033's trajectory directly motivates. EXP-033 proved the cooldown mechanism works and is throughput-neutral but started too early, sacrificing productive strong-aug training; starting later applies the same clean fine-tune from a higher base. Lowest effort, cleanest attribution, highest expected impact among the candidates, safe failure mode, and it advances the only open axis.

**Hypothesis**: Starting the augmentation cooldown at frac 0.90 (COOLDOWN_FRAC = 0.10) instead of 0.85 will raise `best_test_acc` above EXP-033's 96.10 — plausibly clearing the 96.32 bar — at an unchanged ~90 epochs / dt ~8ms / 4,299,866 params, because the model enters the clean fine-tune from a stronger pre-cooldown base (more strong-aug training retained) while still getting ~9 clean-data epochs for the distribution-alignment climb. If the lift shrinks on the higher base, expect a result between 96.10 and ~96.3 (no-improvement but informative for locating the cooldown optimum).
