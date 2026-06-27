# Brainstorm EXP-032
**Created**: 2026-06-10
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

- **SWA — Stochastic Weight Averaging (Izmailov et al. 2018, "Averaging Weights Leads to Wider Optima and Better Generalization", UAI 2018)** (background knowledge; implemented in torch core as `torch.optim.swa_utils` — `AveragedModel`, `update_bn` — NO new packages needed)
  Equal-weight average of SGD iterates collected AFTER the schedule has annealed to a modest constant LR; the average sits nearer the center of the flat basin than any single iterate, raising test accuracy by +0.2–0.6 on CIFAR-10 ResNets/WRNs (WRN-28-10: 96.79→97.02 in the paper). Two implementation requirements: (1) a small CONSTANT tail LR so the iterates actually sample the basin (at LR→0 they freeze and the average degenerates to the final iterate); (2) BN running statistics MUST be re-estimated for the averaged weights via a forward pass over the TRAINING loader (`update_bn`) — the canonical implementation does this with the augmented loader, which is exactly the EXP-029 lesson (stats must match training-time constants).
- **knowledge/README.md scan**: no SWA entry yet; muon/airbench/WRN/one-cycle entries unrelated. EXP-029's report already records that "SWA re-estimates with the AUGMENTED loader for this reason."
- **No new web searches needed** — the technique is core-PyTorch, the paper is canonical background knowledge, and the campaign's own EXP-011/EXP-029 measurements supply the decisive implementation guidance.

## Experimental History Review

- **Current best**: 96.71 @ 1990397 (distribution TOP; mean ≈96.57, σ ≈0.16); bar 96.81 ⇒ true effect ≥ +0.3 needed. 26 consecutive misses (007–031).
- **Closed axes**: capacity (both directions), recipe constants (bracketed local optimum), gradient noise (bracketed), batch+LR, schedule SHAPE (full-run: linear lost −0.50), augmentation pressure (both sides), init, activations, shortcut/head topology, optimizer geometry, kernel/pipeline throughput tiers, eval-time BN substitution (inverted, −10.93), FixRes tail, resolution (EXP-031: +46 cheap epochs converted at ZERO).
- **The one currency that pays**: converged-plateau LEVEL (max-statistic law — EXP-011/016/028/031). Nothing remaining cheapens a full-res step at clean numerics (EXP-021), so the only remaining attack is raising the plateau level directly.
- **The diagnosed near-miss**: EXP-011's EMA lost −0.25 *while improving test loss* — and its implementation copied the LIVE model's BN buffers onto averaged weights (exp-report-011: "per-buffer `.copy_()` for the ~40 BN buffers"). EXP-029 subsequently proved stats/weights mismatch is function damage (−10.93 in the extreme). EMA's failure therefore confounds two things: (a) smoothing-collapses-the-max (decay-weighted average tracking the moving iterate ACROSS the anneal mixes hot-trajectory weights), and (b) un-recalibrated BN stats. Canonical SWA differs on BOTH: it averages only POST-anneal basin iterates (no hot-weight mixing) and re-estimates BN with the augmented loader. This is the strongest "combine previous near-misses with new mechanistic understanding" candidate available.
- **Protocol assets**: 600s wall arithmetic + validated levers (eval thinning, 2× workers — EXP-031); composite phase-aware watchdog; σ calibration; per-segment profiles.

## Candidate Ideas

### 1. SWA tail: freeze the cosine at 85%, average iterates, eval the BN-re-estimated SWA model
**Summary**: Add `SWA_START_FRAC = 0.85`. The timed step changes by ONE line: `if progress >= SWA_START_FRAC: lr_now = lr_at(SWA_START_FRAC)` — i.e., the cosine is FROZEN where it stands at 85% (≈0.030), turning the final 15% into the canonical constant-LR SWA phase (no new tuned LR constant; the anchor is the schedule itself). After each epoch whose end falls in the SWA phase: update an eager `AveragedModel` (equal-weight running mean, uncharged — outside the timed loop), re-estimate its BN running stats with `torch.optim.swa_utils.update_bn` over the AUGMENTED train loader (forward-only, ~1.5–2s wall), and pass the SWA model to `evaluator.evaluate` for that epoch's single eval (once/epoch budget respected; pre-SWA epochs eval `base_model` as today). Training math before 85% is byte-identical to baseline.

**Reasoning**: Attacks the only paying currency — plateau LEVEL — with the strongest remaining external evidence (+0.2–0.6 on CIFAR ResNets in the canonical paper; mechanism = flat-minima centering, which raises the LEVEL, not transit speed). The campaign's own diagnostics de-risk the two known failure modes: BN re-estimation done the EXP-029-correct way (augmented loader), and no hot-weight mixing (only basin iterates, unlike EXP-011's EMA). Late-only intervention dodges the deferral law (zero early-heat/epoch/gradient-quality cost; payoff arrives exactly when the metric harvests). dt unchanged → ~139 epochs. Wall: +~21 epochs × ~2s ≈ +40s → ~535s < 600.

**Sources**: exp-report-011.md (EMA implementation + miss), exp-report-029.md (BN stats law; SWA re-estimation note), exp-report-031.md § Next Steps; Izmailov et al. 2018 (background); torch.optim.swa_utils (core).

**Estimated Effort**: medium — ~20 lines (constant, lr freeze, AveragedModel, per-epoch update_bn + eval switch), plus watchdog (baseline thresholds; wall projection updated).

**Risk Assessment**: (a) The frozen tail (constant 0.03 vs cosine→0) makes RAW tail iterates noisier — if SWA's averaging gain < the forfeited final anneal, the plateau lands BELOW baseline (graceful no-improvement; measured closure of the averaging axis). (b) 1-epoch snapshot spacing at small LR yields correlated samples — averaging still helps but less than paper cycles; bounded downside. (c) Max-statistic variance loss ≤0.03 (EXP-027). (d) Two semi-tuned anchors (0.85 start; freeze-at-cosine value) — a miss leaves interiors unbracketed; flag for analysis. (e) update_bn cost is uncharged forward-only statistics — same accounting class as EXP-029's recalibration and eval itself; documented honestly in the plan.

### 2. Pure-plateau SWA with the schedule untouched (cosine→0 preserved)
**Summary**: Same AveragedModel + update_bn machinery, but NO lr change — average the final-15% iterates of the unmodified baseline schedule and eval the SWA model in the tail.

**Reasoning / why weaker**: Zero schedule risk, but as LR→0 the iterates freeze, so the average ≈ late iterates with slight noise reduction — the basin is never SAMPLED, which is the mechanism SWA's gains come from. Expected +0.0–0.15 (sub-bar); essentially a safer, weaker version of Idea 1 that spends a loop to learn little the EXP-027 σ measurement doesn't already imply.

**Sources**: exp-report-027.md (plateau σ, max harvest ≤0.03); Izmailov et al. 2018 § constant-LR requirement.
**Estimated Effort**: low. **Risk Assessment**: very safe, very likely noise-band; weak information value.

### 3. Deeper-not-wider at near-matched dt: ResNet-26 (n=4) at 4× width
**Summary**: NUM_BLOCKS 3→4 (ResNet-26), widths unchanged 64/128/256 — +33% blocks/params; measured-dt gate decides viability (projected ~28–30ms → ~105 epochs).

**Reasoning / why weak**: The only unbracketed capacity direction is deeper-at-similar-width, but every capacity increase below ~139 epochs has lost (EXP-002/005/007: starvation; EXP-017: +26% params converged yet −0.28), WRN evidence says width>depth on CIFAR, and 105 epochs is squarely in the historically-losing range. Survives the constraint filter but contradicts two High-importance laws without a mechanism for why depth escapes them.

**Sources**: exp-report-017.md, exp-report-007.md; knowledge/README.md WRN row.
**Estimated Effort**: low (one constant + gate). **Risk Assessment**: high probability of repeating the starvation/deferral arithmetic; gate would likely kill it in ~90s (cheap), but the loop is better spent on Idea 1.

## Idea Evaluation

**Evidence strength**: Idea 1 has canonical published gains in the exact metric direction on CIFAR ResNets PLUS two in-project diagnostics showing precisely why the prior averaging attempt (EXP-011) under-delivered — no other remaining candidate has a measured, repaired failure mode. Idea 2's mechanism is self-defeating at LR→0; Idea 3 contradicts two High-importance laws. **Mechanism clarity**: Idea 1's causal path (basin sampling → averaged point deeper in flat minimum → higher converged LEVEL, harvested by the max over ~20 tail evals) is the sharpest available; its known costs (forfeited final anneal, sample correlation) are bounded and observable in the eval trail. **Expected impact**: paper-scale +0.2–0.6 brackets the needed +0.24-over-mean; Ideas 2–3 cap below the screen. **Risk profile**: Idea 1 fails gracefully (tail evals visibly below family ⇒ no-improvement with the averaging axis measured); no crash modes beyond a wall overrun already bounded (+40s, levers in reserve). **Feasibility**: ~20 lines, core-torch utilities, one run.

## Chosen Idea
**Selected**: Idea 1 — SWA tail (freeze cosine at 85%, equal-weight average, augmented-loader BN re-estimation, eval the SWA model)

**Why this idea**:
It is the only remaining candidate that (a) attacks plateau LEVEL — the one currency the max-statistic pays — (b) carries published gains of the required magnitude on matched models/datasets, and (c) repairs a diagnosed in-project failure: EXP-011's EMA averaged weights but copied live BN buffers, the exact stats/weights mismatch EXP-029 proved is function damage. All training before 85% of the budget is byte-identical to baseline, so the deferral law is structurally satisfied.

**Hypothesis**:
Freezing the LR at its 85%-progress cosine value (≈0.030) and equal-averaging the ~20 subsequent end-of-epoch iterates — with BN running stats re-estimated on the augmented loader before each eval — produces an SWA model whose plateau sits ≥ +0.25 above the raw-baseline plateau mean, i.e., best_test_acc ≥ 96.81, with dt/epochs unchanged (~22.4ms / ~139) and total wall ≤ ~545s. Falsifiable: if the SWA eval trail fails to climb above the baseline family within ~8 epochs of SWA start, the forfeited final anneal dominates the averaging gain and the axis closes with a measured number.
