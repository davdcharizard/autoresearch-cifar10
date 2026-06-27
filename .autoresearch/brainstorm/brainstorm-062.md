# Brainstorm EXP-062
**Created**: 2026-06-11
**Goal**: goals/maximize-cifar10-test-accuracy.md

<!-- This file is focused on IDEATION only.
     Goal statement, primary metric, direction, hard constraints, and verification criteria
     live in the goal file (see pointer above). Baseline lives in experiment-indices/{slug}.tsv.
     Do not duplicate those fields here — always point to the source of truth. -->

## Status declaration (required by exp-report-061 Next Steps)

This brainstorm attempts **deeper literature excavation** (option (a) of exp-report-061's three honest moves) and it succeeded: the sweep surfaced a candidate that is genuinely absent from the 61-experiment record and is **not a re-measurement** — Schedule-Free SGD substitutes the entire anneal-plus-implicit-averaging mechanism rather than adding a smoother on top of it (the distinction EXP-011/EXP-032 closed), and its published evidence is the first that holds at ANY training horizon, the regime property every prior external import (0-for-17) lacked.

## Web Search & Literature Review

- **The Road Less Scheduled — Schedule-Free optimization** (arXiv 2405.15682, Defazio et al., Meta FAIR, NeurIPS 2024; https://arxiv.org/abs/2405.15682; reference impl https://github.com/facebookresearch/schedule_free)
  Replaces the LR schedule entirely: train at constant LR (after warmup) with three coupled sequences — z (SGD iterate), x (online weighted average of z), y = (1−β)z + βx (the point where gradients are evaluated). Eval happens at x. Theory unifies scheduling and iterate averaging; empirically on **CIFAR-10 with SGD** the x-curve matches or EXCEEDS the endpoint of every cosine schedule length simultaneously — i.e., at each wall-clock horizon the averaged point is at-or-above what the optimally-keyed cosine would have delivered by then. No new hyperparameters beyond (lr, β=0.9 default); warmup recommended. KEY CAVEAT (repo README): BatchNorm running stats are accumulated at y during training; before each eval they must be refreshed at x (forward-only passes over ~16–50 train batches with optimizer in eval mode) — exactly the EXP-032 `update_bn` machinery this project has already validated, on the EXP-025/033 second-persistent-loader pattern.
- **CIFAR-10 speedrun frontier re-check 2025–2026** (https://github.com/KellerJordan/cifar10-airbench; https://github.com/hiverge/cifar10-speedrun; https://www.hiverge.ai/blog/cifar-speedrun)
  The 2025 record progression (2.73s → 2.59s → 1.98s for 94%) is entirely Muon-optimizer engineering (vectorized NS updates) — speed at a fixed 94% target, no new accuracy mechanism. Muon's per-step geometry is measured-closed here (EXP-028: transit gain decays to zero at the plateau, basin slightly worse). Nothing else new in the lineage since the EXP-060/061 screens (data filtering accuracy-neutral, lookahead averaging-closed). The external frontier remains empty EXCEPT the schedule-free family, which the prior sweeps never covered because it is an optimizer-SCHEDULE construction, not a speedrun component.

## Experimental History Review

- Current best: 96.71 @ 1990397 (EXP-006 recipe); family mean ≈ 96.57, σ ≈ 0.16 (EXP-027); bar 96.81 = mean +1.5σ; 55 consecutive closures since.
- Every catalogued axis is measured-closed (goal-learnings): heat ± and schedule FAMILY (cosine vs linear, EXP-016), warmup, momentum/noise bidirectional, batch level+schedule, WD global ± and per-layer fc bracket, LS flat, augmentation dose+type+order+taper, loss (4 pathways), init both directions, BN constants two-sided, averaging (EMA EXP-011, SWA EXP-032), ensembles, width lattice, depth total + per-stage allocation bidirectional (EXP-017/061), precision both directions, step-time engineering, per-sample treatments.
- **The schedule axis closure is FAMILY-INTERNAL** — every schedule probe (peak ±, warmup, cosine→linear, SWA-frozen tail) varied the anneal's shape or readout while keeping "anneal to ~0 then harvest the cold plateau" as the structure. EXP-032's key learning was that the time-keyed cosine ALREADY performs implicit iterate averaging — which is precisely the claim Schedule-Free formalizes and then EXCEEDS by decoupling the averaging from the LR decay. No experiment has ever run the converse: drop the anneal, train hot forever, let explicit averaging do all convergence work, with gradients at the y-interpolation. That trajectory is not a re-parameterization of any measured run — z never anneals, so the underlying SGD path visits different basins than any cosine run.
- Max-statistic law (project-insights): best-over-checkpoints rewards converged-plateau LENGTH and LEVEL. The cosine's plateau is structurally confined to the last ~10% of the budget (~10–14 near-peak evals). Schedule-Free's x-curve is smooth and monotone-ish from mid-run onward — if the published CIFAR shape transfers, the back HALF of the run evals near the ceiling (~60+ draws), a plateau-length mechanism the cosine family cannot supply at any setting.
- Relevant precedents/warnings: Muon (EXP-028) — per-step optimizer gains decayed to zero at plateau (but its claim was transit speed; Schedule-Free's claim is endpoint LEVEL at every horizon); absorption law 0-for-17 (but it covers regularizers/aug/modules under lighter-aug anchors — Schedule-Free is schedule geometry, evaluated in the paper across full-length training, not a short-budget speedrun trick); EXP-029 BN-constants trap (stats must match the evaluated weight point — directly addressed by the documented x-refresh).

## Candidate Ideas

### 1. Schedule-Free SGD (full anneal→averaging substitution, eval at x)
**Summary**: Replace the time-keyed cosine + plain SGD with a hand-implemented Schedule-Free SGD (no new packages; ~40 lines of foreach tensor ops in train.py): keep the linear time-keyed warmup to PEAK_LR 0.4 over the first WARMUP_FRAC, then HOLD lr constant to budget end; maintain z (SGD iterate) and x (weighted online average); compute each training forward/backward at y = (1−β)z + βx with β = 0.9; nesterov-free (β is the interpolation constant — Schedule-Free replaces momentum); selective WD unchanged (applied at y per reference impl). Eval each epoch at x: swap x into `base_model`, refresh BN running stats at x with forward-only passes over ~16 batches from a second persistent train loader (EXP-032 update_bn pattern, uncharged, outside the timed loop), call `evaluator.evaluate`, restore training state. torch.compile + warmup unchanged (graph identical; optimizer is eager exactly as today).

**Reasoning**: (1) Published evidence is regime-RELEVANT in the exact dimension all 17 failed imports lacked: the paper's CIFAR-10 SGD result holds at every horizon simultaneously, so it does not presuppose a fixed-epoch budget — at our 300s horizon the claim directly predicts x at-or-above the optimally-keyed cosine endpoint, with margins ~0.5–1.0pp at longer horizons in the paper's Pareto figures (passes the +0.3pp effect-size screen — the first candidate since EXP-060 to do so on published numbers). (2) Independent second mechanism: the smooth monotone x-curve converts the max-statistic from harvesting ~10 cold-tail evals to harvesting ~60+ near-ceiling evals — plateau LENGTH, the residual the cosine family structurally cannot produce. (3) Not an interpolation of measured nulls: EMA/SWA averaged points GENERATED BY an annealed process (readout change); this changes the GENERATING process (z never anneals; gradients at y ≠ any prior trajectory). (4) Laws check: gradient-noise scale unchanged (same batch, z-step is plain SGD); numerics-equivalent (same compiled graph/kernels, fp32 eager optimizer math like today); no deferral component (x is competitive from early on); heat law closures are anneal-family-internal and do not price a schedule-free trajectory.

**Sources**: arXiv 2405.15682; github.com/facebookresearch/schedule_free (BN caveat + impl); EXP-032 (update_bn machinery + the implicit-averaging reading this experiment directly tests); EXP-011/028/029 (precedent warnings); goal-learnings § heat/noise/max-statistic entries.

**Estimated Effort**: medium — optimizer math ~40 lines + eval-time x-swap/BN-refresh plumbing + CPU sanity (sequence algebra unit test) + GPU probe to price the extra foreach ops (expect +0.1–0.3ms) and the per-epoch BN refresh wall cost (expect +0.3–0.5s/epoch; family runs ~480s vs 600s cap — must verify ≤ 600s, mitigation: 8-batch refresh).

**Risk Assessment**: (a) lr selection without a tuning budget — the paper notes schedule-free optimal lr is often ≥ the scheduled peak; we anchor at the certified 0.4, accepting one-draw risk in both directions (branches pre-registered). (b) BN-at-x mismatch if the refresh is too small — bounded by EXP-029's measured failure signature (test_loss explosion, instantly diagnosable). (c) The hot constant-lr z-path may lose basin quality the averaging cannot repair (the EXP-016 linear-anneal failure generalized) — that outcome closes the schedule axis at the FAMILY level, a real result. (d) Worst case is a clean no-improvement; no crash/invalid mechanism beyond standard infra.

### 2. Ghost BatchNorm (virtual batch 128 inside the 512 step)
**Summary**: Compute BN statistics over 4 independent 128-sample groups per 512 batch (reshape-based, compile-friendly), restoring the original recipe's BN-noise level while keeping the 512-batch gradient.

**Reasoning**: The one noise channel never isolated — EXP-012/022/059 moved GRADIENT noise, EXP-038/039 moved eval-constant freshness; train-time BN-stat noise at batch 512 is ~4× smoother than the 128-batch regime the He recipe was born in. Hoffer et al. 2017 report generalization gains from ghost BN at large batch.

**Sources**: arXiv 1705.08741; EXP-038/039 (adjacent closures); goal-learnings regularization-dose entries.

**Estimated Effort**: low-medium (BN module surgery + compile probe).

**Risk Assessment**: Fails two standing screens: it is a pressure-INCREASING change on a dose-saturated regularization axis (EXP-013 law: any diversity/noise-adding change is pressure regardless of framing — four points all negative above the optimum), and its anchor evidence is light-aug fixed-epoch (absorption, 0-for-17). Predicted −0.2 to −0.5.

### 3. Plateau-length micro-harvest (compressed cosine completing at p≈0.9 + cold tail)
**Summary**: Complete the anneal early and hold lr ≈ 0 for the last ~10% to manufacture extra converged evals for the max-statistic.

**Reasoning**: Directly targets the metric's max structure within legal bounds.

**Sources**: EXP-016 (plateau-manufacture reading); EXP-027 (plateau scatter); exp-report-061 Next Steps (pre-assessment).

**Estimated Effort**: low.

**Risk Assessment**: Pre-assessed in exp-report-061: converged-plateau evals at lr≈0 are near-identical draws, not independent ones — residual harvest ≤ +0.03, fails the +0.3pp screen by an order of magnitude; the cold tail also flirts with the EXP-033/055 tail-pressure law (nothing may stop moving before budget end). Re-measurement of closed structure.

## Idea Evaluation

**Evidence strength**: Idea 1 dominates. It is the only candidate whose published evidence (NeurIPS 2024, CIFAR-10, SGD, every-horizon Pareto) passes the +0.3pp effect-size screen, and the only one not pre-priced negative by a standing law. Idea 2's anchor is light-aug fixed-epoch (absorption) AND it adds pressure to a closed dose curve — double negative prior. Idea 3 fails the screen by pre-assessment.

**Mechanism clarity**: Idea 1 has two articulable mechanisms — (i) endpoint level: explicit averaging decoupled from LR decay provably dominates the schedule family in the paper's theory, and (ii) plateau length: a monotone x-curve gives the max-statistic ~6× more near-ceiling draws. Idea 2's mechanism (BN noise as regularizer) is exactly the mechanism class measured saturated. Idea 3's mechanism is measured ≤ +0.03.

**Expected impact**: Idea 1 if the paper's CIFAR margin transfers: +0.3–1.0 over the cosine endpoint plus harvest upside — clears the bar. Ideas 2–3: negative to negligible.

**Risk profile**: Idea 1 fails gracefully (no-improvement with a family-band or below-band read; either closes the schedule axis at the FAMILY level — strictly new information). Its engineering risks (BN-at-x, lr anchor) have measured failure signatures and validated mitigations in-record.

**Feasibility**: All three fit one loop. Idea 1's extra plumbing (x-swap eval, BN refresh, second loader) reuses validated patterns from EXP-025/032/033.

**Honesty check (not a re-measurement)**: EXP-011/032 measured *averaging as a readout of an annealed trajectory*. EXP-016 measured *a different anneal shape*. Idea 1 changes the *generating process* — no anneal exists, the SGD iterate z runs hot to budget end, and gradients are evaluated at an interpolation point no prior run used. The closest closure (EXP-032) explicitly left this converse untested: it concluded the anneal already does implicit averaging; this experiment asks whether explicit averaging WITHOUT the anneal does it better, which is the paper's exact theorem-backed claim.

## Chosen Idea
**Selected**: Schedule-Free SGD (full anneal→averaging substitution, eval at x)

**Why this idea**:
It is the only construction surfaced by the deepest remaining honest move (literature excavation) that (1) is absent from the 61-experiment record, (2) passes the +0.3pp effect-size screen on published, regime-relevant (any-horizon) CIFAR-10 SGD evidence, (3) violates no standing law — noise-neutral, numerics-equivalent, deferral-free, outside the anneal-family heat closures — and (4) attacks the max-statistic through a plateau-length mechanism the cosine family structurally cannot supply. Every alternative is either pre-priced negative (Ghost BN) or pre-measured negligible (plateau micro-harvest).

**Hypothesis**:
Replacing the time-keyed cosine + SGD with Schedule-Free SGD (warmup to lr 0.4 then constant; β = 0.9; eval at the averaged point x with BN stats refreshed at x) will hold best_test_acc at-or-above the cosine endpoint at the 300s horizon while extending the near-ceiling eval plateau from ~10 to ~60+ draws, yielding best_test_acc ≥ 96.81 (branch i). Pre-registered alternates: (ii) family band [96.41, 96.73] → the anneal-equivalence of EXP-032 extends to full substitution — the schedule axis closes at the FAMILY level, measured; (iii) < 96.41 → constant-lr hot training loses basin quality that averaging cannot repair (EXP-016's mechanism generalizes; same closure, stronger form); (iv) BN-mismatch signature (test_loss ≥ 0.3 with depressed accuracy at converged plateau) → engineering retry per protocol, not a research verdict.
