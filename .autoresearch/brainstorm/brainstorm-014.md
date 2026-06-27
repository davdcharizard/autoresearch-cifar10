# Brainstorm EXP-014
**Created**: 2026-06-10
**Goal**: goals/maximize-cifar10-test-accuracy.md

<!-- This file is focused on IDEATION only.
     Goal statement, primary metric, direction, hard constraints, and verification criteria
     live in the goal file (see pointer above). Baseline lives in experiment-indices/{slug}.tsv.
     Do not duplicate those fields here — always point to the source of truth. -->

## Web Search & Literature Review

- **Why Warmup the Learning Rate? Underlying Mechanisms and Improvements (NeurIPS 2024)** (https://arxiv.org/abs/2406.09405)
  Across architectures/datasets (incl. CIFAR-10, SGD): the PRIMARY benefit of warmup is enabling a larger peak LR to be tolerated; other benefits are marginal. With cross-entropy loss and standard inits, even the tolerance benefit is reduced. Common practice is warmup = 1–5% of total training. Implication: our WARMUP_FRAC=0.15 (15% of the time budget) is 3–15x longer than needed to protect a peak of 0.4 — especially since EXP-010 showed even peak 0.6 does NOT destabilize this model (it merely defers progress). The excess warmup time is budget spent at below-peak LR for no stability payoff.
- **cifar10-fast (davidcpage)** (https://github.com/davidcpage/cifar10-fast — knowledge/README.md References)
  The reference budget-matched recipe ramps LR to peak over ~4 of 24+ epochs (~15–17%) but against a piecewise-LINEAR decay; our cosine anneal holds near-peak LR longer after the peak, so a shorter ramp is the cheaper way to raise the LR-time integral.
- **Knowledge base** (.autoresearch/knowledge/README.md)
  No entry directly addresses warmup length or weight-decay dose at fixed wall-clock; super-convergence (1708.07120) used symmetric up/down cycles but is a fixed-iteration result (does not transfer per project-insights Medium).

## Experimental History Review

- Current best: **96.71** (EXP-006 recipe @ 1990397) — compiled 4x-wide ResNet-20, batch 512, peak 0.4, WARMUP_FRAC 0.15, WD 5e-4, LS+TA+RE. Eight consecutive no-improvements since (EXP-007 through EXP-013).
- Axes with measured boundaries (goal-learnings § Failed Approaches + Patterns):
  - **Capacity**: closed bidirectionally (wider starves: EXP-002/005/007; shallower converges worse: EXP-008).
  - **Augmentation/regularization (additions)**: saturated, four-point dose-response RE +0.83 → TA +0.17 → reflect −0.14 → mixup −0.46; ANY diversity increase counts as pressure (EXP-013).
  - **Peak LR**: bracketed from above (0.6 lost −0.57pp, EXP-010); 0.4 at-or-near optimum.
  - **Batch/throughput**: 1024 + linear LR metric-neutral (EXP-012); batch 512 at-or-near optimum.
  - **Smoothing/eval (EMA)**: collapses the max-statistic (EXP-011); variance-INCREASING tricks barred as reward hacking.
- **Untried gaps**: WARMUP_FRAC (0.15, set in EXP-000 and never revisited) and WEIGHT_DECAY (5e-4, same) are the only two recipe constants never probed. Both are pressure/shape levers that do NOT touch the data pipeline (immune to the ~3% CPU loader margin, infra-errors EXP-013) and do not add regularization (immune to the saturated axis).
- Strategic frame (project-insights Medium): under fixed wall-clock, levers that DEFER progress all fail identically. Shortening warmup is the rare lever that moves progress EARLIER — the anneal phase (where accuracy accrues) lengthens from 85% to 92% of the budget.

## Candidate Ideas

### 1. SHORTEN WARMUP: WARMUP_FRAC 0.15 → 0.08
**Summary**: Single-constant change in train.py. Warmup currently consumes 45s of the 300s timed budget (~20 epochs) ramping linearly to peak 0.4; cut it to 24s (~11 epochs). The cosine anneal — the phase where essentially all accuracy accrues — stretches from 255s to 276s (+8%), and the LR-time integral rises since less budget idles at sub-peak LR. Peak, batch, WD, augmentation, eval cadence all byte-identical.

**Reasoning**: The NeurIPS 2024 warmup paper says warmup's only first-order job is to make the peak LR tolerable, and 1–5% of training is common practice — 15% is generous by 3x+. This model demonstrably tolerates MORE than its peak (EXP-010 ran 0.6 with zero instability), so the stability margin at 0.4 with an 8% ramp (~1075 steps) is large. Mechanistically this is the inverse of every recorded failure: EXP-010/009/013 all lost by deferring progress that the fixed-length anneal could not repay; this moves progress earlier and lengthens the anneal. It is also orthogonal to both fresh failure modes — zero CPU/loader cost (pure scalar in lr_at) and zero added regularization pressure.

**Sources**: arXiv 2406.09405; reports/exp-report-010.md (0.6 stable); project-insights § Medium (payoff-timing); infra-errors § Important (loader margin — untouched).

**Estimated Effort**: low — one-constant diff, ~480s runtime, byte-identical throughput signatures expected (dt 22.3ms, 139 epochs, 1613MB).

**Risk Assessment**: Failure mode is graceful (no-improvement): if early high-LR steps are noisier than the lengthened anneal repays, the trajectory ends ≤ baseline. Worst case is a small deficit like EXP-012 (−0.05). Cannot crash, cannot bust wall clock. Main assumption: warmup beyond ~8% provides no optimization benefit on this recipe — supported externally but never measured here. Magnitude honestly uncertain: folklore suggests +0.0–0.3pp.

### 2. REDUCE WEIGHT DECAY: WEIGHT_DECAY 5e-4 → 2.5e-4
**Summary**: Halve the selective weight decay (applied to conv/linear weights only) — the only untried lever that moves total regularization pressure DOWN. Everything else byte-identical.

**Reasoning**: The four-point dose-response curve (RE +0.83 → TA +0.17 → reflect −0.14 → mixup −0.46) shows marginal pressure has crossed zero — the recipe is at or past the regularization optimum. If past, reducing dose recovers accuracy, and WD is the one component that can be cut without touching the data pipeline. Counter-argument: the curve crossed zero BETWEEN TA (+0.17) and reflect (−0.14), which reads as "at optimum" rather than "past optimum" — in which case a 2.5e-4 cut undershoots and costs a little. WD also interacts with the LR schedule (effective-LR coupling), making the prior diffuse in both directions.

**Sources**: goal-learnings § Patterns High (dose-response, "only untried move is pressure REDUCTION"); reports/exp-report-009.md and -013.md (converged over-pressure runs).

**Estimated Effort**: low — one-constant diff, same runtime/signatures.

**Risk Assessment**: Graceful failure (no-improvement). At 139 epochs with LS+TA+RE still active, halving WD risks late-schedule overfit ⇒ best may land mid-schedule but the max-statistic tolerates that. Probability of ≥ +0.1pp is symmetric-to-slightly-negative given the "at optimum" reading.

### 3. REDUCE LABEL SMOOTHING: LABEL_SMOOTHING 0.1 → 0.05
**Summary**: Halve label smoothing — the alternative pressure-DOWN lever on the loss side rather than the weight side.

**Reasoning**: Same saturation argument as idea 2. However, goal-learnings (EXP-009 Insight) already flags LS-removal variants as low expected value, and LS interacts directly with the eval test-loss landscape; LS 0.1 has been in every recipe since EXP-000, including all five improvements — weak evidence it is miscalibrated.

**Sources**: goal-learnings § Failed Approaches (EXP-009 insight); exp-report-009.md.

**Estimated Effort**: low.

**Risk Assessment**: Graceful failure; weakest prior of the three — flagged low-value by prior analysis.

## Idea Evaluation

All three are single-constant probes with identical (low) effort, identical (graceful) risk profiles, and identical immunity to the two binding failure modes discovered in EXP-012/013 (loader margin, saturated augmentation). They separate on evidence strength and mechanism clarity:

- **Evidence strength**: Idea 1 has the only EXTERNAL evidence — a dedicated NeurIPS 2024 study concluding warmup beyond what peak-LR tolerance requires is wasted, plus the IN-PROJECT fact that peak 0.6 is stable (EXP-010), proving the tolerance requirement at 0.4 is met with huge margin. Ideas 2 and 3 rest on one internal extrapolation (the dose-response curve), and the curve's own shape ("crossed zero between TA and reflect") argues the recipe is AT the optimum, not past it — which would make pressure-reduction a small negative. Idea 3 is additionally pre-flagged low-value by EXP-009's analysis.
- **Mechanism clarity**: Idea 1's mechanism is arithmetic, not statistical: 21 more seconds (~9 epochs) of near-peak + annealing LR replace sub-peak ramp time; the LR-time integral strictly increases at unchanged stability margin. Ideas 2/3's mechanism requires the over-regularization hypothesis to hold, which is uncertain in sign.
- **Expected impact**: All are sub-half-point ideas. Idea 1's payoff arrives EARLY in the schedule (aligned with the fixed-clock insight); ideas 2/3's payoff, if any, arrives late (less overfit suppression at the end) where the anneal is already cold.
- **Strategic sequencing**: If idea 1 succeeds, the new baseline still permits the WD probe next loop; the levers are independent. Running the better-evidenced probe first is strictly better.

Idea 1 wins on every axis it differs on.

## Chosen Idea
**Selected**: SHORTEN WARMUP: WARMUP_FRAC 0.15 → 0.08

**Why this idea**:
It is the last untouched schedule-shape constant, the only candidate with direct external evidence (warmup beyond peak-LR tolerance is waste, common practice 1–5%; ours is 15% protecting a peak proven stable at 1.5x its value), and the only candidate whose mechanism moves progress EARLIER under the fixed wall clock — the direction every prior failure says this regime rewards. It is immune by construction to the saturated-augmentation axis and the ~3% CPU loader margin, and its failure mode is a graceful no-improvement.

**Hypothesis**:
Halving warmup from 15% to 8% of the time budget reallocates ~21s of sub-peak LR time into the near-peak/anneal phase, raising the LR-time integral at unchanged stability; the trajectory will run AHEAD of baseline from ~ep 15 onward and best_test_acc will reach ≥ 96.81 (baseline 96.71 + 0.1) at byte-identical throughput signatures (dt ~22.3ms, ~139 epochs, ~1613MB VRAM).
