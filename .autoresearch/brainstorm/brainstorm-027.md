# Brainstorm EXP-027
**Created**: 2026-06-10
**Goal**: goals/maximize-cifar10-test-accuracy.md

<!-- This file is focused on IDEATION only. -->

## Web Search & Literature Review

No new external searches — this loop's question is internal calibration. Relevant standing context:
- **Known context — nondeterminism under cudnn.benchmark + bf16 atomics**: with a fixed seed (42), run-to-run variation persists through cudnn algorithm autotuning and non-deterministic reduction orders in bf16 atomics/cudnn kernels. The run-level σ of the final metric is an empirical property of the recipe, not derivable from the seed.
- **In-project σ evidence (only one datapoint-pair)**: EXP-021 ran its (different) config twice at identical settings: 96.41 / 96.51 — within-config spread 0.10 ⇒ rough σ ≈ 0.07. No replicate of the BASELINE config has ever been run.

## Experimental History Review

- **Current best**: 96.71 @ 1990397 (EXP-006). **Twenty-one consecutive misses (EXP-007…026).**
- **Axes closed this session**: distribution time-structure (EXP-025, −0.87, augmented tail load-bearing); activation function (EXP-026, −0.10, smooth activations cost 5–20% of dt; ReLU uniquely free).
- **The calibration gap (the bottleneck this loop targets)**: FIVE misses lie in the −0.05…−0.15 band (EXP-012 −0.05, EXP-026 −0.10, EXP-020 −0.13, EXP-013 −0.14, EXP-022 −0.14). If baseline run-level σ ≈ 0.1, some "axis closed" readings are statistically indistinguishable from baseline, and — more consequentially — **96.71 itself may be a high draw**: a single lucky max harvested from 139 noisy evals. If the baseline's true mean best is ~96.55–96.6, the +0.1pp bar (96.81) sits ~2σ above the mean, which would EXPLAIN twenty-one consecutive misses by interventions whose true effects were ~0±0.1. No future result can be interpreted sharply without this number.
- **Sanction trail**: this replicate was proposed in brainstorm-025 (idea 3, "defer until intervention candidates are exhausted"), exp-report-024 § Next Steps #3, and exp-report-026 § Next Steps #1 (confidence: high as measurement). The named intervention classes are now exhausted or low-prior — the deferral condition is met.
- **Constraint check (no seed hacking / no variance harvesting)**: goal-learnings § max-statistic law bars variance-tail tricks as reward hacking. The replicate is the opposite: NO code change at all (zero diff), seed stays 42, and the verdict is PRE-REGISTERED as `no-improvement` regardless of the measured value — even if a replicate lands ≥ 96.81 by luck, the baseline does not move (nothing to commit; same code cannot be an "improvement" over itself). The experiment produces information, not metric movement, by construction.

## Candidate Ideas

### 1. Baseline variance replicate ×2 (measurement loop, zero diff)
**Summary**: Run the UNMODIFIED baseline train.py twice (sequentially, standard launcher and contention gates), recording best_test_acc for each. Combined with the standing 96.71, this gives three draws of the same configuration — enough for a first real estimate of run-level σ and of whether 96.71 is the distribution's center or its tail.

**Reasoning**: Twenty-one misses span −0.05…−0.99 and the verdict on every one of them implicitly assumed the baseline number is solid. Three concrete decision rules hang on this measurement: (a) if replicates land ~96.6–96.8, σ is small, near-band misses were real losses, and future candidates need true effects ≥ ~+0.25 to clear the bar reliably — the campaign should only spend loops on big-swing mechanisms; (b) if replicates land ~96.5–96.6, the 96.71 was a ~+1.5–2σ draw, the de-facto bar is ~96.81 vs a ~96.57 mean (≈ +0.25 true effect needed) and EVERY band-miss axis (batch, shortcut, activation, reflect-pad) is actually unresolved-at-noise — recorded as a calibration finding for the user and for all future verdict interpretation; (c) if a replicate exceeds 96.81, it is conclusive proof the bar is within the noise envelope — still recorded as no-improvement (pre-registered), but the strategic insight is maximal. All three outcomes are informative; none moves the baseline.

**Sources**: brainstorm-025 idea 3; exp-report-024 § Next Steps; exp-report-026 § Next Steps #1; goal-learnings § Patterns (max-statistic law — anti-variance-harvesting framing); EXP-021 within-config spread (exp-report-021).

**Estimated Effort**: minimal — zero diff; two standard launches (~16–18 min total).

**Risk Assessment**: No code risk (no change). Process risk: a lucky ≥96.81 replicate could tempt a false improvement — neutralized by pre-registering the verdict in the plan. Contention risk handled by standard gates; a contaminated run is rerun once per protocol. Opportunity cost: one loop with no chance of metric improvement — accepted because no intervention can be interpreted without σ.

### 2. Stage-3-only width increase: (64,128,256) → (64,128,320), early-dt-gate screened
**Summary**: Widen only stage 3 (8×8 resolution, cheapest FLOPs) to 320 channels (multiple of 64). The one capacity move goal-learnings explicitly leaves open ("width asymmetry remains the one untried capacity-where-cheap move"). Screened by the now-validated early-dt gate (kill if projected epochs < ~128).

**Reasoning**: EXP-017 isolated its deficit to REMOVED stage-1 depth; adding stage-3 width only adds. But the capacity-without-throughput class has three High-importance failures, and EXP-026 just demonstrated how unforgiving the dt budget is even for cheap ops; a conv-width increase is far more expensive than an activation.

**Sources**: goal-learnings § Failed Approaches (EXP-017 insight); project-insights (H20 channel alignment); exp-report-026 (gate protocol).

**Estimated Effort**: low — one constant in ResNet.__init__ + gate-screened launch.

**Risk Assessment**: graceful (gate kill or converged miss), but interpretation of a −0.05…−0.15 result would AGAIN be noise-ambiguous without idea 1's σ — sequencing argument favors measuring first.

### 3. Terminal BN-stat recalibration (forward-only clean passes, budget-charged)
**Summary**: Re-converge BN running stats on clean data before tail evals (exp-report-025's surviving fragment) — ~50 forward-only batches charged to the timed budget.

**Reasoning**: cannot overfit; captures the pure alignment component. But EXP-025's data caps its plausible magnitude at ≤ +0.35-from-depressed, likely ≪ that from the augmented plateau's top, and possibly negative — another result destined for the noise band that σ-knowledge must precede.

**Sources**: exp-report-025 § Unexplored Avenues; AdaBN-class context.

**Estimated Effort**: low-medium.

**Risk Assessment**: graceful but risks degrading the tail harvest window (−0.1–0.2 exposure); medium-low prior; same noise-ambiguity sequencing argument.

## Idea Evaluation

**Evidence strength**: Idea 1 rests on arithmetic, not literature: five of twenty-one misses are inside a band whose width we have never measured, and the campaign's only within-config replicate pair (EXP-021: spread 0.10) suggests σ is the same order as the success bar increment. Ideas 2–3 are interventions whose results would land in exactly that ambiguous band.

**Mechanism clarity**: Idea 1's "mechanism" is calibration: it converts every past and future verdict from point-comparison to noise-aware comparison. Ideas 2–3 have plausible but weak mechanisms (capacity-where-cheap; stat alignment).

**Expected impact**: On the metric this loop: zero for idea 1 by construction, ~0±0.15 for ideas 2–3. On the CAMPAIGN: idea 1 dominates — it determines what effect size future candidates must target and may reveal that the bar sits ~2σ above the achievable mean, the single most consequential possible finding after twenty-one misses.

**Risk profile**: Idea 1 is the only zero-code-risk option and its sole process risk (lucky bar-cross) is neutralized by pre-registration. Ideas 2–3 are graceful but burn loops on noise-band answers.

**Feasibility**: Idea 1 is trivially executable with the standard launcher.

## Chosen Idea
**Selected**: Baseline variance replicate ×2 (measurement loop, zero diff)

**Why this idea**:
After twenty-one consecutive misses — five inside an unmeasured noise band — no intervention result can be interpreted sharply, and the possibility that 96.71 is itself a high draw (making the de-facto bar ~2σ above the achievable mean) is the most important open question in the campaign. The replicate was explicitly deferred by three prior loops until intervention candidates thinned; that condition is now met (alignment and activation classes just closed). It is constraint-clean: zero diff, seed untouched, verdict pre-registered as no-improvement regardless of outcome — the anti-thesis of seed hacking.

**Hypothesis**:
Two replicates of the unmodified baseline (same code, seed 42, standard gates) will land in **96.5–96.8**, giving a first run-level σ estimate of ~0.07–0.15 around a mean possibly below 96.71. Concretely testable sub-predictions: (a) both replicates < 96.81 (the bar is not casually re-crossable); (b) replicate spread ≤ 0.2; (c) at least one replicate < 96.71 (96.71 is at or above the distribution center). Outcome interpretation is pre-registered in the brainstorm's idea 1 Reasoning; the verdict is `no-improvement` with the mean replicate value recorded, the baseline does NOT move, and goal-learnings receives a Protocol Finding quantifying σ for all future verdicts.
