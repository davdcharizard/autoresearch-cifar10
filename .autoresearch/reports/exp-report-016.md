# Report EXP-016: Linear-to-zero anneal (replace cosine post-warmup branch)
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-016.md
- **Plan**: plans/plan-016.md
- **Log**: logs/exp-log-016.md

## Goal

Maximize `best_test_acc` (%) of the CIFAR-10 ResNet within the fixed 300s training budget (higher is better). Baseline at experiment time: **96.71%** @ 1990397. Specific question: with the recipe certified a single-knob local optimum (EXP-015), does the best-evidenced STRUCTURAL move — swapping cosine for the theoretically-superior linear-to-zero anneal at identical total heat — beat the baseline?

## Idea & Hypothesis

Chosen idea: replace `lr_at()`'s cosine branch with `PEAK_LR * (1 − q)` — same peak (0.4), same warmup (0.15), same time-keying, and mathematically identical LR-time integral (both anneal shapes integrate to 0.5·peak), so the certified-closed heat axis is untouched by construction; only the heat DISTRIBUTION changes. Evidence: Defazio et al. (arXiv 2310.07831, 10-problem evaluation + theory: linear beats cosine, whose early plateau over-invests and flat tail under-anneals), "Straight to Zero" (2502.15938) replication, and the in-domain cifar10-fast piecewise-linear precedent. Hypothesis: behind-then-crossover trajectory and best_test_acc ≥ 96.81. Runners-up: compensated warmup/peak reshape, per-stage depth redistribution.

## Approach

2-line diff in train.py (`lr_at()` formula + comment). Analytic shape spot-check before launch verified all four anchors exactly (warmup branch untouched; peak 0.4 at p=0.15; anneal midpoint 0.2 at p=0.575; 0 at p=1.0). No deviations from plan.

## Execution

One run, no retries (task bnkebo0fk, launched 09:30:34 into a free GPU 0 via the composite launcher + inline watchdog). Pristine execution: zero watchdog SLOW events, post-hoc windowed profile 0 of 266 windows > 30ms (mean 22.4ms), 138 epochs / 13362 steps (~projection), total 510.9s, VRAM 1613.0MB, params 4,286,026 — signatures byte-identical to baseline as required by the pure-scalar-formula claim.

## Results

- **Primary metric**: best_test_acc = 96.21% (baseline: 96.71, delta: −0.50pp, −0.52%); bar was 96.81
- **Observations**: The predicted crossover NEVER HAPPENED — the run trailed baseline at every stage (ep 20: 76.1; ep 60: 87.1; ep 100: 91.5; ep 130: 95.5) and, uniquely among all sixteen experiments' clean runs, was STILL CLIMBING at cutoff: best was first reached at the FINAL epoch (138), with the last eight evals marching 95.69 → 95.52 → 95.73 → 95.73 → 96.01 → 96.02 → 96.17 → 96.21. The mechanism is arithmetic: at ep 130 (p≈0.94) the linear schedule still holds lr ≈ 0.028 while cosine is at ≈ 0.005 — the network keeps taking large noisy steps until the very end and never enters a converged plateau. Baseline's cosine cold-tail produces roughly ten near-converged evals in the 96.4–96.7 band for the max-statistic to harvest; the linear run got exactly one shot at its (unfinished) peak.
- **Analysis**: This is the third distinct way an external schedule result has failed to transfer, and the most instructive: EXP-010/014 failed via total HEAT (axis closed), this failed via METRIC SEMANTICS at matched heat. Defazio's theory and benchmarks optimize the FINAL-iterate value; our metric is a MAX over per-epoch evals under a hard wall clock. Cosine's "theoretically suboptimal" flat tail is precisely what manufactures the long converged plateau that a best-over-checkpoints metric feeds on (the EXP-011 EMA lesson approached from the schedule side). Strikingly, the linear run's final-epoch slope (+0.04 to +0.15/epoch) suggests its asymptote might be fine given more time — but the budget is fixed, and "needs more time" is the canonical failure of this regime (project-insights Medium: deferral always loses). Conclusion: the schedule axis is now closed in EVERY probed dimension — total heat (both sides), warmup length, anneal family/shape — all confirming the cosine 0.4/0.15 configuration. Eleven consecutive misses. Remaining untried space: heat-compensated multi-constant reshapes WITHIN cosine (low prior — EXP-014/016 both suggest the anneal-shape effect is dominated by tail-convergence timing, which cosine already optimizes), and structural architecture moves (per-stage depth redistribution at constant alignment — the one direction with no measured data point).
- **Key Learning**: For a best-over-checkpoints metric under fixed wall clock, the schedule's job near the end is to manufacture a long CONVERGED plateau, not to optimize the final iterate — final-value schedule theory (linear-beats-cosine) inverts; cosine's flat cold tail is load-bearing.

## Verification

- **Conditions**: pre-condition contention sanity CLEAN (138 epochs ≈ projection, watchdog silent, 0/266 slow windows); condition 1 FAILED (best_test_acc 96.21 < 96.81); conditions 2–3 skipped per first-failure stop (observed informally: 510.9s ≤ 600 and 138 evals = 138 epochs would have passed)
- **Review Notes**: trustworthy — metric consistent with the eval trail; signatures byte-identical to baseline so the deficit is attributable to the anneal-shape change alone
- **Verdict**: no-improvement
- **Verdict Basis**: condition failure (valid clean run; primary-metric necessary condition not met)

## Unexplored Avenues

- **Linear-then-flat-zero hybrid (anneal to 0 by p=0.85, hold ~0 after)**: would give linear the converged-plateau tail this metric needs — but it cuts total heat ~7%, re-entering the closed heat axis; a compensated version (slightly higher peak) is a two-constant trade with the usual ambiguity. Moderate interest if the schedule axis is ever revisited.
- **Defazio's refined schedules (rapid end-annealing)**: their own refinement anneals FASTER at the end than linear — directionally toward cosine's tail; reinforces that cosine is already near the refined optimum here. Low value.
- **Cosine-power variants (e.g. cos² for an even flatter tail)**: the plateau-manufacturing logic says a LONGER cold tail could help the max-statistic — but it trades against total heat and EXP-014 showed the schedule is heat-sensitive at the ±0.1pp scale. Within-noise expected.

## Next Steps

1. **Per-stage depth redistribution at constant alignment (blocks [2,3,4] instead of [3,3,3])** — the only structural direction with zero measured data points: stage-3 blocks add 4x the params of stage-1 blocks at near-equal FLOPs, so a 1→3 move buys ~+1.2M params at ~unchanged dt; capacity was closed only for UNIFORM scaling. Requires a dt-gate (compile may reshape costs). Confidence: low-medium.
2. **Width asymmetry at constant alignment (e.g. 64/128/320)** — same "capacity where it is cheap" logic via widths instead of depth; stage-3 features are 8x8 so wider final stage is FLOPs-cheap; must keep multiples of 32 (project-insights High). Confidence: low.
3. **Accept the optimum** — if structural probes also bracket out, the remaining honest move is repeated-seed characterization of the 96.71 recipe... which the no-seed-hacking constraint correctly forbids as a metric-move; treat as last-resort diagnostic only, not an experiment. Confidence: n/a (flagged for completeness).

## Exit Action Results
<!-- Leave empty if no exit actions defined. -->
