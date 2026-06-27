# Report EXP-010: PEAK_LR 0.4 → 0.6 on the compiled 4x recipe
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-010.md
- **Plan**: plans/plan-010.md
- **Log**: logs/exp-log-010.md

## Goal

Maximize `best_test_acc` (%) of the CIFAR-10 ResNet within the fixed 300s training budget (higher is better). Baseline at experiment time: **96.71%** @ 1990397. Specific question: does the never-retuned PEAK_LR (0.4, linearly scaled in EXP-000 for an unaugmented 1x net) sit below the optimum for the current heavily-augmented compiled 4x recipe?

## Idea & Hypothesis

Chosen idea: raise the one-cycle peak 1.5x (0.4 → 0.6), everything else frozen — the first probe of the optimization-hyperparameter surface after capacity (EXP-007/008) and regularization (EXP-009) were closed. Three arguments pointed up: super-convergence's stable peaks ≥1.0 on CIFAR ResNets, augmentation-raised LR tolerance, and width-reduced gradient noise. Hypothesis: best_test_acc ≥ 96.81 with deeper mid-schedule depression recovered by the anneal. Runner-up candidates: EMA eval, WD halving.

## Approach

Single-line change in train.py: `PEAK_LR = 0.4` → `0.6` (comment updated). Architecture, schedule shape, augmentation, WD, compile byte-identical to baseline. No deviations from plan.

## Execution

One run, no retries (task b1lmbkbrp, GPU 0). Health checks clean: params and throughput identical to baseline (dt 22ms, 139 epochs — pure LR effect), epoch-1 eval 34.39% (warmup makes epoch-1 insensitive to the peak), no NaN or instability at any point. Completed in 480.8s.

## Results

- **Primary metric**: best_test_acc = 96.14% (baseline: 96.71, delta: −0.57pp, −0.59%)
- **Observations**: The run tracked ~3pp below EXP-006 through the hot mid-schedule (ep 80: 87.2 vs ~91; ep 120: 94.86 vs ~96) and — the decisive observation — the final anneal did NOT close the gap: the last five epochs were still creeping upward (96.06 → 96.14, final = best). Stability was never an issue; the cost was pure optimization progress.
- **Analysis**: Clean directional answer: 0.4 is at or above the LR optimum for this recipe. The mechanism is informative — the hotter peak didn't destabilize anything, it simply spent more of the fixed budget in a high-LR regime that makes slower test-accuracy progress, and the cosine descent (whose length is tied to the time budget, not to the damage) could not repay the difference; the still-climbing tail shows the run effectively turned itself into an undertrained one. This couples with the epoch-starvation learnings: in fixed-TIME training, ANY change that defers progress to later in the schedule (more capacity, hotter LR, heavier regularization) gets punished by the same mechanism. The super-convergence prior did not transfer because those results compare at fixed iteration counts with longer schedules, not at a fixed 300-second wall clock where recovery time is the scarce resource. The optimization surface now looks locally optimal from above; the remaining cheap probes are downward/sideways (LR 0.3, WD 2.5e-4) but the symmetric inference — that 0.4 was tuned-enough after all — lowers their expected value. The recipe at baseline 1990397 increasingly looks like a genuine local optimum across ALL four axes tried (capacity, topology, regularization, peak LR).
- **Key Learning**: In fixed-time training every lever that defers progress (capacity, hot LR, heavy regularization) fails by the same mechanism — the schedule cannot extend to repay deferred progress; external results benchmarked at fixed iterations do not transfer to fixed wall-clock.

## Verification

- **Conditions**: condition 2 failed (best_test_acc 96.14 < 96.81 = baseline + 0.1pp); condition 1 passed (clean exit, 480.8s ≤ 600, no NaN); condition 3 skipped per first-failure stop (informally compliant: 139 eval lines = 139 epochs)
- **Review Notes**: results confirmed trustworthy — metric consistent with the eval trail; throughput/params byte-identical to baseline confirming single-variable attribution; no constraint violations
- **Verdict**: no-improvement
- **Verdict Basis**: condition failure (valid run, metric below baseline + 0.1pp)

## Unexplored Avenues

- **PEAK_LR 0.3 (downward probe)**: the search is bracketed only from above; 0.4 could still be slightly above optimum. But EXP-006's healthy convergence shape (final≈best with small gap) suggests 0.4 is near-optimal, so expected gain is small.
- **WARMUP_FRAC reduction (0.15 → 0.08)**: spends less budget ramping up, giving the anneal more room — same family of schedule-shape tuning, untested, cheap.
- **Joint LR/WD move**: the classic coupled tune (LR up + WD down keeps effective regularization constant); muddier attribution but addresses the possibility that single-axis moves are blocked by the coupling.

## Next Steps

1. **EMA weight averaging for eval** — the last untried orthogonal lever; harvests the ±0.1pp final-epoch noise without touching the schedule; the bar is only +0.1pp. Confidence: low-medium.
2. **Schedule-shape tune (WARMUP_FRAC 0.15 → 0.08)** — redistributes budget from warmup to anneal, directly addressing the recovery-time scarcity this experiment exposed. Confidence: low-medium.
3. **GPU-side augmentation / loader overlap** — total_seconds shows ~50s of loader stalls outside the timed budget; restructuring augmentation could convert some into epochs, the one mechanism that consistently paid (+0.48 at EXP-006). Larger diff. Confidence: low-medium.

## Exit Action Results
<!-- Leave empty if no exit actions defined. -->
