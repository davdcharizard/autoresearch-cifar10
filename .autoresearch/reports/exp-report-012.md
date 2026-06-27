# Report EXP-012: BATCH_SIZE 1024 + PEAK_LR 0.8 (linear scaling)
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-012.md
- **Plan**: plans/plan-012.md
- **Log**: logs/exp-log-012.md

## Goal

Maximize `best_test_acc` (%) of the CIFAR-10 ResNet within the fixed 300s training budget (higher is better). Baseline at experiment time: **96.71%** @ 1990397. Specific question: does the H20's utilization headroom at batch 512 (22ms/step on tiny 32x32 kernels) let batch 1024 buy epochs cheaply enough to convert via the only mechanism with a 100% hit rate (throughput → epochs → accuracy)?

## Idea & Hypothesis

Chosen idea: double batch and peak LR together (Goyal linear scaling — the same rule that set 0.4 @ 512), the last untried throughput lever after precision/compile were exhausted and GPU-residency was eliminated by first principles. Explicitly distinguished from EXP-010's fixed-batch LR raise: scaling both is supposed to preserve gradient-noise scale. Hypothesis: dt < 44ms ⇒ ≥ ~150 epochs ⇒ best_test_acc ≥ 96.81, with the mid-schedule trajectory TRACKING baseline. Runner-ups: WARMUP_FRAC 0.08, WD 2.5e-4. (A terminal-LR-floor idea was discarded as reward hacking — it would fish the max-statistic without improving the model.)

## Approach

Two-constant diff in train.py: `BATCH_SIZE = 1024`, `PEAK_LR = 0.8` (comment updated). Steps/epoch 97→48; warmup batch, loader, and time-keyed schedule all follow the constants. No deviations from plan.

## Execution

One run, no retries (task bm70wed4e, GPU 0, launched into a fully idle node). Contention protocol from EXP-011 applied end-to-end: throughput detector (SLOW > 55ms) fired zero events; post-run epoch sanity passed (151 epochs vs ~150 projected from the step-100 dt of 41–42ms). Cold-cache compile for the new shape cost 23.8s startup as planned. total_seconds 560.8 — inside the 600s cap but the tightest margin yet (151 evals ride the wall clock). VRAM 3134.6MB (~2x, as predicted).

## Results

- **Primary metric**: best_test_acc = 96.66% (baseline: 96.71, delta: −0.05pp, −0.05%)
- **Observations**: The throughput half of the hypothesis HELD exactly: dt 41.5ms (1.86x for 2x the work ⇒ +8% img/s), 151 epochs vs 139. The optimization half FAILED: instead of tracking baseline, the trajectory ran far below it through the hot phase (ep 20: 69.5 vs ~88; ep 60: 83.2 vs ~92; ep 100: 91.1 vs ~96) with visibly bouncy evals, then the cosine tail recovered +5.5pp over the last 50 epochs to a CONVERGED plateau (96.53–96.66 over ep 146–151, final 96.63 ≈ best 96.66).
- **Analysis**: Cleanest possible decomposition of the lever: the hardware delivered (+12 epochs) and the optimizer gave it back (−12 epochs' worth of progress). Linear LR scaling did not preserve the trajectory at this scale — at batch 1024/LR 0.8 the run spends its mid-schedule in a much hotter effective regime (the linear-scaling equivalence is known to degrade as batch grows; BN-statistics quality and the curvature limit at LR 0.8 are the likely culprits — this is a real optimization-physics effect, not noise: the deficit peaked at ~18pp). Crucially the tail CONVERGED (final ≈ best, flat plateau), so unlike EXP-010 this is not an unrepaid-deferral story — the recovery completed and simply landed −0.05pp from baseline, i.e. batch 512 → 1024 at linear-scaled LR is metric-NEUTRAL on this recipe: epochs gained ≈ trajectory quality lost. The throughput axis is now closed: 512 is at-or-near the batch optimum (smaller batches lose img/s; larger trade epochs for trajectory 1:1). With capacity, regularization, peak LR (both directions at fixed batch), smoothing, and now batch/throughput all measured, the baseline recipe is a measured local optimum across every axis the budget exposes.
- **Key Learning**: Linear batch/LR scaling is metric-neutral at 512→1024 on this recipe: +8% throughput (+12 epochs) was exactly cancelled by the hotter trajectory's converged-but-lower endpoint — the throughput axis is closed, not because the GPU lacks headroom but because the optimizer cannot use the extra epochs at the scaled LR.

## Verification

- **Conditions**: pre-condition contention sanity CLEAN (151 vs ~150 projected; detector silent); condition 1 passed (clean exit, 560.8s ≤ 600); condition 2 failed (96.66 < 96.81); condition 3 skipped per first-failure stop (informally compliant: 151 eval lines = 151 epochs)
- **Review Notes**: trustworthy — metric consistent with the eval trail; throughput profile flat (no contention); params unchanged; single-variable (well, single-principle: two coupled constants) attribution intact
- **Verdict**: no-improvement
- **Verdict Basis**: condition failure (valid clean run, metric below baseline + 0.1pp)

## Unexplored Avenues

- **Batch 1024 with sub-linear LR scaling (e.g. sqrt: PEAK_LR ≈ 0.57)**: the deficit came from the hot trajectory, so a cooler peak at the same batch might keep most of the +12 epochs without the damage. But EXP-010 + this result bracket the schedule tightly, and the converged tail here means the headroom is ≤ ~0.2pp — borderline against the bar.
- **Batch 768 (1.5x, still 256-aligned)**: splits the difference — ~+4–5% throughput at a milder LR 0.6; same logic, smaller expected magnitude on both sides of the trade.
- **Warmup lengthening AT batch 1024**: large-batch recipes often need longer warmup, and ours kept WARMUP_FRAC 0.15; a 0.25 warmup at 1024 might tame the bouncy hot phase. Confounded two-variable move; low confidence.

## Next Steps

1. **WARMUP_FRAC 0.15 → 0.08 at the baseline batch** — the last surviving simple schedule-shape probe; trivial, clean, but expected within noise. Confidence: low-medium.
2. **WD re-tune (5e-4 → 2.5e-4)** — the only base hyperparameter never revisited; the EXP-009 saturation result hints total regularization sits at the edge, and explicit WD is part of that budget. Confidence: low-medium.
3. **Batch 1024 @ PEAK_LR ~0.57 (sqrt scaling)** — salvages the validated +12-epoch throughput gain if the trajectory damage is LR-driven rather than BN-driven; the one remaining idea with a mechanism that could exceed +0.1pp. Confidence: low-medium.

## Exit Action Results
<!-- Leave empty if no exit actions defined. -->
