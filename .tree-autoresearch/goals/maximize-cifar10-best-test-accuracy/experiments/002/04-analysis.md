# Report EXP-002: Front-Loaded Probabilistic CutMix
- **Created**: 2026-08-05

## Goal

Increase CIFAR-10 `best_test_acc` (%), where higher is better, under the frozen 300-second charged training budget. EXP-002 grew from EXP-001 at 94.62%; after insertion, the global best is EXP-002 at 95.23%.

## Idea & Hypothesis

The chosen idea added one-pass CutMix to the validated EXP-001 WRN recipe only during the early 75% of charged training time. A fixed 0.5 gate interleaved clean and mixed batches, `Beta(1,1)` supplied rectangle area, and the final quarter stayed clean. The hypothesis was that input-level regularization would reduce the parent model's observed overfitting while preserving late low-LR convergence, producing at least 94.72%.

## Approach

Only `train.py` changed. A standard shared-rectangle CutMix helper derives side lengths from `sqrt(1-lambda)`, safely clones permuted source pixels, corrects lambda from clipped area, and weights original/paired cross-entropy from one forward pass. Dedicated seed-42 CPU and CUDA generators isolate CutMix gate/geometry/permutation draws from the parent's global shuffle, crop/flip, and drop-path streams. All mixing work remains between the parent `t0` and CUDA synchronization, so its cost is charged. The architecture, optimizer, BF16/layout path, LR and drop-path schedules, global seed, evaluator, cadence, and summary keys are unchanged.

## Execution

One run was launched on physical GPU 0 with `timeout 600s env CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`; no retries or metric-driven adjustments occurred. A deterministic helper smoke verified target orientation, clipped-area lambda identity, patch safety, and the zero-area case. The run completed exit 0 in 467.1 total seconds with no NaN/Inf, traceback, CUDA, or memory error. It completed 27,950 steps and 144 epochs, with one evaluation per epoch.

## Results

- **Primary metric**: 95.23% (parent: 94.62%, delta vs parent: +0.61 points, +0.64%; global best: 95.23%)
- **Observations**: CutMix applied to 10,257 of 20,668 eligible batches, a 0.4963 ratio consistent with the fixed 0.5 gate. Best accuracy occurred at epoch 143 and final accuracy was 95.19%, only 0.04 points lower. Final test loss improved from the parent's 0.2302 to 0.2044. The run completed 840 fewer steps than EXP-001 (27,950 vs 28,790), showing a small charged compute cost, but total time stayed comparable and peak VRAM was unchanged at 1,178.9 MiB.
- **Analysis**: The hypothesis was validated. CutMix moved both test accuracy and test loss in the intended direction despite slightly lower optimizer exposure, supporting genuine generalization improvement rather than extra compute. The fixed exposure and isolated RNG streams make the comparison stronger than a naive augmentation addition. The clean-phase transition coincided with drop-path annealing at 75%, so final-quarter dynamics remain jointly attributable; the overall parent-relative gain is attributable to the full front-loaded CutMix intervention.
- **Key Learning**: Front-loaded probabilistic CutMix adds 0.61 points to the time-aware WRN while preserving throughput and clean late convergence.

## Verification

- **Conditions**: All passed. The result exceeded the 94.72% parent-relative threshold, completed cleanly under both time limits, used GPU 0, modified only `train.py`, preserved model size/config, and evaluated no more than once per epoch.
- **Review Notes**: Results are trustworthy. The frozen evaluator produced all metrics; the log was fresh; helper geometry/orientation was deterministically checked; 49.63% exposure matched the preregistered gate; CutMix work was charged; and no seed selection, evaluator modification, stale output, or scope violation occurred.
- **Verdict**: improvement
- **Verdict Basis**: All necessary conditions passed and 95.23% exceeded the 94.62% parent by 0.61 points, above the required 0.10-point margin.

## Unexplored Avenues

- Tune CutMix probability, alpha, and cutoff around the successful recipe. The current values were fixed hypotheses rather than known optima, and the 0.61-point gain establishes useful local headroom.
- Test Mixup or an efficient Mixup/CutMix hybrid on the same parent; mixed-sample theory predicts distinct regularization behavior, so CutMix success does not exhaust the family.
- Reduce maximum drop path from 0.08 now that CutMix supplies strong early regularization; redundant regularization may limit fit during the first 75%.
- Add sparse EMA on the successful CutMix branch only after resolving its BatchNorm-statistics and online-checkpoint visibility issues.

## Next Steps

- **High confidence**: Explore a restrained CutMix operating-point change, such as lower probability or alpha, while preserving the proven early/clean phase structure.
- **Medium confidence**: Test reduced stochastic depth on top of EXP-002 to remove potentially redundant early regularization.
- **Medium confidence**: Compare efficient Mixup against CutMix from EXP-001 or EXP-002 to identify the stronger mixed-sample mechanism.

## Exit Action Results

No exit actions were defined for this goal.
