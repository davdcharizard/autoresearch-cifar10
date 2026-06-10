# Report EXP-045: Sparse Late EMA After First LR Drop
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-045.md
- **Plan**: plans/plan-045.md
- **Log**: logs/exp-log-045.md

## Goal
Maximize CIFAR-10 `best_test_acc` under the fixed harness. The active baseline before this experiment was 93.97% from EXP-038 / commit `755be2c`, and the goal requires at least +0.10 percentage points to count as improvement, so EXP-045 needed `best_test_acc >= 94.07%`.

## Idea & Hypothesis
The chosen idea was a bounded late EMA path: initialize EMA only after the first LR drop, update it sparsely every 100 optimizer steps, and evaluate exactly one model per epoch. The hypothesis was that a low-overhead late EMA could smooth post-drop weights without repeating the full-run per-step overhead from EXP-004 or the long equal-averaging collapse from EXP-021.

## Approach
Only `train.py` changed. The implementation imported `AveragedModel` and `get_ema_multi_avg_fn`, added EMA constants, kept `base_model` as the optimizer and EMA source, used `train_model` as the compiled training wrapper, and evaluated raw weights before EMA activation and EMA weights afterward. `use_buffers=False` was used to avoid the BatchNorm integer-buffer crash seen in EXP-021.

## Execution
One local single-GPU run completed on GPU 1. Startup, syntax, lint, parameter count, batch geometry, and validation cadence were all valid. Due to machine contention, the run reached only 23,495 steps; the first LR drop still occurred at step 21000, EMA activated, and `ema_updates=25`.

## Results
- **Primary metric**: 89.07% (baseline: 93.97%, delta: -4.90pp, -5.21%)
- **Observations**: Raw pre-drop accuracy reached only 88.71% by epoch 53. EMA evaluation activated at epoch 54 and peaked immediately at 89.07%, then collapsed to 84.59%, 83.18%, 81.42%, and final 79.45%.
- **Analysis**: The hypothesis was rejected. Sparse late EMA avoided the implementation crash and full-run per-step update overhead, but it still produced an averaged model whose evaluation quality degraded sharply. The most likely mechanism is parameter/BatchNorm-state mismatch plus too little compatible low-LR trajectory after a very late first drop.
- **Key Learning**: Sparse late EMA with copied/live buffers still collapses after activation, so weight averaging remains unsafe for this harness without BN recalibration or a different evaluation design.

## Verification
- **Conditions**: Hard constraints and run-integrity checks passed; metric improvement condition failed.
- **Review Notes**: Results are trustworthy: run completed, numeric metrics were printed, only `train.py` changed, `num_params` stayed 822,790, first LR drop and EMA activation were observed, and wall-clock stayed below 10 minutes.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid completed run, but `best_test_acc=89.07%` is below the 94.07% improvement threshold.

## Unexplored Avenues
- EMA with explicit BatchNorm recalibration might fix the mismatch, but it risks violating the fixed evaluation/training protocol unless carefully designed.
- Evaluating both raw and EMA models in the same epoch would diagnose whether EMA helps while preserving raw peaks, but it violates the once-per-epoch validation constraint.
- A clean schedule alternative, such as no-restart cosine on the final anchor, remains a distinct idea and is not discredited by this EMA failure.

## Next Steps
High confidence: stop weight-averaging variants for this goal unless the plan directly solves BatchNorm mismatch without extra validation.

Medium confidence: run a clean no-restart cosine schedule on the final `2e-4` anchor as a distinct schedule-shape probe.

Medium confidence: consider a targeted color-only augmentation rather than policy/masking augmentation, because reflection padding helped but destructive augmentation has failed.

## Exit Action Results
