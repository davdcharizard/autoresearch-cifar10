# Report EXP-044: Mild RandAugment After Crop/Flip
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-044.md
- **Plan**: plans/plan-044.md
- **Log**: logs/exp-log-044.md

## Goal
Maximize CIFAR-10 `best_test_acc` under the fixed harness. The active baseline before this experiment was 93.97% from EXP-038 / commit `755be2c`, and the goal requires at least +0.10 percentage points to count as improvement, so EXP-044 needed `best_test_acc >= 94.07%`.

## Idea & Hypothesis
The chosen idea was a conservative policy-augmentation probe: add `transforms.RandAugment(num_ops=1, magnitude=5)` after crop/flip and before tensor conversion. The hypothesis was that mild non-erasing augmentation could improve generalization beyond the current reflection-padding, label-smoothed, `2e-4` weight-decay anchor without changing architecture, optimizer, schedule, or validation cadence.

## Approach
Only `train.py` was modified. The training transform gained one line, `transforms.RandAugment(num_ops=1, magnitude=5)`, after `RandomHorizontalFlip()` and before `ToTensor()`. All anchor settings were preserved: `STAGE_WIDTHS=(28,56,112)`, `LR=0.1`, `WEIGHT_DECAY=2e-4`, `LR_MILESTONES=[21000,64000]`, reflected crop padding, label smoothing 0.05, FP32/channels-last/compile, seed, and once-per-epoch validation.

## Execution
There was one local single-GPU run on GPU 0. Startup was clean, the first progress lines used `lr: 0.1000`, the first LR drop occurred at step 21000 with `lr: 0.0100`, and no second LR drop occurred. The run completed successfully with 39,015 steps and 458.2 seconds total wall-clock time.

## Results
- **Primary metric**: 93.83% (baseline: 93.97%, delta: -0.14pp, -0.15%)
- **Observations**: Accuracy peaked at epoch 80, then fluctuated below the peak and ended at 92.80%. RandAugment increased total runtime relative to recent non-RandAugment anchor runs but stayed below the 10-minute cap.
- **Analysis**: The hypothesis was not supported. Mild RandAugment was not destructive enough to crash or invalidate the run, but it did not improve the anchor and likely added augmentation difficulty/overhead without enough generalization benefit.
- **Key Learning**: Mild RandAugment at `num_ops=1, magnitude=5` stays below the 94.07% threshold and is not a useful isolated augmentation on the current anchor.

## Verification
- **Conditions**: Hard constraints and run-integrity checks passed; metric improvement condition failed.
- **Review Notes**: Results are trustworthy: numeric metric present, parameter count unchanged at 822,790, schedule geometry preserved, and tracked source diff was only the planned transform insertion.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid completed run, but `best_test_acc=93.83%` is below the 94.07% improvement threshold.

## Unexplored Avenues
- A still milder policy setting, such as `magnitude=3`, could reduce augmentation difficulty, but the runtime overhead and sub-baseline result make it lower priority.
- A targeted color-only augmentation could test whether the harmful part was geometric distortion rather than policy augmentation broadly.
- Bounded late EMA remains a separate late-stability idea and is not discredited by this transform-only failure.

## Next Steps
Prioritize a bounded late-stability probe with medium confidence: late EMA or short-window averaging may target the repeated peak-to-final drift while preserving the proven training recipe. As a lower-risk alternative, test a clean no-restart cosine schedule on the final `2e-4` anchor, but local schedule-only evidence is weaker.

## Exit Action Results
