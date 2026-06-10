# Report EXP-038: Increase Weight Decay to 2e-4
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-038.md
- **Plan**: plans/plan-038.md
- **Log**: logs/exp-log-038.md

## Goal
Improve CIFAR-10 `best_test_acc` under the fixed `prepare.py` evaluation harness while modifying only `train.py`. The pre-experiment baseline was 93.70% from EXP-032, and the goal's +0.10 percentage-point rule required EXP-038 to reach at least 93.80% to count as an improvement.

## Idea & Hypothesis
EXP-038 tested whether increasing `WEIGHT_DECAY` from `1e-4` to `2e-4` improves the current reflection-padding, label-smoothed 28/56/112 anchor. The prior lower-weight-decay probe (`5e-5`) underperformed, so the stronger side of the bracket was a clean, no-throughput regularization test.

## Approach
Only `train.py` changed: `WEIGHT_DECAY = 1e-4` became `WEIGHT_DECAY = 2e-4`. Architecture, reflected `RandomCrop`, `BATCH_SIZE = 128`, `LR_MILESTONES = [21000, 64000]`, `LR = 0.1`, `MOMENTUM = 0.9`, `label_smoothing=0.05`, FP32 channels-last compile path, seed, fixed time budget, and once-per-epoch validation were preserved. No deviations from the plan were needed.

## Execution
One local single-GPU run was launched on GPU 0 with output captured to `run.log`. Startup confirmed CUDA execution, 822,790 parameters, the fixed 300s training budget, and 390 batches per epoch. The first LR drop fired at step 21000, no second drop occurred, and the process exited cleanly before the 10-minute wall-clock limit.

## Results
- **Primary metric**: 93.97% (baseline: 93.70%, delta: +0.27 points, +0.29%)
- **Observations**: Accuracy crossed the 93.80% improvement bar at epoch 69 and peaked at epoch 74, then ended at 93.54% final accuracy after 107 epochs and 41,389 steps.
- **Analysis**: The hypothesis was supported. Stronger weight decay improved the late post-drop plateau without changing throughput geometry, parameter count, or validation cadence.
- **Key Learning**: The label-smoothed reflection anchor benefits from stronger `2e-4` weight decay, making shrinkage strength a validated regularization lever.

## Verification
- **Conditions**: all passed
- **Review Notes**: Results are trustworthy: the run completed cleanly, modified only `train.py`, preserved the fixed harness and single-GPU setup, reported a numeric metric, and exceeded the +0.10 threshold.
- **Verdict**: improvement
- **Verdict Basis**: all hard constraints and verification checks passed, and 93.97% exceeds the 93.70% baseline by +0.27 points.

## Unexplored Avenues
- Bracket stronger weight decay around the new anchor, such as `1.5e-4` or `3e-4`, to test whether 2e-4 is near the optimum or just the first useful stronger setting.
- Couple `2e-4` with a distinct non-schedule lever such as initial LR or low-frequency post-drop averaging, but avoid adjacent smoothing and first-drop retunes that recently stayed inside the noise band.

## Next Steps
Try a focused `WEIGHT_DECAY = 3e-4` probe with medium confidence; it is the nearest unexplored extension of the successful regularization axis.

Try `LR = 0.12` on the new `2e-4` anchor with medium confidence; stronger shrinkage may tolerate slightly larger high-LR exploration.

Defer EMA-style averaging with low-to-medium confidence until scalar brackets are exhausted, because prior averaging variants had overhead or collapse risks.

## Exit Action Results
- PR creation skipped: no git remote is configured for this repository; EXP-038 was committed locally and merged into `autoresearch/dev`.
