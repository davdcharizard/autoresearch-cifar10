# Report EXP-039: Increase Weight Decay to 3e-4
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-039.md
- **Plan**: plans/plan-039.md
- **Log**: logs/exp-log-039.md

## Goal
Improve CIFAR-10 `best_test_acc` under the fixed `prepare.py` evaluation harness while modifying only `train.py`. The pre-experiment baseline was 93.97% from EXP-038, and the goal's +0.10 percentage-point rule required EXP-039 to reach at least 94.07% to count as an improvement.

## Idea & Hypothesis
EXP-039 tested whether the successful stronger-weight-decay direction from EXP-038 continued to help by raising `WEIGHT_DECAY` from `2e-4` to `3e-4`. The hypothesis was that a stronger bracket might further improve the late post-drop plateau.

## Approach
Only `train.py` changed: `WEIGHT_DECAY = 2e-4` became `WEIGHT_DECAY = 3e-4`. Architecture, reflected `RandomCrop`, `BATCH_SIZE = 128`, `LR_MILESTONES = [21000, 64000]`, `LR = 0.1`, `MOMENTUM = 0.9`, `label_smoothing=0.05`, FP32 channels-last compile path, seed, fixed time budget, and once-per-epoch validation were preserved. No deviations from the plan were needed.

## Execution
One local single-GPU run was launched on GPU 0 with output captured to `run.log`. Startup confirmed CUDA execution, 822,790 parameters, the fixed 300s training budget, and 390 batches per epoch. The first LR drop fired at step 21000, no second drop occurred, and the process exited cleanly before the 10-minute wall-clock limit.

## Results
- **Primary metric**: 93.55% (baseline: 93.97%, delta: -0.42 points, -0.45%)
- **Observations**: The run was slower and weaker than EXP-038, completing 37,782 steps and peaking at 93.55%; final accuracy fell to 92.67%.
- **Analysis**: The hypothesis was not supported. `3e-4` appears too strong for the current label-smoothed reflection anchor and over-regularizes relative to the validated `2e-4` setting.
- **Key Learning**: `WEIGHT_DECAY = 3e-4` over-regularizes the current anchor; `2e-4` should remain the stronger-decay default.

## Verification
- **Conditions**: metric improvement condition failed; hard constraints passed
- **Review Notes**: Results are trustworthy: the run completed cleanly, modified only `train.py`, preserved the fixed harness and single-GPU setup, and reported a numeric metric.
- **Verdict**: no-improvement
- **Verdict Basis**: valid run, but 93.55% is below both the 93.97% baseline and the 94.07% threshold.

## Unexplored Avenues
- Test an intermediate value such as `1.5e-4` if the goal is to map the bracket more precisely, though it may not clear the new threshold.
- Try a distinct optimizer-dynamics lever such as `LR = 0.12` while keeping `WEIGHT_DECAY = 2e-4`, since stronger decay may support more high-LR exploration.

## Next Steps
Try `LR = 0.12` on the `2e-4` anchor with medium confidence; it is a distinct lever and avoids the now-bounded stronger-weight-decay side.

Try `WEIGHT_DECAY = 1.5e-4` with low-to-medium confidence only if bracket mapping is preferred over expected immediate improvement.

Defer further isolated weight-decay increases with high confidence; `3e-4` is already too strong.

## Exit Action Results
