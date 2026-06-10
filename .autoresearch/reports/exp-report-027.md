# Report EXP-027: Exclude BatchNorm and Bias from Weight Decay
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-027.md
- **Plan**: plans/plan-027.md
- **Log**: logs/exp-log-027.md

## Goal
EXP-027 targeted higher CIFAR-10 `best_test_acc` under the fixed harness and fixed 300s training budget. The current experiment-index baseline was 93.23% from commit `f187edf`, so the goal's +0.10 percentage-point rule required at least 93.33% to count as an improvement.

## Idea & Hypothesis
The chosen idea was to keep `WEIGHT_DECAY=1e-4` on conv/linear weight tensors while excluding BatchNorm and bias parameters from weight decay. The hypothesis was that targeted no-decay groups would improve calibration or generalization without reducing throughput or weakening main-weight regularization.

## Approach
`train.py` changed the optimizer setup from a single `model.parameters()` SGD parameter set to two parameter groups. Parameters with rank > 1 and non-bias names use `weight_decay=WEIGHT_DECAY`; 1D tensors and bias parameters use `weight_decay=0.0`. Architecture, batch size, LR, momentum, schedule, augmentation, FP32 compile/channels-last path, fixed training budget, and once-per-epoch validation were preserved.

## Execution
One local single-GPU run was launched on GPU 0 with stdout/stderr captured to `run.log`. Startup was clean, CUDA saw one NVIDIA H20, the expected 822,790-parameter model was used, `Batches per epoch: 390` confirmed the batch size was preserved, and the first LR drop fired at step 21000 with `lr=0.0100`. The run completed normally with no traceback, OOM, NaN, or Inf patterns.

## Results
- **Primary metric**: 92.99% (baseline: 93.23%, delta: -0.24 points, -0.26%)
- **Observations**: Pre-drop accuracy looked healthy and reached 89.52%, but post-drop refinement plateaued below the anchor. The best late value was 92.99% at epoch 103, followed by final accuracy 92.86% and final loss 0.3516.
- **Analysis**: The hypothesis was rejected. Excluding BatchNorm and bias from weight decay did not improve peak accuracy or final calibration enough to approach the 93.33% threshold. This result is more specific than EXP-023: even preserving `1e-4` on main weights while removing decay from normalization/bias parameters underperforms the baseline.
- **Key Learning**: Removing weight decay from BatchNorm and bias improved neither calibration nor peak accuracy, plateauing at 92.99%.

## Verification
- **Conditions**: Process, optimizer-group, schedule, and hard-constraint checks passed; the metric improvement condition failed.
- **Review Notes**: Results are trustworthy. The run completed successfully, reported numeric metrics, modified only `train.py`, preserved once-per-epoch validation, hit the step-21000 LR drop, and finished in 392.6 total seconds.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid result, but `best_test_acc=92.99%` is below the 93.23% baseline and the required 93.33% improvement threshold.

## Unexplored Avenues
- A more coupled optimizer retune could combine no-decay groups with a different LR or weight decay, but isolated targeted no-decay is not promising.
- Zero-initialized residual branch BatchNorm remains a distinct initialization lever, though it may slow early learning.
- Reflection padding for random crop remains a lightweight augmentation-quality change, but expected impact is small.

## Next Steps
Medium confidence: test zero-initialized residual branch last BatchNorm, because it is a distinct initialization change with no throughput penalty.

Low confidence: test reflection padding for random crop only if initialization changes fail.

Low confidence: revisit optimizer param groups only as part of a broader LR/weight-decay retune, not as an isolated no-decay change.
