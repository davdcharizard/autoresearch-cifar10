# Report EXP-023: Lower Weight Decay on 28/56/112
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-023.md
- **Plan**: plans/plan-023.md
- **Log**: logs/exp-log-023.md

## Goal
EXP-023 targeted higher CIFAR-10 `best_test_acc` under the fixed harness and fixed 300s training budget. The current experiment-index baseline was 93.23% from commit `f187edf`, so the goal's +0.10 percentage-point rule required at least 93.33% to count as an improvement.

## Idea & Hypothesis
The chosen idea was an isolated regularization reduction on the current best 28/56/112 ResNet-20 anchor. Prior stronger-regularization experiments had failed, so the hypothesis was that reducing `WEIGHT_DECAY` from `1e-4` to `5e-5` might let the widened model use its capacity more fully without changing throughput or schedule.

## Approach
`train.py` changed only `WEIGHT_DECAY` from `1e-4` to `5e-5`. The architecture, 21k first LR drop, optimizer, batch size, augmentation, FP32 compile/channels-last path, fixed training budget, and once-per-epoch validation were preserved.

## Execution
One local single-GPU run was launched on GPU 0 with stdout/stderr captured to `run.log`. Startup was clean, CUDA saw one NVIDIA H20, the expected 822,790-parameter model was used, and the first LR drop fired at step 21000 with `lr: 0.0100`. The run completed normally with no traceback, OOM, NaN, or Inf patterns.

## Results
- **Primary metric**: 92.83% (baseline: 93.23%, delta: -0.40 points, -0.43%)
- **Observations**: Accuracy jumped after the 21k LR drop, reaching 91.17% at epoch 54, then climbed slowly to a late peak of 92.83% at epoch 107. Final accuracy fell to 92.46% with final loss 0.3541, indicating weaker late generalization than the current anchor rather than a throughput failure.
- **Analysis**: The result rejects the hypothesis that the 28/56/112 anchor is over-regularized at `1e-4`. Lower L2 preserved throughput and schedule integrity but landed well below both the 93.23% baseline and the 93.33% threshold, so regularization reduction is not a promising isolated lever for this recipe.
- **Key Learning**: Lower weight decay weakened generalization on the current anchor, so `1e-4` remains better for 28/56/112.

## Verification
- **Conditions**: All process and hard-constraint checks passed; the metric improvement condition failed.
- **Review Notes**: Results are trustworthy. The run completed successfully, reported numeric metrics, modified only `train.py`, preserved once-per-epoch validation, hit the planned step-21000 LR drop, and finished in 390.6 total seconds.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid result, but `best_test_acc=92.83%` is below the 93.23% baseline and the required 93.33% improvement threshold.

## Unexplored Avenues
- A milder value such as `7.5e-5` could test whether the failure was due to reducing L2 too far, but the observed 0.40-point drop makes this low priority.
- A coupled schedule or augmentation change could revisit weight decay later, but isolated scalar regularization changes now look weaker than optimizer or training-loop ideas.
- Batch-size changes remain a separate way to alter implicit regularization, though they will confound throughput, update count, and schedule semantics.

## Next Steps
Medium confidence: test a different non-capacity lever that preserves the 28/56/112, 21k anchor, such as optimizer hyperparameter tuning that does not add overhead. Medium confidence: explore batch size only with explicit step-budget and LR-milestone recalibration. Low confidence: finer weight-decay interpolation, because `5e-5` moved the result in the wrong direction by a large margin.
