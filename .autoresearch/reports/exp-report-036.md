# Report EXP-036: Batch Size 112 on Current Anchor
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-036.md
- **Plan**: plans/plan-036.md
- **Log**: logs/exp-log-036.md

## Goal
Improve CIFAR-10 `best_test_acc` under the fixed single-GPU, fixed-budget harness. The active baseline before EXP-036 was 93.70%, and the goal requires at least a +0.10 percentage-point absolute gain, so this experiment needed `best_test_acc >= 93.80%` to count as an improvement.

## Idea & Hypothesis
EXP-036 tested whether a milder smaller batch could improve the reflection-padding, label-smoothed 28/56/112 anchor through beneficial update stochasticity. Batch size 96 had previously failed, but batch size 112 was chosen as a less aggressive probe that might preserve enough schedule coverage while changing gradient noise.

## Approach
The implementation changed only `BATCH_SIZE` in `train.py`, from 128 to 112. Architecture, reflection crop padding, `label_smoothing=0.05`, `LR_MILESTONES = [21000, 64000]`, optimizer settings, FP32 channels-last compile path, seed, fixed budget, and once-per-epoch evaluation were preserved. There were no deviations from the plan.

## Execution
One local single-GPU run completed cleanly on 2026-06-09. Startup confirmed CUDA execution, `822,790` parameters, the 300s training budget, and `Batches per epoch: 446`. The first LR drop fired at step 21000 with `lr: 0.0100`, and the second milestone at 64000 remained unreachable.

## Results
- **Primary metric**: 93.43% (baseline: 93.70%, delta: -0.27 points, -0.29%)
- **Observations**: The run completed 90 epochs and 39,859 steps with `final_test_acc=93.22%`, `final_test_loss=0.2476`, `total_seconds=390.3`, and `peak_vram_mb=600.0`.
- **Analysis**: Batch size 112 preserved the first LR drop but reduced useful epoch and step coverage versus the batch-128 anchor. The post-drop plateau peaked at 93.43%, below the active baseline and far below the 93.80% threshold. This confirms that smaller-batch stochasticity is not a useful isolated lever for the current anchor.
- **Key Learning**: Batch size 112 preserved the first LR drop but cut useful coverage to 39,859 steps and plateaued at 93.43%.

## Verification
- **Conditions**: One necessary condition failed: `best_test_acc` did not reach the 93.80% improvement threshold.
- **Review Notes**: Results are trustworthy. The process exited cleanly, produced numeric metrics, preserved the fixed budget and architecture, changed only `train.py`, used the planned batch geometry and LR drop, and showed no error, OOM, NaN, or Inf signatures.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid completed run, but the primary metric was below both the 93.70% baseline and the 93.80% threshold required by the goal.

## Unexplored Avenues
- Larger batch sizes could preserve more images-per-step but would change update count and may need schedule recalibration; evidence for this is weak.
- Smaller batches should only be revisited with a fundamentally different schedule or optimizer rationale, not as an isolated stochasticity tweak.

## Next Steps
Move away from smaller-batch tuning. Medium confidence: test stronger smoothing such as `label_smoothing=0.08` as a no-throughput scalar regularization probe. Medium-low confidence: test a tightly bounded late averaging variant only if the implementation avoids per-step overhead and long equal-average collapse.

## Exit Action Results
