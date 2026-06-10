# Report EXP-037: Stronger Label Smoothing 0.08
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-037.md
- **Plan**: plans/plan-037.md
- **Log**: logs/exp-log-037.md

## Goal
Improve CIFAR-10 `best_test_acc` under the fixed single-GPU, fixed-budget harness. The active baseline before EXP-037 was 93.70%, and the goal requires at least a +0.10 percentage-point absolute gain, so this experiment needed `best_test_acc >= 93.80%` to count as an improvement.

## Idea & Hypothesis
EXP-037 tested the stronger side of the validated label-smoothing anchor. Since `label_smoothing=0.05` was the current successful baseline and `label_smoothing=0.03` had nearly cleared the threshold, the hypothesis was that `label_smoothing=0.08` might improve late generalization without changing throughput or schedule reachability.

## Approach
The implementation changed only the training loss call in `train.py`, from `label_smoothing=0.05` to `label_smoothing=0.08`. Architecture, reflection crop padding, batch size 128, `LR_MILESTONES = [21000, 64000]`, optimizer settings, FP32 channels-last compile path, seed, fixed budget, and once-per-epoch evaluation were preserved. There were no deviations from the plan.

## Execution
One local single-GPU run completed cleanly on 2026-06-09. Startup confirmed CUDA execution, `822,790` parameters, the 300s training budget, and `Batches per epoch: 390`. The first LR drop fired at step 21000 with `lr: 0.0100`, and the second milestone at 64000 remained unreachable.

## Results
- **Primary metric**: 93.73% (baseline: 93.70%, delta: +0.03 points, +0.03%)
- **Observations**: The run completed 104 epochs and 40,315 steps with `final_test_acc=93.34%`, `final_test_loss=0.2682`, `total_seconds=402.3`, and `peak_vram_mb=660.4`.
- **Analysis**: Stronger smoothing produced a small lift over the baseline but remained inside the explicit +0.10 noise band. Together with EXP-033, this suggests the current `0.05` smoothing value is close to the best useful point, while neighboring smoothing values do not deliver a meaningful improvement.
- **Key Learning**: Stronger smoothing reached 93.73%, a small lift over baseline but still inside the +0.10 noise band.

## Verification
- **Conditions**: One necessary condition failed: `best_test_acc` did not reach the 93.80% improvement threshold.
- **Review Notes**: Results are trustworthy. The process exited cleanly, produced numeric metrics, preserved the fixed budget and architecture, changed only `train.py`, used the planned schedule and batch geometry, and showed no error, OOM, NaN, or Inf signatures.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid completed run, but the primary metric improved by only +0.03 points, below the +0.10 threshold required by the goal.

## Unexplored Avenues
- Smoothing values between 0.05 and 0.08 are possible but low priority because both 0.03 and 0.08 stayed inside the noise band.
- Late averaging remains a distinct mechanism that targets evaluation stability rather than stronger regularization, but it must avoid the prior long-window collapse and per-step overhead.

## Next Steps
Move away from isolated smoothing-value bracketing. Medium confidence: test a bounded late-stability mechanism only if it preserves throughput and once-per-epoch validation. Low-medium confidence: try an orthogonal no-throughput scalar such as modestly higher weight decay, but only with explicit awareness that `1e-4` is the current anchor.

## Exit Action Results
