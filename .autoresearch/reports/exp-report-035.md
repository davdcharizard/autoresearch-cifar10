# Report EXP-035: Lower Smoothing with 22k First Drop
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-035.md
- **Plan**: plans/plan-035.md
- **Log**: logs/exp-log-035.md

## Goal
Improve CIFAR-10 `best_test_acc` under the fixed single-GPU, fixed-budget harness. The active baseline before EXP-035 was 93.70%, and the goal requires at least a +0.10 percentage-point absolute gain, so this experiment needed `best_test_acc >= 93.80%` to count as an improvement.

## Idea & Hypothesis
EXP-033 and EXP-034 each reached 93.79% with one-scalar changes: lowering label smoothing to 0.03 and moving the first LR drop to 22k. EXP-035 combined those two near-misses to test whether sharper class separation plus a slightly longer high-LR phase could clear the 93.80% threshold.

## Approach
The implementation changed only `train.py`: `LR_MILESTONES` moved from `[21000, 64000]` to `[22000, 64000]`, and the training loss changed from `label_smoothing=0.05` to `label_smoothing=0.03`. The 28/56/112 ResNet-20 architecture, reflection crop padding, batch size 128, optimizer, weight decay, FP32 channels-last compile path, seed, fixed budget, and once-per-epoch evaluation were preserved. There were no deviations from the plan.

## Execution
One local single-GPU run completed cleanly on 2026-06-09. Startup confirmed CUDA execution, `822,790` parameters, a 300s training budget, and 390 batches per epoch. The schedule behaved as intended: step 21000 stayed at `lr: 0.1000`, step 22000 dropped to `lr: 0.0100`, and the second milestone at 64000 was not reached.

## Results
- **Primary metric**: 93.63% (baseline: 93.70%, delta: -0.07 points, -0.07%)
- **Observations**: The run completed 107 epochs and 41,444 steps with `final_test_acc=93.36%`, `final_test_loss=0.2340`, `total_seconds=403.4`, and `peak_vram_mb=660.4`.
- **Analysis**: The coupled change was worse than both single-axis near-misses. Lower smoothing and a later first drop appear to interfere rather than compose constructively on the current label-smoothed reflection anchor.
- **Key Learning**: Combining the two 93.79 near-misses regressed to 93.63%, so their effects are not additive.

## Verification
- **Conditions**: One necessary condition failed: `best_test_acc` did not reach the 93.80% improvement threshold.
- **Review Notes**: Results are trustworthy. The process exited cleanly, produced numeric metrics, preserved the fixed budget and architecture, used the planned schedule, modified only `train.py`, and showed no error, OOM, NaN, or Inf signatures.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid completed run, but the primary metric was below both the 93.70% baseline and the 93.80% threshold required by the goal.

## Unexplored Avenues
- Keep `label_smoothing=0.05` and try a distinct lever such as batch size 112; this avoids stacking two local near-misses that now appear non-additive.
- Revisit late averaging only with a short bounded window or lower-frequency updates; EXP-021 showed naive long equal averaging fails, but a constrained variant could preserve throughput.

## Next Steps
Move to a distinct no-overhead or low-overhead lever rather than further adjacent smoothing/schedule composition. Batch size 112 is a reasonable medium-confidence next probe because it changes gradient noise less aggressively than the failed batch-size-96 run while preserving the current anchor. Stronger smoothing to 0.08 is lower confidence because lower smoothing already produced the closest smoothing-only result.

## Exit Action Results
