# EXP-034: Move First LR Drop to 22k on Label-Smoothed Anchor

## Execution

Overall Status & Info:
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-034.md
- **Plan**: plans/plan-034.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-034
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed - no-improvement

## Implementation Notes

### Summary

Implemented EXP-034 exactly as planned by changing only `LR_MILESTONES` in `train.py` from `[21000, 64000]` to `[22000, 64000]`. The reflection-padding 28/56/112 ResNet-20 anchor, `label_smoothing=0.05`, batch size, optimizer settings, FP32 channels-last compile path, seed, and once-per-epoch validation path were preserved. Preflight checks passed for Python syntax, ruff, diff scope, validation cadence, reflected padding, label-smoothing preservation, and schedule update.

### Surprises & Discoveries

No implementation surprises. The change was a one-scalar schedule edit to the current EXP-032 anchor.

### Decisions

No deviations from the plan were needed. The second milestone remains at 64000 so the experiment isolates the first-drop timing and does not retry the failed second-drop family.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local command session 86821
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-08 21:09 UTC
- **Ended**: 2026-06-08 21:16 UTC

Description:
- Run the EXP-032 reflection-padding, label-smoothed 28/56/112 ResNet-20 anchor with only the first LR drop moved from step 21000 to step 22000. The experiment tests whether the label-smoothed model benefits from slightly more high-LR fitting before LR 0.01 refinement. The run must complete under the fixed 300s training budget and under 10 minutes total wall-clock time. It must reach `best_test_acc >= 93.80%` to count as an improvement over the current `93.70%` baseline.

Observations:
- Startup confirms CUDA execution, `ResNet-20 | params: 822,790`, `Time budget: 300s`, and `Batches per epoch: 390`. GPU 0 is active for the run. (source: run.log L1-L4)
- Early epoch evaluations are present, with best test accuracy reaching 83.34% by epoch 8 and no error/OOM/NaN/Inf signatures in the initial log. (source: `grep "eval ep" run.log | head -8`)
- The planned schedule behavior is confirmed: step 21000 remained at `lr: 0.1000`, and step 22000 dropped to `lr: 0.0100`. Post-drop accuracy reached 93.79% by epoch 64, just below the 93.80% improvement threshold. (source: run.log schedule grep and eval tail)
- Late epochs did not cross the threshold; the run exited cleanly with `best_test_acc: 93.79%`, `final_test_acc: 93.64%`, `final_test_loss: 0.2443`, and `total_seconds: 400.5`. The +0.09 point gain over baseline is no-improvement under the +0.10 rule.

Key Metrics:
- `best_test_acc`: 93.79%
- `final_test_acc`: 93.64%
- `final_test_loss`: 0.2443
- `training_seconds`: 300.0
- `total_seconds`: 400.5
- `startup_seconds`: 1.9
- `peak_vram_mb`: 660.4
- `num_epochs`: 107
- `num_steps`: 41412
- `num_params`: 822,790

## Verification Results

### Conditions Checked
- Passed: Process exited with code 0 and completed before the 10-minute wall-clock limit (`total_seconds: 400.5`).
- Passed: The run reported a numeric `best_test_acc` of 93.79%.
- Failed: The current baseline is 93.70%, so the goal requires at least 93.80% for improvement; EXP-034 reached 93.79%, which is only +0.09 points.
- Passed: Fixed training budget was preserved (`training_seconds: 300.0`).
- Passed: Architecture stayed unchanged (`num_params: 822,790`).
- Passed: Batch size stayed unchanged (`Batches per epoch: 390`).
- Passed: Step 21000 remained at `lr: 0.1000`.
- Passed: First LR drop occurred at step 22000 with `lr: 0.0100`.
- Passed: No second LR drop occurred; `step 64000` was absent and the run ended at 41,412 steps.
- Passed: The tracked source diff during the run was limited to the planned `train.py` LR milestone change.
- Passed: No error, exception, CUDA OOM, NaN, or Inf signatures were found in `run.log`.

### Informational Metrics
- Moving the first drop to 22k matched EXP-033's 93.79% best but did not clear the tightened threshold.
- Final accuracy was 0.15 points below the best accuracy, with late epochs plateauing below the required 93.80%.

## Errors & Dead Ends

## Human Notes

> No human notes yet.
