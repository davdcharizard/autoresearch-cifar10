# EXP-036: Mild Batch Size 112 on Current Anchor

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-036.md
- **Plan**: plans/plan-036.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-036
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed - no-improvement

## Implementation Notes

### Summary

Implemented EXP-036 exactly as planned by changing only `BATCH_SIZE` in `train.py` from 128 to 112. The reflection-padding 28/56/112 ResNet-20 anchor, `label_smoothing=0.05`, `LR_MILESTONES = [21000, 64000]`, optimizer settings, FP32 channels-last compile path, seed, fixed training budget, and once-per-epoch validation path were preserved. Preflight checks passed for Python syntax, ruff, diff scope, anchor settings, batch size, and validation cadence.

### Surprises & Discoveries

No implementation surprises. The code path already centralizes the batch size in one top-level constant, and the DataLoader derives batches per epoch directly from that value.

### Decisions

No deviations from the plan were needed. The first LR milestone remains at 21000 to preserve the current label-smoothed anchor schedule, and the second milestone remains unreachable at 64000 to avoid retrying isolated second-drop retuning.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local command session 60984; shell PID 2149313
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-09 03:49 UTC
- **Ended**: 2026-06-09 03:56 UTC

Description:
- Run the reflection-padding, label-smoothed 28/56/112 ResNet-20 anchor with only `BATCH_SIZE` changed from 128 to 112. The experiment tests whether a mild smaller-batch stochasticity change can improve the current 93.70% baseline while still reaching the step-21000 first LR drop. It must complete under the fixed 300s training budget and under 10 minutes total wall-clock time. It must reach `best_test_acc >= 93.80%` to count as an improvement under the goal's +0.10 percentage-point rule.

Observations:
- Startup confirms CUDA execution, `ResNet-20 | params: 822,790`, `Time budget: 300s`, and `Batches per epoch: 446`, matching the planned batch-size geometry for `BATCH_SIZE=112`. (source: run.log L1-L4)
- Early epoch evaluations are present through epoch 12, with best test accuracy reaching 84.94% by epoch 11 and no traceback/OOM/NaN/Inf signatures found in the log. (source: `grep "eval ep" run.log | tail -10`; error-signature grep)
- The planned first LR drop fired at step 21000 with `lr: 0.0100`; no `step 64000` line is present. Post-drop accuracy reached 93.09% by epoch 52, still below the 93.80% improvement threshold. (source: `grep "step 21000" run.log`; `grep "eval ep" run.log | tail -12`)
- The run exited cleanly with `best_test_acc: 93.43%`, `final_test_acc: 93.22%`, `final_test_loss: 0.2476`, and `total_seconds: 390.3`. The mild batch-size reduction underperformed the 93.70% baseline and is a clear no-improvement result. (source: final summary in run.log)

Key Metrics:
- `best_test_acc`: 93.43%
- `final_test_acc`: 93.22%
- `final_test_loss`: 0.2476
- `training_seconds`: 300.0
- `total_seconds`: 390.3
- `startup_seconds`: 2.1
- `peak_vram_mb`: 600.0
- `num_epochs`: 90
- `num_steps`: 39859
- `num_params`: 822,790

## Verification Results

### Conditions Checked
- Passed: Process exited with code 0 and completed before the 10-minute wall-clock limit (`total_seconds: 390.3`).
- Passed: The run reported a numeric `best_test_acc` of 93.43%.
- Failed: The current baseline is 93.70%, so the goal requires at least 93.80% for improvement; EXP-036 reached 93.43%, below baseline.
- Passed: Fixed training budget was preserved (`training_seconds: 300.0`).
- Passed: Architecture stayed unchanged (`num_params: 822,790`).
- Passed: Batch-size change took effect (`Batches per epoch: 446`).
- Passed: First LR drop occurred at step 21000 with `lr: 0.0100`.
- Passed: No second LR drop occurred; `step 64000` was absent and the run ended at 39,859 steps.
- Passed: The tracked source diff during the run was limited to the planned `train.py` batch-size scalar change.
- Passed: No error, exception, CUDA OOM, NaN, or Inf signatures were found in `run.log`.

### Informational Metrics
- The smaller batch reduced epoch count and step budget versus the batch-128 anchor, completing 90 epochs and 39,859 steps.
- Peak accuracy plateaued at 93.43%, 0.27 points below the baseline and 0.37 points below the required improvement threshold.

## Errors & Dead Ends

## Human Notes

> No human notes yet.
