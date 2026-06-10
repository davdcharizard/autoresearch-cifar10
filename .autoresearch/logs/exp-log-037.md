# EXP-037: Stronger Label Smoothing 0.08

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-037.md
- **Plan**: plans/plan-037.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-037
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed - no-improvement

## Implementation Notes

### Summary

Implemented EXP-037 exactly as planned by changing only the training loss smoothing scalar in `train.py` from `label_smoothing=0.05` to `label_smoothing=0.08`. The reflection-padding 28/56/112 ResNet-20 anchor, batch size 128, `LR_MILESTONES = [21000, 64000]`, optimizer settings, FP32 channels-last compile path, seed, fixed training budget, and once-per-epoch validation path were preserved. Preflight checks passed for Python syntax, ruff, diff scope, stronger smoothing, anchor settings, and validation cadence.

### Surprises & Discoveries

No implementation surprises. The smoothing value is isolated in the training loss call, so the experiment is a one-scalar change.

### Decisions

No deviations from the plan were needed. The schedule and batch geometry are intentionally preserved to isolate smoothing strength.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local command session 66095; shell PID 2167618
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-09 04:03 UTC
- **Ended**: 2026-06-09 04:10 UTC

Description:
- Run the reflection-padding 28/56/112 ResNet-20 anchor with only the training loss smoothing changed from 0.05 to 0.08. The experiment tests whether stronger confidence regularization improves the current 93.70% baseline without changing throughput, schedule reachability, architecture, augmentation, optimizer, or evaluation cadence. It must complete under the fixed 300s training budget and under 10 minutes total wall-clock time. It must reach `best_test_acc >= 93.80%` to count as an improvement under the goal's +0.10 percentage-point rule.

Observations:
- Startup confirms CUDA execution, `ResNet-20 | params: 822,790`, `Time budget: 300s`, and `Batches per epoch: 390`, matching the preserved batch-128 anchor. (source: run.log L1-L4)
- Early epoch evaluations are present through epoch 9, with best test accuracy reaching 83.59% by epoch 8 and no traceback/OOM/NaN/Inf signatures found in the log. (source: `grep "eval ep" run.log | tail -10`; error-signature grep)
- The planned first LR drop fired at step 21000 with `lr: 0.0100`; no `step 64000` line is present. Post-drop accuracy reached 93.47% by epoch 61, still below the 93.80% improvement threshold. (source: `grep "step 21000" run.log`; `grep "eval ep" run.log | tail -15`)
- The run exited cleanly with `best_test_acc: 93.73%`, `final_test_acc: 93.34%`, `final_test_loss: 0.2682`, and `total_seconds: 402.3`. This is +0.03 over the 93.70% baseline but below the 93.80% improvement threshold, so it is a valid no-improvement. (source: final summary in run.log)

Key Metrics:
- `best_test_acc`: 93.73%
- `final_test_acc`: 93.34%
- `final_test_loss`: 0.2682
- `training_seconds`: 300.0
- `total_seconds`: 402.3
- `startup_seconds`: 3.1
- `peak_vram_mb`: 660.4
- `num_epochs`: 104
- `num_steps`: 40315
- `num_params`: 822,790

## Verification Results

### Conditions Checked
- Passed: Process exited with code 0 and completed before the 10-minute wall-clock limit (`total_seconds: 402.3`).
- Passed: The run reported a numeric `best_test_acc` of 93.73%.
- Failed: The current baseline is 93.70%, so the goal requires at least 93.80% for improvement; EXP-037 reached 93.73%, inside the noise band.
- Passed: Fixed training budget was preserved (`training_seconds: 300.0`).
- Passed: Architecture stayed unchanged (`num_params: 822,790`).
- Passed: Batch size stayed unchanged (`Batches per epoch: 390`).
- Passed: First LR drop occurred at step 21000 with `lr: 0.0100`.
- Passed: No second LR drop occurred; `step 64000` was absent and the run ended at 40,315 steps.
- Passed: The tracked source diff during the run was limited to the planned `train.py` label-smoothing scalar change.
- Passed: No error, exception, CUDA OOM, NaN, or Inf signatures were found in `run.log`.

### Informational Metrics
- Stronger smoothing reached 93.73%, a small +0.03 point lift over baseline but still below the +0.10 point improvement threshold.
- Final accuracy was 0.39 points below the peak, indicating the best value came from the late post-drop plateau rather than the final epoch.

## Errors & Dead Ends

## Human Notes

> No human notes yet.
