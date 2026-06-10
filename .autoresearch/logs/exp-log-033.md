# EXP-033: Lower Label Smoothing to 0.03

## Execution

Overall Status & Info:
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-033.md
- **Plan**: plans/plan-033.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-033
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed - no-improvement

## Implementation Notes

### Summary

Implemented EXP-033 exactly as planned by changing only the training loss smoothing value in `train.py` from `label_smoothing=0.05` to `label_smoothing=0.03`. The reflection-padding 28/56/112 ResNet-20 anchor, batch size, optimizer settings, LR schedule, FP32 channels-last compile path, seed, and once-per-epoch validation path were preserved. Preflight checks passed for Python syntax, ruff, diff scope, validation cadence, reflected padding, lower label-smoothing presence, and schedule preservation.

### Surprises & Discoveries

No implementation surprises. The change was a one-scalar edit to the existing EXP-032 loss call.

### Decisions

No deviations from the plan were needed. Kept the experiment isolated to the smoothing value so any result can be attributed to whether 0.03 is a better point on the newly validated label-smoothing axis.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local command session 86529
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-08 20:55 UTC
- **Ended**: 2026-06-08 21:02 UTC

Description:
- Run the EXP-032 reflection-padding, label-smoothed 28/56/112 ResNet-20 anchor with only the cross-entropy smoothing value changed from 0.05 to 0.03. The experiment tests whether a slightly milder confidence regularizer preserves the late-stability gain while improving peak class separation. The run must complete under the fixed 300s training budget and under 10 minutes total wall-clock time. It must reach `best_test_acc >= 93.80%` to count as an improvement over the current `93.70%` baseline.

Observations:
- Startup confirms CUDA execution, `ResNet-20 | params: 822,790`, `Time budget: 300s`, and `Batches per epoch: 390`. GPU 0 is active for the run. (source: run.log L1-L4)
- Early epoch evaluations are present, with best test accuracy reaching 61.93% by epoch 3 and no error/OOM/NaN/Inf signatures in the initial log. (source: run.log startup excerpt)
- Pre-LR-drop training remains healthy through epoch 19, with best test accuracy at 86.97% and no error signatures. (source: `grep "eval ep" run.log | tail -12`)
- The planned first LR drop fired at step 21000 with `lr: 0.0100`; post-drop accuracy climbed to 93.52% by epoch 62, still below the 93.80% improvement threshold. (source: run.log step 21000 and eval tail)
- Late refinement reached 93.77% by epoch 91, above the 93.70 baseline but still below the required 93.80% threshold. Final summary remains pending. (source: run.log eval tail)
- The run exited cleanly with `best_test_acc: 93.79%`, `final_test_acc: 93.57%`, `final_test_loss: 0.2289`, and `total_seconds: 398.4`. The +0.09 point gain over baseline does not clear the +0.10 point rule, so the result is no-improvement.

Key Metrics:
- `best_test_acc`: 93.79%
- `final_test_acc`: 93.57%
- `final_test_loss`: 0.2289
- `training_seconds`: 300.0
- `total_seconds`: 398.4
- `startup_seconds`: 1.8
- `peak_vram_mb`: 660.4
- `num_epochs`: 108
- `num_steps`: 41773
- `num_params`: 822,790

## Verification Results

### Conditions Checked
- Passed: Process exited with code 0 and completed before the 10-minute wall-clock limit (`total_seconds: 398.4`).
- Passed: The run reported a numeric `best_test_acc` of 93.79%.
- Failed: The current baseline is 93.70%, so the goal requires at least 93.80% for improvement; EXP-033 reached 93.79%, which is only +0.09 points.
- Passed: Fixed training budget was preserved (`training_seconds: 300.0`).
- Passed: Architecture stayed unchanged (`num_params: 822,790`).
- Passed: Batch size stayed unchanged (`Batches per epoch: 390`).
- Passed: First LR drop occurred at step 21000 with `lr: 0.0100`.
- Passed: No second LR drop occurred; `step 64000` was absent and the run ended at 41,773 steps.
- Passed: The tracked source diff during the run was limited to the planned `train.py` label-smoothing scalar change.
- Passed: No error, exception, CUDA OOM, NaN, or Inf signatures were found in `run.log`.

### Informational Metrics
- `label_smoothing=0.03` nearly matched the improvement threshold but finished lower than the 0.05 anchor's required successor threshold.
- Final accuracy was 0.22 points below the best accuracy, suggesting the milder setting may allow a slightly sharper but less stable late peak than 0.05.

## Errors & Dead Ends

## Human Notes

> No human notes yet.
