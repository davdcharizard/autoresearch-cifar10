# EXP-041: Weight Decay 1.5e-4 Local Bracket

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-041.md
- **Plan**: plans/plan-041.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-041
- **Commit**: (pending - committed on loop success)
- **PR**: (pending - skipped if no remote exists)
- **Outcome**: completed - no-improvement

## Implementation Notes

### Summary

Implemented the planned isolated scalar change in `train.py`: `WEIGHT_DECAY` was reduced from `2e-4` to `1.5e-4`. The rest of the current anchor was preserved, including the 28/56/112 width, batch size 128, LR 0.1, milestones `[21000, 64000]`, momentum 0.9, reflection crop padding, label smoothing 0.05, FP32 channels-last compile path, fixed seed, and once-per-epoch evaluation.

### Surprises & Discoveries

No implementation surprises. The planned change was a one-line hyperparameter edit, and local syntax/lint/anchor checks passed.

### Decisions

No deviations from the plan. The experiment intentionally leaves optimizer class, schedule, model shape, augmentation, loss smoothing, and evaluation cadence unchanged so the result isolates the `1.5e-4` weight-decay bracket.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local PID 2560661 (`uv run train.py`; Python worker PID 2560664)
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log`
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-09 08:36 UTC
- **Ended**: 2026-06-09 08:44 UTC

Description:
- Run the current CIFAR-10 training harness locally on a single selected GPU with only `WEIGHT_DECAY = 1.5e-4` changed. This tests whether a slightly softer decay than the EXP-038 `2e-4` best can reduce over-shrinkage while retaining the stronger-regularization gain. The success threshold is `best_test_acc >= 94.07%`, because the active baseline is 93.97% and the goal requires a +0.10 percentage-point improvement.

Observations:
- Startup is clean: `run.log` reports CUDA device, `ResNet-20 | params: 822,790`, `Time budget: 300s`, and `Batches per epoch: 390`. (source: `run.log` L1-L4)
- Early and mid-run training stayed finite with best pre-drop accuracy reaching 89.19% by epoch 53. The planned first LR drop occurred at step 21000 with `lr: 0.0100`, followed by a jump to 91.58% at epoch 54. (source: `run.log` L104-L112)
- Post-drop accuracy reached 93.49% by epoch 62, below the 94.07% improvement threshold at this point. (source: `run.log` L114-L128)
- The late plateau peaked at 93.61% by epoch 72 and did not recover. The run exited cleanly with `best_test_acc: 93.61%`, `final_test_acc: 93.18%`, `final_test_loss: 0.2535`, `training_seconds: 300.0`, and `total_seconds: 404.4`. (source: `run.log` L148-L229)

Key Metrics:
- `best_test_acc`: 93.61%
- `final_test_acc`: 93.18%
- `final_test_loss`: 0.2535
- `training_seconds`: 300.0
- `total_seconds`: 404.4
- `startup_seconds`: 3.0
- `peak_vram_mb`: 660.4
- `num_epochs`: 107
- `num_steps`: 41,353
- `num_params`: 822,790

## Verification Results

### Conditions Checked
- Passed: Process exited with code 0 and completed before the 10-minute wall-clock limit (`total_seconds: 404.4`).
- Passed: The run reported a numeric `best_test_acc` of 93.61%.
- Failed: The current baseline is 93.97%, so the goal requires at least 94.07% for improvement; EXP-041 reached only 93.61%.
- Passed: Fixed training budget was preserved (`training_seconds: 300.0`).
- Passed: Architecture stayed unchanged (`num_params: 822,790`).
- Passed: Batch size stayed unchanged (`Batches per epoch: 390`).
- Passed: First LR drop occurred at step 21000 with `lr: 0.0100`.
- Passed: No second LR drop occurred; `step 64000` was absent and the run ended at 41,353 steps.
- Passed: The tracked source diff during the run was limited to the planned `train.py` weight-decay scalar change.
- Passed: No error, exception, CUDA OOM, NaN, or Inf signatures were found in `run.log`.

### Informational Metrics
- `WEIGHT_DECAY = 1.5e-4` reached only 93.61%, below the 93.97% `2e-4` anchor.
- Final accuracy was 0.43 points below the peak, indicating the late plateau drifted below the best epoch.

## Errors & Dead Ends

## Human Notes

> No human notes yet.

<!-- NOTE: Human notes are high trust and privileged relative to other info in this document. -->
