# EXP-039: Increase Weight Decay to 3e-4

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-039.md
- **Plan**: plans/plan-039.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-039
- **Commit**: (pending - committed on loop success)
- **PR**: (pending - created on loop success)
- **Outcome**: completed - no-improvement

## Implementation Notes

### Summary

Implemented EXP-039 exactly as planned by changing only `WEIGHT_DECAY` in `train.py` from `2e-4` to `3e-4`. The reflection-padding 28/56/112 ResNet-20 anchor, batch size 128, `LR_MILESTONES = [21000, 64000]`, optimizer class, momentum, `label_smoothing=0.05`, FP32 channels-last compile path, seed, fixed training budget, and once-per-epoch validation path were preserved. Preflight checks passed for Python syntax, ruff, diff scope, stronger weight decay, anchor settings, and validation cadence.

### Surprises & Discoveries

No implementation surprises. Weight decay remains a single top-level constant used directly in SGD optimizer construction.

### Decisions

No deviations from the plan were needed. This run intentionally preserves the EXP-038 anchor to isolate whether stronger shrinkage continues to help.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local command session 98500; shell PID 2483823
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-09 08:06 UTC
- **Ended**: 2026-06-09 08:13 UTC

Description:
- Run the reflection-padding 28/56/112 ResNet-20 anchor with only `WEIGHT_DECAY` changed from `2e-4` to `3e-4`. The experiment tests whether the stronger-shrinkage direction validated by EXP-038 continues to improve the late post-drop plateau without changing throughput, schedule reachability, architecture, augmentation, loss smoothing, optimizer class, or evaluation cadence. It must complete under the fixed 300s training budget and under 10 minutes total wall-clock time. It must reach `best_test_acc >= 94.07%` to count as an improvement under the goal's +0.10 percentage-point rule.

Observations:
- Preflight checks passed: `python3 -m py_compile train.py`, `uv run ruff check train.py`, diff scope, `WEIGHT_DECAY = 3e-4`, preserved anchor grep, and validation-cadence grep all succeeded before launch. (source: command outputs, 2026-06-09)
- Baseline check reported `baseline=93.97`, making the concrete improvement threshold `best_test_acc >= 94.07`; both visible H20 GPUs were idle, GPU 0 was selected, and CUDA isolation reported one visible `NVIDIA H20`. (source: exp-index, nvidia-smi, CUDA smoke test, 2026-06-09)
- Startup confirms CUDA execution, `ResNet-20 | params: 822,790`, `Time budget: 300s`, and `Batches per epoch: 390`, matching the preserved batch-128 anchor. (source: run.log L1-L4)
- The planned first LR drop fired at step 21000 with `lr: 0.0100`; early post-drop accuracy reached 93.55% by epoch 62, still below the 94.07% improvement threshold. (source: `grep "step 21000" run.log`; `grep "eval ep" run.log | tail -20`)
- The late plateau did not recover; the run exited cleanly with `best_test_acc: 93.55%`, `final_test_acc: 92.67%`, `final_test_loss: 0.2613`, and `total_seconds: 407.0`. This is -0.42 below the 93.97% baseline and below the 94.07% improvement threshold. (source: final summary in run.log)

Key Metrics:
- `best_test_acc`: 93.55%
- `final_test_acc`: 92.67%
- `final_test_loss`: 0.2613
- `training_seconds`: 300.0
- `total_seconds`: 407.0
- `startup_seconds`: 3.1
- `peak_vram_mb`: 660.4
- `num_epochs`: 97
- `num_steps`: 37782
- `num_params`: 822,790

## Verification Results

### Conditions Checked
- Passed: Process exited with code 0 and completed before the 10-minute wall-clock limit (`total_seconds: 407.0`).
- Passed: The run reported a numeric `best_test_acc` of 93.55%.
- Failed: The current baseline is 93.97%, so the goal requires at least 94.07% for improvement; EXP-039 reached only 93.55%.
- Passed: Fixed training budget was preserved (`training_seconds: 300.0`).
- Passed: Architecture stayed unchanged (`num_params: 822,790`).
- Passed: Batch size stayed unchanged (`Batches per epoch: 390`).
- Passed: First LR drop occurred at step 21000 with `lr: 0.0100`.
- Passed: No second LR drop occurred; `step 64000` was absent and the run ended at 37,782 steps.
- Passed: The tracked source diff during the run was limited to the planned `train.py` weight-decay scalar change.
- Passed: No error, exception, CUDA OOM, NaN, or Inf signatures were found in `run.log`.

### Informational Metrics
- `WEIGHT_DECAY = 3e-4` over-regularized relative to the `2e-4` anchor, reducing best accuracy by 0.42 points.
- Final accuracy was 0.88 points below the peak, indicating late training did not recover after the post-drop plateau.

## Errors & Dead Ends

## Human Notes

> No human notes yet.
