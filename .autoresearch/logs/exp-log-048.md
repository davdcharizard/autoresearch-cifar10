# EXP-048: Lower BatchNorm Momentum to 0.05

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-048.md
- **Plan**: plans/plan-048.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-048
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary

Implemented the planned BatchNorm running-statistics change in `train.py` by adding `BN_MOMENTUM = 0.05` and passing it to every `nn.BatchNorm2d` layer in `BasicBlock` and the ResNet stem. All optimizer, schedule, architecture width, augmentation, label smoothing, weight decay, compile, channels-last, batch-size, and fixed-harness settings were preserved.

### Surprises & Discoveries

No implementation surprises. The current model has exactly three BatchNorm construction sites in source form: two in `BasicBlock` and one in the ResNet stem.

### Decisions

- Used the planned `0.05` BatchNorm momentum value without changing any other normalization or optimizer setting, keeping the result attributable to running-statistics update speed.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local foreground process on GPU0, shell PID 3135427, uv PID 3135428, python PID 3135431
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log`
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-09 15:34:06 UTC
- **Ended**: 2026-06-09 15:41:16 UTC

Description:
- Run the EXP-048 lower BatchNorm momentum recipe on a single local GPU with output captured to `run.log`. The experiment preserves the EXP-038 optimizer, schedule, architecture, reflection crop, smoothing, weight-decay, compile, and channels-last anchor. Expected behavior is unchanged throughput relative to the anchor, a first LR drop at step 21000, and final `best_test_acc` compared against the 94.07% threshold.

Observations:
- Preflight passed: `python3 -m py_compile train.py` exited 0 and `uv run ruff check train.py` reported `All checks passed!`.
- Diff scope confirmed: tracked code changes are limited to `train.py` and only add/pass `BN_MOMENTUM = 0.05`.
- 2026-06-09 15:34 UTC: Both H20 GPUs were idle (`0 MiB`, `0%` utilization); selected GPU0 and removed stale `run.log` immediately before launch.
- 2026-06-09 15:34 UTC: Foreground training process launched; shell PID 3135427, `uv` PID 3135428, main Python PID 3135431. Startup log confirms CUDA, expected parameter count, 300s budget, and 390 batches per epoch.
- 2026-06-09 15:35 UTC: Early training healthy through epoch 6 with best test accuracy 78.72%, no traceback/OOM patterns, and enough remaining budget to reach the step-21000 LR drop.
- 2026-06-09 15:41 UTC: Run exited cleanly with numeric final metrics. The first LR drop occurred at step 21000 (`lr: 0.0100`), and the best post-drop accuracy reached 93.48% at epoch 57.

Key Metrics:
- `best_test_acc`: 93.48%
- `final_test_acc`: 93.37%
- `final_test_loss`: 0.2270
- `training_seconds`: 300.0
- `total_seconds`: 377.9
- `startup_seconds`: 2.9
- `peak_vram_mb`: 660.4
- `num_epochs`: 61
- `num_steps`: 23449
- `num_params`: 822,790
- Classification for analysis: valid no-improvement candidate; 93.48% is below the 94.07% threshold.

## Verification Results

### Conditions Checked
- Baseline check: passed. `exp-index.sh baseline` reported `baseline=93.97`, `baseline_commit=755be2c`, `total_experiments=49`, `improvements=9`.
- Diff scope: passed. `git diff --name-only` listed only `train.py` as a tracked code change.
- Compile check: passed. `python3 -m py_compile train.py` exited 0.
- Style check: passed. `uv run ruff check train.py` reported `All checks passed!`.
- Experiment completion: passed. `run.log` reports numeric `best_test_acc: 93.48%` and total runtime 377.9 seconds, below the 600-second cap.
- First LR drop: passed. `run.log` contains `step 21000 ... lr: 0.0100` and subsequent post-drop progress lines.
- Metric extraction: passed. All expected summary metrics are present in `run.log`.
- Classification: no-improvement. The run is valid, but `best_test_acc=93.48%` is below the required `94.07%` improvement threshold.

### Informational Metrics
- `final_test_acc`: 93.37%
- `final_test_loss`: 0.2270
- `training_seconds`: 300.0
- `total_seconds`: 377.9
- `startup_seconds`: 2.9
- `peak_vram_mb`: 660.4
- `num_epochs`: 61
- `num_steps`: 23449
- `num_params`: 822,790

## Errors & Dead Ends

## Human Notes

> No human interventions during autopilot execution.
