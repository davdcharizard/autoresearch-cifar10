# EXP-053: Batch Size 160 Probe

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-053.md
- **Plan**: plans/plan-053.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-053
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary

Implemented the planned larger-batch probe in `train.py` by changing only `BATCH_SIZE` from 128 to 160. All anchor settings besides batch size remain unchanged: model widths, optimizer, LR milestones, momentum, weight decay, reflection crop padding, label smoothing, compile, channels-last, and once-per-epoch validation.

### Surprises & Discoveries

No implementation surprises. The diff is a single constant change, which keeps the experiment tightly scoped to batch geometry.

### Decisions

- Kept step-based LR milestones unchanged at `[21000, 64000]` to isolate the effect of the larger batch on update count, epoch coverage, and first-drop reachability.
- Kept validation cadence unchanged so the fixed harness comparison remains directly comparable to the current anchor.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local foreground process on GPU0, session 28313, shell PID 3352190, uv PID 3352191, main python PID 3352194
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log`
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-09 16:47:49 UTC
- **Ended**: 2026-06-09 16:55:41 UTC

Description:
- Run the EXP-053 batch-size 160 probe on a single local GPU with output captured to `run.log`. The run tests whether increasing batch size from 128 to 160 improves the fixed-time tradeoff between image coverage, update count, and gradient stability while preserving the current anchor recipe. Expected behavior is startup reporting `Batches per epoch: 312`, unchanged `num_params=822,790`, first LR drop at step 21000 with `lr: 0.0100`, and final `best_test_acc` compared against the 94.07% threshold.

Observations:
- Preflight passed: tracked diff is limited to `train.py`, `python3 -m py_compile train.py` exited 0, and `uv run ruff check train.py` reported `All checks passed!`.
- Baseline check passed before launch: `baseline=93.97`, `baseline_commit=755be2c`, so the improvement threshold is `94.07%`.
- 2026-06-09 16:47 UTC: GPU0 selected. Pre-launch `nvidia-smi` showed GPU0 at 0 MiB / 0% utilization; GPU1 had a separate sibling-repo run. Stale `run.log` was removed immediately before launch.
- 2026-06-09 16:48 UTC: Foreground session launched on GPU0. Startup log confirms CUDA, `num_params=822,790`, 300s budget, and `Batches per epoch: 312`, matching the batch-size 160 geometry. No traceback/OOM/nan/inf patterns observed at startup.
- 2026-06-09 16:49 UTC: Early training healthy through epoch 12 with best test accuracy 84.56%. Step timing is mostly 7-8ms despite the larger batch, and LR remains at 0.1000 before the planned step-21000 transition.
- 2026-06-09 16:50 UTC: Mid pre-drop training healthy through epoch 38 with best test accuracy 88.53%. The run remains on LR 0.1000 with no error patterns and enough remaining budget to reach step 21000.
- 2026-06-09 16:52 UTC: First LR transition confirmed. The progress line at `step 21000` reports `lr: 0.0100`; post-drop convergence reached 93.45% by epoch 77, still below the 94.07% threshold.
- 2026-06-09 16:55 UTC: Run exited cleanly with final summary metrics present. The best accuracy was 93.71% at epoch 89, below the 94.07% improvement threshold, so the experiment is a valid no-improvement.

Key Metrics:
- `best_test_acc`: 93.71%
- `final_test_acc`: 93.31%
- `final_test_loss`: 0.2532
- `training_seconds`: 300.0
- `total_seconds`: 416.2
- `startup_seconds`: 2.4
- `peak_vram_mb`: 785.4
- `num_epochs`: 118
- `num_steps`: 36,597
- `num_params`: 822,790

## Verification Results

### Conditions Checked
- Baseline check: passed. `exp-index.sh baseline` reported `baseline=93.97` and `baseline_commit=755be2c`, so the improvement threshold is 94.07%.
- Scope check: passed. The tracked code diff is limited to `train.py`; the only code change is `BATCH_SIZE = 128` to `BATCH_SIZE = 160`.
- Compile check: passed. `python3 -m py_compile train.py` exited 0.
- Style check: passed. `uv run ruff check train.py` reported `All checks passed!`.
- Run completion: passed. `run.log` reports numeric summary metrics after a clean 300.0s training run and 416.2s total runtime.
- Batch geometry: passed. Startup reports `Batches per epoch: 312`.
- LR milestone: passed. The `step 21000` progress line reports `lr: 0.0100`.
- Parameter count: passed. Final summary reports `num_params: 822,790`.
- Classification: passed. `best_test_acc=93.71%` is below the 94.07% threshold, so verdict is valid no-improvement.

### Informational Metrics
- Final test accuracy: 93.31%.
- Final test loss: 0.2532.
- Training seconds: 300.0.
- Total seconds: 416.2.
- Startup seconds: 2.4.
- Peak VRAM: 785.4 MB.
- Epochs completed: 118.
- Steps completed: 36,597.

## Errors & Dead Ends

## Human Notes

> No human interventions during autopilot execution.
