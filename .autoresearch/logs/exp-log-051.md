# EXP-051: Partial Residual-Branch BN Scale Initialization

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-051.md
- **Plan**: plans/plan-051.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-051
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary

Implemented the planned partial residual-branch initialization in `train.py`. After the normal model-wide Kaiming initialization, `ResNet.__init__` now loops over `self.modules()` and sets each `BasicBlock.bn2.weight` to `0.1`. This is an initialization-only change; architecture, optimizer, schedule, augmentation, label smoothing, compile, channels-last, batch size, and validation cadence were preserved.

### Surprises & Discoveries

No implementation surprises. The local `BasicBlock` class is in scope when `ResNet.__init__` runs, so an `isinstance(m, BasicBlock)` loop is direct and does not require helper functions or imports.

### Decisions

- Used `0.1` exactly as planned to keep residual branches active, distinguishing EXP-051 from EXP-028's full zero-gamma initialization.
- Applied the partial scale after `self.apply(self._weights_init)` so the Kaiming initialization path remains unchanged and the residual-branch BN scale override is explicit.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local foreground process on GPU0, shell PID 3289095, uv PID 3289096, main python PID 3289099
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log`
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-09 16:17:53 UTC
- **Ended**: 2026-06-09 16:25:27 UTC

Description:
- Run the EXP-051 partial residual-branch BN scale initialization on a single local GPU with output captured to `run.log`. The run tests whether `bn2.weight=0.1` can capture a milder identity-bias benefit without the undertraining seen in full zero-gamma EXP-028. Expected behavior is unchanged parameter count, clean startup, first LR drop at step 21000, and final `best_test_acc` compared against the 94.07% threshold.

Observations:
- Preflight passed: tracked diff is limited to `train.py`, `python3 -m py_compile train.py` exited 0, and `uv run ruff check train.py` reported `All checks passed!`.
- Baseline check passed before launch: `baseline=93.97`, `baseline_commit=755be2c`, so the improvement threshold is `94.07%`.
- 2026-06-09 16:17 UTC: Both GPUs were idle at launch; GPU0 selected with 0 MiB allocated and 0% utilization. Stale `run.log` was removed immediately before launch.
- 2026-06-09 16:18 UTC: Foreground session launched on GPU0; process table shows shell PID 3289095, uv PID 3289096, and main python PID 3289099 in this workspace. Startup log confirms CUDA, `num_params=822,790`, 300s budget, and 390 batches per epoch.
- 2026-06-09 16:19 UTC: Early training healthy through epoch 18 with best test accuracy 85.02%, no traceback/OOM patterns, and normal 6-8ms step timing. Partial scale does not show the severe early collapse seen in full zero-gamma EXP-028.
- 2026-06-09 16:22 UTC: First LR drop confirmed at step 21000 (`lr: 0.0100`) with about 146s remaining. Pre-drop best was 88.99% at epoch 46; after the drop, accuracy climbed to 93.39% by epochs 61-63. No traceback/OOM patterns observed.
- 2026-06-09 16:25 UTC: Run exited cleanly with final summary metrics present. Best accuracy reached 93.64% at epoch 94, below the 94.07% improvement threshold; verdict is valid no-improvement.

Key Metrics:
- `best_test_acc`: 93.64%
- `final_test_acc`: 93.12%
- `final_test_loss`: 0.2481
- `training_seconds`: 300.0
- `total_seconds`: 402.2
- `startup_seconds`: 1.9
- `peak_vram_mb`: 660.4
- `num_epochs`: 109
- `num_steps`: 42354
- `num_params`: 822,790
- Classification: no-improvement (`93.64% < 94.07%`)

## Verification Results

### Conditions Checked
- Baseline check: passed. `exp-index.sh baseline` reported `baseline=93.97`, `baseline_commit=755be2c`, `total_experiments=52`, `improvements=9`; threshold is 94.07%.
- Scope check: passed. `git diff --name-only` listed only `train.py`.
- Compile check: passed. `python3 -m py_compile train.py` exited 0.
- Style check: passed. `uv run ruff check train.py` reported `All checks passed!`.
- Run completion: passed. Foreground process exited 0, `run.log` contains numeric final summary metrics, and `total_seconds=402.2` is below the 600s cap.
- LR-drop check: passed. `run.log` contains step 21000 with `lr: 0.0100`.
- Parameter count check: passed. Final summary reports `num_params: 822,790`.
- Classification check: passed. `best_test_acc=93.64%`, which is below the 94.07% improvement threshold, so the result is a valid no-improvement.

### Informational Metrics
- Best test accuracy: 93.64%
- Final test accuracy: 93.12%
- Final test loss: 0.2481
- Runtime: 300.0s training, 402.2s total
- Startup: 1.9s
- Peak VRAM: 660.4 MB
- Epochs / steps: 109 / 42354

## Errors & Dead Ends

## Human Notes

> No human interventions during autopilot execution.
