# EXP-054: Very Mild Residual Stochastic Depth

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-054.md
- **Plan**: plans/plan-054.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-054
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary

Implemented the planned very mild stochastic-depth probe in `train.py`. The patch adds `STOCHASTIC_DEPTH_MAX_P = 0.03`, stores a per-block drop probability in `BasicBlock`, applies a training-only per-sample residual-branch mask before shortcut addition, and assigns linearly increasing probabilities across the nine residual blocks. The evaluation path uses the full deterministic network because the mask is gated by `self.training`.

### Surprises & Discoveries

No implementation surprises. The change stayed within the existing ResNet block structure and did not require altering optimizer, schedule, augmentation, or evaluation code.

### Decisions

- Used per-sample masks shaped `(batch, 1, 1, 1)` so stochastic depth drops each residual branch consistently across spatial positions for each sample.
- Scaled by keep probability to preserve expected residual magnitude during training.
- Kept the first block at zero drop probability and the final block at 0.03 to make the intervention conservative for this shallow fixed-budget model.
- Added a startup print for `Stochastic depth max p` so the run log can verify the experimental config directly.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local foreground process on GPU1, session 70048, shell PID 3383052, uv PID 3383053, main python PID 3383056
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log`
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-09 17:02:49 UTC
- **Ended**: 2026-06-09 17:09:50 UTC

Description:
- Run the EXP-054 stochastic-depth probe on a single local GPU with output captured to `run.log`. The run tests whether very mild train-time residual branch dropping improves generalization without changing parameter count or evaluation behavior. Expected behavior is startup reporting `Stochastic depth max p: 0.03`, `Batches per epoch: 390`, unchanged `num_params=822,790`, first LR drop at step 21000 with `lr: 0.0100`, and final `best_test_acc` compared against the 94.07% threshold.

Observations:
- Preflight passed: tracked diff is limited to `train.py`, `python3 -m py_compile train.py` exited 0, and `uv run ruff check train.py` reported `All checks passed!`.
- Baseline check passed before launch: `baseline=93.97`, `baseline_commit=755be2c`, so the improvement threshold is `94.07%`.
- 2026-06-09 17:02 UTC: GPU1 selected. Pre-launch `nvidia-smi` showed GPU1 at 0 MiB / 0% utilization, while GPU0 was active.
- 2026-06-09 17:02 UTC: Foreground session launched on GPU1. Startup log confirms CUDA, `num_params=822,790`, `Stochastic depth max p: 0.03`, 300s budget, and `Batches per epoch: 390`. No traceback/OOM/nan/inf patterns observed at startup.
- 2026-06-09 17:03 UTC: Early training healthy through epoch 10 with best test accuracy 81.87%. Step timing is mostly 7-8ms, LR remains 0.1000, and no error patterns are present.
- 2026-06-09 17:04 UTC: Mid pre-drop training healthy through epoch 22 with best test accuracy 87.04%. Throughput remains sufficient for the step-21000 LR transition.
- 2026-06-09 17:06 UTC: First LR transition confirmed. The progress line at `step 21000` reports `lr: 0.0100`; post-drop convergence reached 93.12% by epoch 60, still below the 94.07% threshold.
- 2026-06-09 17:09 UTC: Run exited cleanly with final summary metrics present. Best accuracy plateaued at 93.40% from epoch 71 onward and did not approach the 94.07% threshold.

Key Metrics:
- `best_test_acc`: 93.40%
- `final_test_acc`: 92.61%
- `final_test_loss`: 0.2717
- `training_seconds`: 300.0
- `total_seconds`: 398.4
- `startup_seconds`: 2.5
- `peak_vram_mb`: 660.9
- `num_epochs`: 101
- `num_steps`: 39,018
- `num_params`: 822,790

## Verification Results

### Conditions Checked
- Baseline check: passed. `exp-index.sh baseline` reported `baseline=93.97` and `baseline_commit=755be2c`, so the improvement threshold is 94.07%.
- Scope check: passed. The tracked code diff is limited to `train.py`; `.autoresearch/` artifacts are local loop metadata.
- Compile check: passed. `python3 -m py_compile train.py` exited 0.
- Style check: passed. `uv run ruff check train.py` reported `All checks passed!`.
- Run completion: passed. `run.log` reports numeric summary metrics after a clean 300.0s training run and 398.4s total runtime.
- Stochastic-depth config: passed. Startup reports `Stochastic depth max p: 0.03`.
- Batch geometry: passed. Startup reports `Batches per epoch: 390`.
- LR milestone: passed. The `step 21000` progress line reports `lr: 0.0100`.
- Parameter count: passed. Final summary reports `num_params: 822,790`.
- Classification: passed. `best_test_acc=93.40%` is below the 94.07% threshold, so verdict is valid no-improvement.

### Informational Metrics
- Final test accuracy: 92.61%.
- Final test loss: 0.2717.
- Training seconds: 300.0.
- Total seconds: 398.4.
- Startup seconds: 2.5.
- Peak VRAM: 660.9 MB.
- Epochs completed: 101.
- Steps completed: 39,018.

## Errors & Dead Ends

## Human Notes

> No human interventions during autopilot execution.
