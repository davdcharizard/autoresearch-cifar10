# EXP-055: Reliable Mild Mixup Retry

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-055.md
- **Plan**: plans/plan-055.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-055
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary

Implemented the planned direct retry of EXP-042 mixup in `train.py`. The patch adds `MIXUP_ALPHA = 0.1`, prints the active alpha at startup, constructs a `torch.distributions.Beta` sampler once before training, samples one lambda per batch, mixes inputs on-device with a random batch permutation, and trains with weighted two-target cross entropy while preserving the existing `label_smoothing=0.05`.

### Surprises & Discoveries

No code-structure surprises. The current training loop already has a clean point after device transfer and before model forward where mixup can be inserted without touching data loading, evaluation, optimizer setup, or model code.

### Decisions

- Kept `MIXUP_ALPHA=0.1` exactly as in EXP-042 to make the retry scientifically comparable to the prior crash-only attempt.
- Used one scalar lambda per batch rather than per-sample lambdas to keep the implementation simple and minimize overhead.
- Left evaluation unchanged, so the measured `best_test_acc` comes from the standard deterministic model and fixed `Eval.evaluate()` harness.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local foreground process on GPU0, session 27907, shell PID 3408952, uv PID 3408960, main python PID 3408964
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log`
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-09 17:17:49 UTC
- **Ended**: 2026-06-09 17:25:15 UTC

Description:
- Run the EXP-055 mild mixup retry on one local GPU with output captured to `run.log`. This tests whether `MIXUP_ALPHA=0.1` can produce a completed post-drop result under the now-reliable foreground launch path. Expected behavior is startup reporting `Mixup alpha: 0.1`, unchanged `Batches per epoch: 390`, unchanged `num_params=822,790`, first LR drop at step 21000 with `lr: 0.0100`, and final `best_test_acc` compared against the 94.07% threshold.

Observations:
- Preflight passed: tracked diff is limited to `train.py`, `python3 -m py_compile train.py` exited 0, and `uv run ruff check train.py` reported `All checks passed!`.
- Baseline check passed before launch: `baseline=93.97`, `baseline_commit=755be2c`, so the improvement threshold is `94.07%`.
- 2026-06-09 17:17 UTC: Both H20 GPUs were free (`0MiB`, `0%` utilization, no running processes). GPU0 selected for EXP-055.
- 2026-06-09 17:18 UTC: Foreground process tree confirmed in the project cwd for shell PID 3408952, uv PID 3408960, and Python PID 3408964.
- 2026-06-09 17:18 UTC: Startup log confirms CUDA, `num_params=822,790`, `Mixup alpha: 0.1`, 300s budget, and `Batches per epoch: 390`. Early epoch-1 throughput is 7-9ms/batch and no traceback/OOM/nan/inf patterns are present.
- 2026-06-09 17:19 UTC: Run is healthy through epoch 17 with best test accuracy 85.11%. Step timing remains mostly 7-8ms and the first LR drop remains reachable.
- 2026-06-09 17:20 UTC: Pre-drop training is healthy through epoch 38 with best test accuracy 88.30%. The run is around step 15k, still at `lr: 0.1000`, with enough remaining budget for the 21k transition.
- 2026-06-09 17:21 UTC: First LR transition confirmed. The progress line at `step 21000` reports `lr: 0.0100`; post-drop convergence reached 93.37% by epoch 58.
- 2026-06-09 17:25 UTC: Run exited cleanly with final summary metrics present. Best accuracy peaked at 93.85% at epoch 95, below the 94.07% improvement threshold.

Key Metrics:
- `best_test_acc`: 93.85%
- `final_test_acc`: 93.48%
- `final_test_loss`: 0.2594
- `training_seconds`: 300.0
- `total_seconds`: 394.5
- `startup_seconds`: 2.5
- `peak_vram_mb`: 661.9
- `num_epochs`: 97
- `num_steps`: 37,547
- `num_params`: 822,790

## Verification Results

### Conditions Checked
- Baseline check: passed. `exp-index.sh baseline` reported `baseline=93.97` and `baseline_commit=755be2c`, so the improvement threshold is 94.07%.
- Scope check: passed. The tracked code diff is limited to `train.py`; `.autoresearch/` artifacts are local loop metadata.
- Compile check: passed. `python3 -m py_compile train.py` exited 0.
- Style check: passed. `uv run ruff check train.py` reported `All checks passed!`.
- Run completion: passed. `run.log` reports numeric summary metrics after a clean 300.0s training run and 394.5s total runtime.
- Mixup config: passed. Startup reports `Mixup alpha: 0.1`.
- Batch geometry: passed. Startup reports `Batches per epoch: 390`.
- LR milestone: passed. The `step 21000` progress line reports `lr: 0.0100`.
- Parameter count: passed. Final summary reports `num_params: 822,790`.
- Classification: passed. `best_test_acc=93.85%` is below the 94.07% threshold, so verdict is valid no-improvement.

### Informational Metrics
- Final test accuracy: 93.48%.
- Final test loss: 0.2594.
- Training seconds: 300.0.
- Total seconds: 394.5.
- Startup seconds: 2.5.
- Peak VRAM: 661.9 MB.
- Epochs completed: 97.
- Steps completed: 37,547.

## Errors & Dead Ends

## Human Notes

> No human interventions during autopilot execution.
