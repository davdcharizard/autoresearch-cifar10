# EXP-060: Mixup Without Additional Label Smoothing

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-060.md
- **Plan**: plans/plan-060.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-060
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary

Implemented the planned mixup/smoothing coupling test in `train.py` only. The patch adds `MIXUP_ALPHA = 0.1` and `MIXUP_LABEL_SMOOTHING = 0.0`, prints both settings at startup, constructs one beta sampler before training, samples one scalar lambda per batch, mixes images on-device after data transfer, and trains with weighted two-target cross entropy using unsmoothed endpoint labels.

### Surprises & Discoveries

No code-structure surprises. The EXP-055 implementation path still maps cleanly onto the current training loop: the batch can be mixed after device transfer and before forward pass without touching data loading, evaluation, optimizer setup, scheduler setup, or model structure.

### Decisions

- Kept `MIXUP_ALPHA=0.1` exactly aligned with EXP-055 so the experiment isolates the label-smoothing change.
- Used one scalar lambda per batch, matching the reliable EXP-055 mechanics and minimizing overhead.
- Set `MIXUP_LABEL_SMOOTHING=0.0` only in the two endpoint cross-entropy calls. This intentionally removes additional endpoint smoothing while retaining mixup's label interpolation.
- Left evaluation unchanged so `best_test_acc` remains measured by the fixed `Eval.evaluate()` harness.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local session 27257; shell PID 3527467; uv PID 3527468; main python PID 3527471
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log`
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-09 18:31:22 UTC
- **Ended**: 2026-06-09 18:38:19 UTC

Description:
- Local foreground run of EXP-060 on one selected GPU with output captured to `run.log`. This tests whether EXP-055's near-miss was caused by compounded target softening from combining mixup interpolation with `label_smoothing=0.05`. Expected behavior is startup reporting `Mixup alpha: 0.1, mixup label smoothing: 0.0`, unchanged batch geometry, unchanged parameter count, first LR drop at step 21000, and final `best_test_acc` classified against the 94.07% improvement threshold.

Observations:
- Preflight passed: tracked diff is limited to `train.py`, `python3 -m py_compile train.py` exited 0, and `uv run ruff check train.py` reported `All checks passed!`.
- Baseline for classification: `93.97%`; improvement threshold: `94.07%`.
- 2026-06-09 18:31 UTC: GPU0 selected after `nvidia-smi` showed GPU0 at `0MiB` and `0%` utilization; GPU1 was busy.
- 2026-06-09 18:31 UTC: Foreground run launched on GPU0. Process table showed the shell, uv process, and main Python process in this project cwd.
- Startup confirmed CUDA, `ResNet-20 | params: 822,790`, `Mixup alpha: 0.1, mixup label smoothing: 0.0`, 300s budget, and `Batches per epoch: 390`.
- 2026-06-09 18:32 UTC: Early training is healthy through epoch 12 with best test accuracy 85.12%, mostly 7-8ms batch timings, no traceback/OOM/runtime-error patterns, and GPU0 active.
- 2026-06-09 18:33 UTC: First LR drop confirmed in `run.log` at `step 21000 ep 54` with `lr: 0.0100`; post-drop accuracy climbed rapidly from best 89.60% pre-drop to 93.81% by epoch 72.
- 2026-06-09 18:34 UTC: Late run remains healthy through epoch 99 with no traceback/OOM/runtime-error patterns. Best remains 93.81%, below the 94.07% improvement threshold, but the run is still being allowed to finish for final summary metrics.
- 2026-06-09 18:38 UTC: Run exited cleanly with final summary metrics. Best test accuracy was 93.81%, which is 0.16 percentage points below the 93.97% baseline and 0.26 percentage points below the 94.07% improvement threshold. Classification: valid `no-improvement`.

Key Metrics:
- `best_test_acc`: 93.81%
- `final_test_acc`: 93.04%
- `final_test_loss`: 0.2313
- `training_seconds`: 300.0
- `total_seconds`: 396.7
- `startup_seconds`: 2.3
- `peak_vram_mb`: 660.4
- `num_epochs`: 106
- `num_steps`: 41,074
- `num_params`: 822,790
- Verdict against goal: no-improvement (`93.81% < 94.07%`).

## Verification Results

### Conditions Checked
- Baseline check: passed. `exp-index.sh baseline` reported `baseline=93.97` and `baseline_commit=755be2c`; improvement threshold is 94.07%.
- Scope check: passed. `git diff --name-only` listed only `train.py`.
- Compile check: passed. `python3 -m py_compile train.py` exited 0.
- Style check: passed. `uv run ruff check train.py` reported `All checks passed!`.
- Completion check: passed. The foreground process exited 0 and `run.log` reported numeric final summary metrics.
- Mixup config check: passed. `run.log` line 3 reports `Mixup alpha: 0.1, mixup label smoothing: 0.0`.
- Batch geometry check: passed. `run.log` line 5 reports `Batches per epoch: 390`.
- LR-drop check: passed. `run.log` line 112 includes `step 21000` with `lr: 0.0100`.
- Parameter-count check: passed. `run.log` line 228 reports `num_params: 822,790`.
- Classification check: passed. `best_test_acc=93.81%` is below the 94.07% threshold, so EXP-060 is a valid no-improvement.

### Informational Metrics
- Final test accuracy: 93.04%.
- Final test loss: 0.2313.
- Training seconds: 300.0.
- Total seconds: 396.7.
- Startup seconds: 2.3.
- Peak VRAM: 660.4 MB.
- Epochs completed: 106.
- Steps completed: 41,074.

## Errors & Dead Ends

## Human Notes

> Autopilot mode; no human approval or intervention requested during execution.
