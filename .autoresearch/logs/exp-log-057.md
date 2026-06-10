# EXP-057: Post-Drop Label Smoothing Anneal

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-057.md
- **Plan**: plans/plan-057.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-057
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary

Implemented the planned post-drop label-smoothing schedule in `train.py`. The patch adds explicit constants for pre-drop smoothing `0.05`, post-drop smoothing `0.0`, and switch step `LR_MILESTONES[0]`, prints that schedule at startup, selects the active smoothing value before each loss computation, and includes the value in the existing every-50-step progress log as `ls: ...`.

### Surprises & Discoveries

No code-structure surprises. Because `scheduler.step()` runs after `optimizer.step()`, the first progress line printed at step 21000 shows the scheduler has just dropped LR; the first batch trained entirely under the post-drop LR uses `label_smoothing=0.0` on the next optimizer step.

### Decisions

- Kept the loss schedule keyed to the existing `step` counter before increment, preserving the anchor exactly for all batches before the 21k milestone.
- Used constants instead of inline numeric branches so the startup print, loss branch, and plan verification all refer to a single source of truth.
- Added smoothing to progress logs rather than adding extra logging calls, preserving the current logging cadence.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local foreground process on GPU0, session 74577, shell PID 3459168, uv PID 3459169, main python PID 3459172
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log`
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-09 17:46:22 UTC
- **Ended**: 2026-06-09 17:53:09 UTC

Description:
- Run the EXP-057 label-smoothing schedule probe on one local GPU with output captured to `run.log`. This tests whether keeping `label_smoothing=0.05` during high-LR training and switching to `0.0` after the first LR drop improves late low-LR refinement. Expected behavior is startup reporting the schedule, pre-drop progress lines with `ls: 0.050`, post-drop progress lines with `ls: 0.000`, unchanged `Batches per epoch: 390`, unchanged `num_params=822,790`, first LR drop at step 21000 with `lr: 0.0100`, and final `best_test_acc` compared against the 94.07% threshold.

Observations:
- Preflight passed: tracked diff is limited to `train.py`, `python3 -m py_compile train.py` exited 0, and `uv run ruff check train.py` reported `All checks passed!`.
- Baseline check passed before launch: `baseline=93.97`, `baseline_commit=755be2c`, so the improvement threshold is `94.07%`.
- 2026-06-09 17:46 UTC: Both H20 GPUs were free (`0MiB`, `0%` utilization, no running processes). GPU0 selected for EXP-057.
- 2026-06-09 17:46 UTC: Foreground process tree confirmed in the project cwd for shell PID 3459168, uv PID 3459169, and Python PID 3459172.
- 2026-06-09 17:46 UTC: Startup log confirms CUDA, `num_params=822,790`, `Label smoothing schedule: pre_drop=0.05, post_drop=0.0, switch_step=21000`, 300s budget, and `Batches per epoch: 390`. No traceback/OOM/runtime/nan/inf patterns are present.
- 2026-06-09 17:47 UTC: Early training is stable through epoch 22 with progress logs showing `ls: 0.050` before the first LR drop and best test accuracy 87.74%. Step timing is mostly 6-8ms/batch and the step-21000 milestone is reachable.
- 2026-06-09 17:49 UTC: First LR drop confirmed at `step 21000 ep 54` with `lr: 0.0100`; this line still reports `ls: 0.050` because it is the batch that triggers the scheduler milestone. The following progress line at `step 21050` reports `lr: 0.0100` and `ls: 0.000`, confirming the post-drop smoothing switch. Post-drop best reached 93.42% by epoch 62, still below the 94.07% threshold.
- 2026-06-09 17:53 UTC: Run completed cleanly with `best_test_acc=93.42%`, `final_test_acc=92.96%`, `num_epochs=109`, and `num_steps=42,285`. The result is a valid no-improvement because it is below the 93.97% baseline and the required 94.07% improvement threshold.

Key Metrics:
- `best_test_acc`: 93.42%
- `final_test_acc`: 92.96%
- `final_test_loss`: 0.2869
- `training_seconds`: 300.0
- `total_seconds`: 402.3
- `startup_seconds`: 1.9
- `peak_vram_mb`: 660.4
- `num_epochs`: 109
- `num_steps`: 42,285
- `num_params`: 822,790

## Verification Results

### Conditions Checked
- Baseline: pass. `exp-index.sh baseline` reported `baseline=93.97`, `baseline_commit=755be2c`; improvement threshold is 94.07%.
- Scope: pass. `git diff --name-only` lists only `train.py`.
- Compile: pass. `python3 -m py_compile train.py` exited 0 during preflight.
- Style: pass. `uv run ruff check train.py` reported `All checks passed!` during preflight.
- Run completion: pass. Local foreground process exited 0 and `run.log` reports numeric `best_test_acc`.
- Startup schedule: pass. `run.log` reports `Label smoothing schedule: pre_drop=0.05, post_drop=0.0, switch_step=21000`.
- Active smoothing switch: pass. Pre-drop progress lines report `ls: 0.050`; post-drop progress lines after `step 21050` report `ls: 0.000`.
- Batch geometry: pass. `run.log` reports `Batches per epoch: 390`.
- LR drop: pass. `run.log` reports `step 21000 ep 54 ... lr: 0.0100`.
- Final metrics: pass. Summary metrics are present and `num_params` remains `822,790`.
- Classification: no-improvement. `best_test_acc=93.42%` is below baseline 93.97% and below the required improvement threshold 94.07%.

### Informational Metrics
- Best epoch appears at epoch 62 with `test_acc=93.42%`; after smoothing was removed, training loss collapsed sharply but validation accuracy plateaued and ended at 92.96%.

## Errors & Dead Ends

## Human Notes

> No human interventions during autopilot execution.
