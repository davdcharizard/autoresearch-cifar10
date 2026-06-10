# EXP-056: Strong Weight Decay on Weights Only

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-056.md
- **Plan**: plans/plan-056.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-056
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary

Implemented the planned optimizer parameter-group split in `train.py`. The patch adds `make_weight_decay_param_groups(model)`, assigns BatchNorm direct parameters and all `bias` parameters to a no-decay group, assigns all other trainable parameters to a `WEIGHT_DECAY=2e-4` group, constructs those groups before `torch.compile`, and passes them into SGD without a global optimizer-level `weight_decay`.

### Surprises & Discoveries

No code-structure surprises. Iterating over direct module parameters with `recurse=False` provides a straightforward way to ensure each trainable parameter is assigned exactly once before compile wrapping.

### Decisions

- Constructed parameter groups before `torch.compile(model)` so module types and names remain easy to inspect.
- Used per-group `weight_decay` values and removed the top-level SGD `weight_decay` argument to avoid accidental double decay.
- Added a startup print for decay/no-decay parameter counts so the run log verifies the intended grouping.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local foreground process on GPU0, session 95356, shell PID 3435244, uv PID 3435245, main python PID 3435248
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log`
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-09 17:31:40 UTC
- **Ended**: 2026-06-09 17:39:07 UTC

Description:
- Run the EXP-056 optimizer parameter-group probe on one local GPU with output captured to `run.log`. This tests whether the current `WEIGHT_DECAY=2e-4` anchor improves if decay is applied only to convolution and linear weights while BatchNorm affine and bias parameters receive zero decay. Expected behavior is startup reporting nonzero decay/no-decay group counts, unchanged `Batches per epoch: 390`, unchanged `num_params=822,790`, first LR drop at step 21000 with `lr: 0.0100`, and final `best_test_acc` compared against the 94.07% threshold.

Observations:
- Preflight passed: tracked diff is limited to `train.py`, `python3 -m py_compile train.py` exited 0, and `uv run ruff check train.py` reported `All checks passed!`.
- Baseline check passed before launch: `baseline=93.97`, `baseline_commit=755be2c`, so the improvement threshold is `94.07%`.
- 2026-06-09 17:31 UTC: Both H20 GPUs were free (`0MiB`, `0%` utilization, no running processes). GPU0 selected for EXP-056.
- 2026-06-09 17:32 UTC: Foreground process tree confirmed in the project cwd for shell PID 3435244, uv PID 3435245, and Python PID 3435248.
- 2026-06-09 17:32 UTC: Startup log confirms CUDA, `num_params=822,790`, `Weight decay groups: decay_params=820,372, no_decay_params=2,418`, 300s budget, and `Batches per epoch: 390`. No traceback/OOM/nan/inf patterns are present.
- 2026-06-09 17:32 UTC: Early training is healthy through epoch 16 with best test accuracy 85.81%. Step timing is mostly 6-8ms/batch and the first LR drop is reachable.
- 2026-06-09 17:35 UTC: First LR drop confirmed at `step 21000 ep 54` with `lr: 0.0100`. Post-drop accuracy rose quickly from 91.49% at epoch 54 to 93.68% by epoch 63, then plateaued below the 94.07% improvement threshold through epoch 77.
- 2026-06-09 17:39 UTC: Run completed cleanly with `best_test_acc=93.68%`, `final_test_acc=93.47%`, `num_epochs=107`, and `num_steps=41,416`. The result is a valid no-improvement because it is below the required 94.07% threshold.

Key Metrics:
- `best_test_acc`: 93.68%
- `final_test_acc`: 93.47%
- `final_test_loss`: 0.2441
- `training_seconds`: 300.0
- `total_seconds`: 403.7
- `startup_seconds`: 2.5
- `peak_vram_mb`: 660.4
- `num_epochs`: 107
- `num_steps`: 41,416
- `num_params`: 822,790

## Verification Results

### Conditions Checked
- Baseline: pass. `exp-index.sh baseline` reported `baseline=93.97`, `baseline_commit=755be2c`; improvement threshold is 94.07%.
- Scope: pass. `git diff --name-only` lists only `train.py`.
- Compile: pass. `python3 -m py_compile train.py` exited 0 during preflight.
- Style: pass. `uv run ruff check train.py` reported `All checks passed!` during preflight.
- Run completion: pass. Local foreground process exited 0 and `run.log` reports numeric `best_test_acc`.
- Weight-decay groups: pass. `run.log` reports `Weight decay groups: decay_params=820,372, no_decay_params=2,418`.
- Batch geometry: pass. `run.log` reports `Batches per epoch: 390`.
- LR drop: pass. `run.log` reports `step 21000 ep 54 ... lr: 0.0100`.
- Final metrics: pass. Summary metrics are present and `num_params` remains `822,790`.
- Classification: no-improvement. `best_test_acc=93.68%` is below baseline 93.97% and below the required improvement threshold 94.07%.

### Informational Metrics
- Best epoch appears at epoch 63 with `test_acc=93.68%`; subsequent epochs plateaued or degraded, ending at 93.47%.

## Errors & Dead Ends

## Human Notes

> No human interventions during autopilot execution.
