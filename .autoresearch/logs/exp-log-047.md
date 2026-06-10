# EXP-047: Mild ColorJitter After Crop/Flip

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-047.md
- **Plan**: plans/plan-047.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-047
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary

Implemented the planned targeted photometric augmentation in `train.py` by inserting `transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02)` after crop/flip and before tensor conversion. All architecture, optimizer, schedule, smoothing, weight decay, batch size, compile, and channels-last settings were preserved.

### Surprises & Discoveries

No implementation surprises. The transform stack already had the correct insertion point between spatial augmentation and `ToTensor()`.

### Decisions

- Used the planned conservative jitter values without changing any non-augmentation anchors, so the result cleanly attributes to targeted photometric augmentation.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local foreground process on GPU0, shell PID 3028374, uv PID 3028375, python PID 3028378
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log`
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-09 13:25:27 UTC
- **Ended**: 2026-06-09 13:32:26 UTC

Description:
- Run the EXP-047 mild ColorJitter augmentation on a single local GPU with output captured to `run.log`. The experiment preserves the EXP-038 optimizer, schedule, architecture, reflection crop, smoothing, and weight-decay anchor. Expected behavior is the usual `lr: 0.1000` phase, a first drop at step 21000 if enough steps are reached, and final `best_test_acc` compared against the 94.07% threshold.

Observations:
- 2026-06-09 13:25 UTC: Selected GPU0 because it had slightly lower memory use than GPU1 (`16332/97871 MiB` vs `17292/97871 MiB`). Both GPUs showed 100% external utilization and no local process-table entries from `nvidia-smi pmon`; the run proceeds under contention because memory headroom is ample and prior valid runs completed under the same infrastructure pattern.
- Removed stale `run.log` immediately before launch.
- 2026-06-09 13:25 UTC: Foreground training process launched; local process IDs are shell PID 3028374, `uv` PID 3028375, and Python PID 3028378.
- 2026-06-09 13:26 UTC: Startup healthy. `run.log` reports `Device: cuda`, `ResNet-20 | params: 822,790`, `Time budget: 300s`, and first evaluations through epoch 3 with best test accuracy 71.19%. No traceback/OOM/error pattern found.
- 2026-06-09 13:29 UTC: Pre-drop progress lagged under external GPU contention; at about step 11650 with 119 seconds remaining, first LR drop at step 21000 became unlikely.
- 2026-06-09 13:32 UTC: Run completed naturally without crashing, but never reached the first LR drop. Final `num_steps=20321`, all logged LR entries remained `lr: 0.1000`, and final `best_test_acc=88.89%`.

Key Metrics:
- `best_test_acc`: 88.89%
- `final_test_acc`: 86.90%
- `final_test_loss`: 0.4170
- `training_seconds`: 300.0
- `total_seconds`: 380.6
- `startup_seconds`: 3.1
- `peak_vram_mb`: 660.4
- `num_epochs`: 53
- `num_steps`: 20321
- `num_params`: 822,790
- Baseline / threshold: 93.97% baseline, 94.07% improvement threshold.

## Verification Results

### Conditions Checked
- Baseline check: PASS. `exp-index.sh baseline` reports `baseline=93.97`, `baseline_commit=755be2c`; improvement threshold is 94.07%.
- Code scope: PASS. `git diff --name-only` lists only tracked code path `train.py` (plus local `.autoresearch/` metadata not shown by git because it is ignored).
- Compile: PASS. `python3 -m py_compile train.py` exits 0.
- Style: PASS. `uv run ruff check train.py` reports `All checks passed!`.
- Run completion: PASS. `env CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` exited 0 and produced numeric final metrics.
- LR milestone behavior: FAIL/CAVEAT. The run ended at `num_steps=20321`, below the planned first drop at step 21000; no `lr: 0.0100` line appears in `run.log`.
- Improvement threshold: FAIL. `best_test_acc=88.89%`, below both the 93.97% baseline and the 94.07% threshold.

### Informational Metrics
- Final summary from `run.log`: `best_test_acc=88.89%`, `final_test_acc=86.90%`, `final_test_loss=0.4170`, `training_seconds=300.0`, `total_seconds=380.6`, `startup_seconds=3.1`, `peak_vram_mb=660.4`, `num_epochs=53`, `num_steps=20321`, `num_params=822,790`.
- Training ran under visible external contention on both GPUs; this likely explains the missed step-21000 LR drop and makes the run a weak attribution test of ColorJitter itself.

## Errors & Dead Ends

## Human Notes

> No human interventions during autopilot execution.
