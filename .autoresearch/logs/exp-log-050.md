# EXP-050: Clean Mild ColorJitter Retry

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-050.md
- **Plan**: plans/plan-050.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-050
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary

Implemented the planned clean ColorJitter retry in `train.py`. The training transform now applies `transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02)` after reflection crop and horizontal flip, before `ToTensor()` and normalization. All model, optimizer, schedule, label smoothing, compile, channels-last, batch-size, and evaluation settings were preserved.

### Surprises & Discoveries

No implementation surprises. The change is an isolated transform insertion and the preflight diff confirmed `train.py` is the only tracked code path modified.

### Decisions

- Retried the exact EXP-047 ColorJitter values rather than changing magnitude, because EXP-047 was confounded by missed step-21000 LR drop and the clean attribution question is whether the same intervention works under normal schedule conditions.
- Kept the existing transform order of crop/flip before photometric perturbation, then tensor conversion and normalization, matching the prior experiment and keeping the comparison direct.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local foreground process on GPU1, shell PID 3243654, uv PID 3243655, main python PID 3243658
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log`
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-09 16:02:27 UTC
- **Ended**: 2026-06-09 16:11:56 UTC

Description:
- Run the EXP-050 clean mild ColorJitter retry on a single local GPU with output captured to `run.log`. The run tests whether the EXP-047 augmentation can clear the 94.07% threshold when it reaches the critical step-21000 first LR drop. Expected behavior is a clean startup, normal FP32 compile/channels-last throughput, a visible `lr: 0.0100` line at or after step 21000, and final summary metrics for classification.

Observations:
- Preflight passed: tracked diff is limited to `train.py`, `python3 -m py_compile train.py` exited 0, and `uv run ruff check train.py` reported `All checks passed!`.
- Baseline check passed before launch: `baseline=93.97`, `baseline_commit=755be2c`, so the improvement threshold is `94.07%`.
- 2026-06-09 16:02 UTC: GPU1 selected because `nvidia-smi` showed GPU1 at 0 MiB and 0% utilization, while GPU0 was busy at 90% utilization with 1489 MiB allocated. Stale `run.log` was removed immediately before launch.
- 2026-06-09 16:02 UTC: Foreground session launched on GPU1; process table shows shell PID 3243654, uv PID 3243655, and main python PID 3243658 in this workspace. GPU1 allocation rose to 383 MiB during startup.
- 2026-06-09 16:03 UTC: Startup healthy through epoch 5 with best test accuracy 78.87%, no traceback/OOM patterns, and step timing around 7-8ms after compile warmup. Throughput is sufficient to reach the step-21000 LR drop.
- 2026-06-09 16:05 UTC: Mid pre-drop progress healthy at step ~10750 / epoch 28, with best test accuracy 86.56% and no error patterns. Run remains on pace to reach step 21000.
- 2026-06-09 16:07 UTC: First LR drop confirmed at step 21000 (`lr: 0.0100`). Post-drop accuracy rose quickly to 93.05% by epoch 57, so EXP-050 is a clean ColorJitter attribution run rather than a missed-milestone repeat of EXP-047.
- 2026-06-09 16:10 UTC: Late-stage best reached 93.49% by epoch 83 and then plateaued below the 94.07% threshold through about epoch 96. Awaiting final summary before classification.
- 2026-06-09 16:11 UTC: Run exited cleanly with `best_test_acc=93.49%`, `final_test_acc=93.03%`, and `total_seconds=525.0`. EXP-050 is a valid clean no-improvement because it reached the LR drop but stayed below the 94.07% threshold.

Key Metrics:
- `best_test_acc`: 93.49%
- `final_test_acc`: 93.03%
- `final_test_loss`: 0.2585
- `training_seconds`: 300.0
- `total_seconds`: 525.0
- `startup_seconds`: 3.0
- `peak_vram_mb`: 660.4
- `num_epochs`: 106
- `num_steps`: 41280
- `num_params`: 822,790

## Verification Results

### Conditions Checked
- [x] Baseline check passed: experiment index reports baseline `93.97%` at commit `755be2c`; improvement threshold is `94.07%`.
- [x] Diff scope passed: tracked code changes are limited to `train.py`.
- [x] Compile passed: `python3 -m py_compile train.py` exited 0.
- [x] Style passed: `uv run ruff check train.py` reported `All checks passed!`.
- [x] Completion passed: training exited 0 with numeric `best_test_acc=93.49%` and `total_seconds=525.0`, below the 600s limit.
- [x] First LR drop passed: `run.log` contains a step 21000 progress line with `lr: 0.0100`.
- [x] Metric extraction passed: all expected final summary metrics were present in `run.log`.
- [x] Classification: valid no-improvement because `93.49% < 94.07%`.

### Informational Metrics
- `final_test_acc`: 93.03%
- `final_test_loss`: 0.2585
- `training_seconds`: 300.0
- `total_seconds`: 525.0
- `startup_seconds`: 3.0
- `peak_vram_mb`: 660.4
- `num_epochs`: 106
- `num_steps`: 41280
- `num_params`: 822,790

## Errors & Dead Ends

## Human Notes

> No human interventions during autopilot execution.
