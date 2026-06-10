# EXP-025: Batch Size 96 with Step-Budget-Aware Milestones

## Execution

Overall Status & Info:
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-025.md
- **Plan**: plans/plan-025.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-025
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed - no-improvement

## Implementation Notes

### Summary

EXP-025 applies the planned batch-size plus schedule recalibration change: `BATCH_SIZE` in `train.py` was reduced from `128` to `96`, and `LR_MILESTONES` was changed from `[21000, 64000]` to `[26000, 44000]`. The current-best 28/56/112 architecture, `1e-4` weight decay, optimizer class, augmentation, FP32 compile/channels-last throughput path, fixed 300s training budget, and once-per-epoch validation were left unchanged.

### Surprises & Discoveries

No implementation surprises. Both edited values are top-level constants consumed by the existing DataLoader and `MultiStepLR` setup.

### Decisions

The milestones are intentionally changed with batch size because smaller batches change epoch length and likely reachable optimizer-step count. The run must confirm `Batches per epoch: 520`, first drop at step 26000, and second drop at step 44000 to validate the schedule calibration.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local session 36133; shell PID 1781061; uv PID 1781062; main Python PID 1781065
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-08 18:52 UTC
- **Ended**: 2026-06-08 18:58 UTC

Description:
- Run the unchanged 28/56/112 ResNet-20 anchor with `BATCH_SIZE=96` and `LR_MILESTONES=[26000, 44000]`. This tests whether smaller-batch stochasticity and a higher optimizer-step budget improve `best_test_acc` under the fixed 300s training budget. The run must stay on one GPU, preserve once-per-epoch validation, report `Batches per epoch: 520`, reach both planned LR drops, and report at least 93.33% `best_test_acc` to count as an improvement.

Observations:
- Baseline query before launch reported `baseline=93.23`, `baseline_commit=f187edf`, `total_experiments=26`, and `improvements=6`; EXP-025 threshold is 93.33%.
- GPU check showed both physical GPUs idle, so the run was launched with `CUDA_VISIBLE_DEVICES=0`.
- CUDA preflight confirmed one visible NVIDIA H20. Startup confirmed `Device: cuda`, `ResNet-20 | params: 822,790`, `Time budget: 300s`, and `Batches per epoch: 520`, validating the `BATCH_SIZE=96` change. (source: run.log startup lines)
- Early training is healthy with epoch evaluations printing and no traceback/OOM/NaN/Inf patterns found. Pre-drop evaluation reached `best_test_acc=84.63%` by epoch 11. (source: run.log early eval lines through epoch 11)
- The first LR drop was reached at `step 26000` with `lr: 0.0100`, validating the planned first milestone. Post-drop accuracy climbed to `best_test_acc=93.11%` by epoch 60, still below the `93.33%` improvement threshold at the latest mid-run check. No second LR drop at step 44000 has appeared yet. (source: run.log lines containing `step 26000` and eval lines through epoch 62)
- The run completed normally with exit code 0 and printed summary metrics. It ended after 32,996 optimizer steps, so the planned second LR drop at step 44,000 was unreachable. Final `best_test_acc=93.11%`, below the 93.23% baseline and the 93.33% improvement threshold, so EXP-025 is no-improvement. (source: run.log summary metrics)

Key Metrics:
- `best_test_acc`: 93.11%
- `final_test_acc`: 93.06%
- `final_test_loss`: 0.2605
- `training_seconds`: 300.0
- `total_seconds`: 364.1
- `startup_seconds`: 1.9
- `peak_vram_mb`: 539.0
- `num_epochs`: 64
- `num_steps`: 32996
- `num_params`: 822,790

## Verification Results

### Conditions Checked
- Baseline and threshold: PASS. Baseline query reported `baseline=93.23`; EXP-025 threshold is `93.33%`. (source: exp-index.sh baseline output)
- Scope before launch/result: PASS. `git diff --name-only` reported only `train.py`; `git status --short --branch` showed branch `autoresearch/exp-025`, modified `train.py`, and untracked `data/`.
- Syntax and lint: PASS. `python3 -m py_compile train.py` exited 0 and `uv run ruff check train.py` reported all checks passed before launch.
- Validation cadence: PASS. `rg -n "evaluator\\.evaluate|Eval\\(" train.py` reported one `Eval()` construction and one `evaluator.evaluate(...)` call.
- Batch-size and schedule: FAIL. `Batches per epoch: 520` confirmed `BATCH_SIZE=96`, and `step 26000` showed `lr: 0.0100`, but the run ended at `num_steps=32996` and never reached the planned `step 44000` second drop.
- Experiment completion: PASS. The process exited 0, printed numeric summary metrics, and `total_seconds=364.1`, under the 10-minute wall-clock cap.
- Metric improvement: FAIL. `best_test_acc=93.11%`, below the required `93.33%` threshold. Under the goal's +0.10 percentage-point rule, this is no-improvement.
- Hard constraints: PASS. Only the planned `BATCH_SIZE = 96` and `LR_MILESTONES = [26000, 44000]` diff was present during the run, `training_seconds=300.0`, `total_seconds=364.1`, and no protected files changed.

### Informational Metrics
- `final_test_acc`: 93.06%
- `final_test_loss`: 0.2605
- `training_seconds`: 300.0
- `total_seconds`: 364.1
- `startup_seconds`: 1.9
- `peak_vram_mb`: 539.0
- `num_epochs`: 64
- `num_steps`: 32996
- `num_params`: 822,790

## Errors & Dead Ends

## Human Notes

> No human intervention during autopilot execution.
