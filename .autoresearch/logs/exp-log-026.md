# EXP-026: Momentum 0.95 on Current Anchor

## Execution

Overall Status & Info:
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-026.md
- **Plan**: plans/plan-026.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-026
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed - no-improvement

## Implementation Notes

### Summary

EXP-026 applies the planned isolated optimizer-coefficient change: `MOMENTUM` in `train.py` was increased from `0.9` to `0.95`. The current-best 28/56/112 architecture, batch size, learning rate, weight decay, LR milestones, FP32 compile/channels-last throughput path, fixed 300s training budget, augmentation, seed, optimizer class, and once-per-epoch validation were left unchanged.

### Surprises & Discoveries

No implementation surprises. `MOMENTUM` is a single top-level constant passed directly into `optim.SGD`.

### Decisions

The experiment intentionally changes only the momentum coefficient, not Nesterov, LR, weight decay, or schedule. This isolates whether stronger classical SGD velocity improves post-drop refinement without the throughput or capacity penalties seen in recent failures.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local session 12766; shell PID 1798406; uv PID 1798407; main Python PID 1798410
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-08 19:05 UTC
- **Ended**: 2026-06-08 19:12 UTC

Description:
- Run the unchanged 28/56/112 ResNet-20 anchor with `MOMENTUM=0.95` instead of `0.9`. This tests whether stronger classical SGD velocity improves post-drop refinement without reducing throughput, changing capacity, or adding explicit regularization. The run must stay on one GPU, preserve once-per-epoch validation, report `Batches per epoch: 390`, hit the step-21000 LR drop, and report at least 93.33% `best_test_acc` to count as an improvement.

Observations:
- Baseline query before launch reported `baseline=93.23`, `baseline_commit=f187edf`, `total_experiments=27`, and `improvements=6`; EXP-026 threshold is 93.33%.
- GPU check showed both physical GPUs idle, so the run was launched with `CUDA_VISIBLE_DEVICES=0`.
- CUDA preflight confirmed one visible NVIDIA H20. Startup confirmed `Device: cuda`, `ResNet-20 | params: 822,790`, `Time budget: 300s`, and `Batches per epoch: 390`, validating that batch size, architecture, and fixed budget were preserved. (source: run.log startup lines)
- Early training is healthy with epoch evaluations printing and no traceback/OOM/NaN/Inf patterns found. Pre-drop evaluation reached `best_test_acc=82.58%` by epoch 15. (source: run.log early eval lines through epoch 15)
- The first LR drop was reached at `step 21000` with `lr: 0.0100`, preserving the current anchor schedule. Post-drop accuracy climbed to `best_test_acc=92.63%` by epoch 57, still below the `93.33%` improvement threshold at the latest mid-run check. (source: run.log lines containing `step 21000` and eval lines through epoch 57)
- The run completed normally with exit code 0 and printed summary metrics. Final `best_test_acc=92.90%`, below the 93.23% baseline and the 93.33% improvement threshold, so EXP-026 is no-improvement. (source: run.log summary metrics)

Key Metrics:
- `best_test_acc`: 92.90%
- `final_test_acc`: 92.28%
- `final_test_loss`: 0.3322
- `training_seconds`: 300.0
- `total_seconds`: 390.3
- `startup_seconds`: 1.8
- `peak_vram_mb`: 660.4
- `num_epochs`: 109
- `num_steps`: 42439
- `num_params`: 822,790

## Verification Results

### Conditions Checked
- Baseline and threshold: PASS. Baseline query reported `baseline=93.23`; EXP-026 threshold is `93.33%`. (source: exp-index.sh baseline output)
- Scope before launch/result: PASS. `git diff --name-only` reported only `train.py`; `git status --short --branch` showed branch `autoresearch/exp-026`, modified `train.py`, and untracked `data/`.
- Syntax and lint: PASS. `python3 -m py_compile train.py` exited 0 and `uv run ruff check train.py` reported all checks passed before launch.
- Validation cadence: PASS. `rg -n "evaluator\\.evaluate|Eval\\(" train.py` reported one `Eval()` construction and one `evaluator.evaluate(...)` call.
- Preserved batch size and schedule: PASS. `Batches per epoch: 390` confirmed `BATCH_SIZE=128`, and `step 21000` showed `lr: 0.0100`.
- Experiment completion: PASS. The process exited 0, printed numeric summary metrics, and `total_seconds=390.3`, under the 10-minute wall-clock cap.
- Metric improvement: FAIL. `best_test_acc=92.90%`, below the required `93.33%` threshold. Under the goal's +0.10 percentage-point rule, this is no-improvement.
- Hard constraints: PASS. Only the planned `MOMENTUM = 0.95` diff was present during the run, `training_seconds=300.0`, `total_seconds=390.3`, and no protected files changed.

### Informational Metrics
- `final_test_acc`: 92.28%
- `final_test_loss`: 0.3322
- `training_seconds`: 300.0
- `total_seconds`: 390.3
- `startup_seconds`: 1.8
- `peak_vram_mb`: 660.4
- `num_epochs`: 109
- `num_steps`: 42439
- `num_params`: 822,790

## Errors & Dead Ends

## Human Notes

> No human intervention during autopilot execution.
