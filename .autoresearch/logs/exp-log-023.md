# EXP-023: Lower Weight Decay on 28/56/112 Anchor

## Execution

Overall Status & Info:
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-023.md
- **Plan**: plans/plan-023.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-023
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed - no-improvement

## Implementation Notes

### Summary

EXP-023 applies the planned regularization-only change: `WEIGHT_DECAY` in `train.py` was reduced from `1e-4` to `5e-5`. The current-best 28/56/112 architecture, 21k first LR drop, batch size, optimizer class, augmentation, FP32 compile/channels-last throughput path, and once-per-epoch validation were left unchanged.

### Surprises & Discoveries

No implementation surprises. Weight decay is a single top-level constant passed directly into `optim.SGD`.

### Decisions

The experiment intentionally keeps all other anchor settings fixed to isolate whether lower L2 regularization helps the widened 28/56/112 model. No coupled schedule or architecture changes were made.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local session 71887; shell PID 1596777; uv PID 1596778; main Python PID 1596781
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-08 18:24 UTC
- **Ended**: 2026-06-08 18:31 UTC

Description:
- Run the unchanged 28/56/112 ResNet-20 anchor with `WEIGHT_DECAY` reduced from `1e-4` to `5e-5`. This tests whether the current anchor is slightly over-regularized under the fixed 300s budget. The run must stay on one GPU, preserve the 21k first LR drop and one validation per epoch, and report at least 93.33% `best_test_acc` to count as an improvement.

Observations:
- Baseline query before launch reported `baseline=93.23`, `baseline_commit=f187edf`, `total_experiments=24`, and `improvements=6`; EXP-023 threshold is 93.33%.
- GPU check showed both physical GPUs idle, so the run was launched with `CUDA_VISIBLE_DEVICES=0`.
- CUDA preflight confirmed one visible NVIDIA H20. Startup confirmed `Device: cuda`, `ResNet-20 | params: 822,790`, `Time budget: 300s`, and `Batches per epoch: 390`. (source: run.log L1-L4)
- Early training is healthy with no traceback/OOM/NaN patterns found. Pre-drop evaluation reached `best_test_acc=84.23%` by epoch 12. (source: run.log early eval lines through epoch 13)
- The first LR drop fired at `step 21000` with `lr: 0.0100`, preserving the planned 21k anchor. The epoch 54 evaluation immediately after the drop reached `test_acc=91.17%` and `best_test_acc=91.17%`; no traceback/OOM/NaN/Inf patterns were present. (source: run.log step 21000 line and eval ep 54)
- The run completed normally with exit code 0 and printed summary metrics. Final `best_test_acc=92.83%`, below both the 93.23% baseline and the 93.33% improvement threshold, so EXP-023 is no-improvement. (source: run.log summary metrics)

Key Metrics:
- `best_test_acc`: 92.83%
- `final_test_acc`: 92.46%
- `final_test_loss`: 0.3541
- `training_seconds`: 300.0
- `total_seconds`: 390.6
- `startup_seconds`: 2.1
- `peak_vram_mb`: 660.4
- `num_epochs`: 110
- `num_steps`: 42754
- `num_params`: 822,790

## Verification Results

### Conditions Checked
- Baseline and threshold: PASS. Baseline query reported `baseline=93.23`; EXP-023 threshold is `93.33%`. (source: exp-index.sh baseline output)
- Scope before launch/result: PASS. `git diff --name-only` reported only `train.py`; `git status --short --branch` showed branch `autoresearch/exp-023`, modified `train.py`, and untracked `data/`.
- Syntax and lint: PASS. `python3 -m py_compile train.py` exited 0 and `uv run ruff check train.py` reported all checks passed.
- Validation cadence: PASS. `rg -n "evaluator\\.evaluate|Eval\\(" train.py` reported one `Eval()` construction and one `evaluator.evaluate(...)` call.
- Anchor schedule: PASS. `grep "step 21000" run.log` showed the step 21000 line with `lr: 0.0100`.
- Experiment completion: PASS. The process exited 0, printed numeric summary metrics, and `total_seconds=390.6`, under the 10-minute wall-clock cap.
- Metric improvement: FAIL. `best_test_acc=92.83%`, below the required `93.33%` threshold. Under the goal's +0.10 percentage-point rule, this is no-improvement.
- Hard constraints: PASS. Only the planned `WEIGHT_DECAY = 5e-5` diff was present during the run, `training_seconds=300.0`, `total_seconds=390.6`, and no protected files changed.

### Informational Metrics
- `final_test_acc`: 92.46%
- `final_test_loss`: 0.3541
- `training_seconds`: 300.0
- `total_seconds`: 390.6
- `startup_seconds`: 2.1
- `peak_vram_mb`: 660.4
- `num_epochs`: 110
- `num_steps`: 42754
- `num_params`: 822,790

## Errors & Dead Ends

## Human Notes

> No human intervention during autopilot execution.
