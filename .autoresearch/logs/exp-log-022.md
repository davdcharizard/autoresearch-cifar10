# EXP-022: 20k First LR Drop on 28/56/112

## Execution

Overall Status & Info:
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-022.md
- **Plan**: plans/plan-022.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-022
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: failed - no-improvement; `best_test_acc=93.18%` did not reach the required `93.33%` threshold

## Implementation Notes

### Summary

EXP-022 applies the schedule-only change from the plan: `LR_MILESTONES` in `train.py` was changed from `[21000, 64000]` to `[20000, 64000]`. The current-best 28/56/112 architecture, optimizer, weight decay, batch size, augmentation, FP32 compile/channels-last throughput path, and once-per-epoch validation were left unchanged.

### Surprises & Discoveries

No implementation surprises. The schedule is centralized in a single top-level constant, so the intervention is a one-line isolated change.

### Decisions

The experiment intentionally keeps the second LR milestone at 64000 even though the run is time-budgeted and unlikely to reach it. This preserves the current anchor's schedule shape and isolates only the first-drop bracket.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local session 99195; shell PID 1425323; uv PID 1425325; main Python PID 1425329
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-08 18:10 UTC
- **Ended**: 2026-06-08 18:17 UTC

Description:
- Run the unchanged 28/56/112 ResNet-20 anchor with the first LR milestone moved from 21000 to 20000. This tests whether the local trend toward earlier LR 0.01 refinement continues without confounding capacity, averaging, or regularization changes. The run must stay on one GPU, preserve the fixed 300s training budget and one validation per epoch, reach the first drop at step 20000, and report at least 93.33% `best_test_acc` to count as an improvement.

Observations:
- Baseline query before launch reported `baseline=93.23`, `baseline_commit=f187edf`, `total_experiments=23`, and `improvements=6`; EXP-022 threshold is 93.33%.
- GPU check showed physical GPU 1 idle while GPU 0 had active utilization, so the run was launched with `CUDA_VISIBLE_DEVICES=1`.
- CUDA preflight confirmed one visible NVIDIA H20. Startup confirmed `Device: cuda`, `ResNet-20 | params: 822,790`, `Time budget: 300s`, and `Batches per epoch: 390`. (source: run.log L1-L4)
- Early training is healthy with no traceback/OOM/NaN patterns found. Pre-drop evaluation reached `best_test_acc=84.43%` by epoch 11. (source: run.log early eval lines through epoch 12)
- The planned first LR drop fired at step 20000 with `lr: 0.0100`; post-drop evaluation jumped to 92.23% at epoch 52 and 92.81% by epoch 55. (source: run.log step 20000 and eval lines through epoch 55)
- Later post-drop evaluations plateaued below the target, reaching 93.05% by epoch 63 and remaining under the 93.33% threshold through epoch 71. (source: run.log eval lines through epoch 71)
- Run 1 completed without traceback, OOM, NaN, or Inf patterns. The final best was 93.18%, below the 93.23% baseline and the 93.33% improvement threshold. (source: run.log final summary)

Key Metrics:
- `best_test_acc`: 93.18%
- `final_test_acc`: 92.88%
- `final_test_loss`: 0.3196
- `training_seconds`: 300.0
- `total_seconds`: 392.7
- `startup_seconds`: 2.7
- `peak_vram_mb`: 660.4
- `num_epochs`: 108
- `num_steps`: 41876
- `num_params`: 822,790

## Verification Results

### Conditions Checked
- Baseline and threshold: pass for measurement context; baseline query reported `baseline=93.23`, so the required EXP-022 improvement threshold was `93.33%`.
- Scope: pass; `git diff --name-only` reported only `train.py` among tracked files, with `data/` still untracked.
- Syntax and lint: pass before launch; `python3 -m py_compile train.py` and `uv run ruff check train.py` exited 0.
- Validation cadence: pass; `train.py` retained one `Eval()` construction and one `evaluator.evaluate(...)` call site.
- Schedule activation: pass; `step 20000` showed `lr: 0.0100` in `run.log`.
- Experiment completion: pass; Run 1 exited 0 and reported numeric summary metrics.
- Metric improvement: fail; `best_test_acc=93.18%` is below both the 93.23% baseline and the 93.33% improvement threshold.
- Schedule and hard constraints: pass; only the planned `LR_MILESTONES` change was present during the run, training used the fixed 300.0 second budget, total wall-clock was 392.7 seconds, and no protected files changed.

### Informational Metrics
- `final_test_acc`: 92.88%
- `final_test_loss`: 0.3196
- `training_seconds`: 300.0
- `total_seconds`: 392.7
- `peak_vram_mb`: 660.4
- `num_epochs`: 108
- `num_steps`: 41876
- `num_params`: 822,790

## Errors & Dead Ends

## Human Notes

> No human intervention during autopilot execution.
