# EXP-024: Reachable Second LR Drop on 28/56/112 Anchor

## Execution

Overall Status & Info:
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-024.md
- **Plan**: plans/plan-024.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-024
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed - no-improvement

## Implementation Notes

### Summary

EXP-024 applies the planned schedule-only change: `LR_MILESTONES` in `train.py` was changed from `[21000, 64000]` to `[21000, 36000]`. The current-best 28/56/112 architecture, weight decay, batch size, optimizer, augmentation, FP32 compile/channels-last throughput path, fixed 300s training budget, and once-per-epoch validation were left unchanged.

### Surprises & Discoveries

No implementation surprises. `LR_MILESTONES` is a single top-level constant passed directly into `optim.lr_scheduler.MultiStepLR`.

### Decisions

The experiment intentionally keeps the validated first LR drop at 21000 while moving only the unreachable second milestone. This isolates whether a short late `lr=0.001` refinement phase helps the 28/56/112 anchor.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local session 17349; shell PID 1761755; uv PID 1761756; main Python PID 1761759
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-08 18:39 UTC
- **Ended**: 2026-06-08 18:45 UTC

Description:
- Run the unchanged 28/56/112 ResNet-20 anchor with `LR_MILESTONES` changed to `[21000, 36000]`. This tests whether making the second LR drop reachable provides late `lr=0.001` refinement and improves `best_test_acc`. The run must stay on one GPU, preserve once-per-epoch validation, hit both planned LR drops, and report at least 93.33% `best_test_acc` to count as an improvement.

Observations:
- Baseline query before launch reported `baseline=93.23`, `baseline_commit=f187edf`, `total_experiments=25`, and `improvements=6`; EXP-024 threshold is 93.33%.
- GPU check showed GPU 0 busy and GPU 1 idle, so the run was launched with `CUDA_VISIBLE_DEVICES=1`.
- CUDA preflight confirmed one visible NVIDIA H20. Startup confirmed `Device: cuda`, `ResNet-20 | params: 822,790`, `Time budget: 300s`, and `Batches per epoch: 390`. (source: run.log L1-L4)
- Early training is healthy with epoch evaluations printing and no traceback/OOM/NaN/Inf patterns found. Pre-drop evaluation reached `best_test_acc=85.98%` by epoch 19. (source: run.log early eval lines through epoch 19)
- The first LR drop fired at `step 21000` with `lr: 0.0100`, preserving the validated 21k anchor. Post-drop accuracy rose to `best_test_acc=92.89%` by epoch 60; the planned second drop at step 36000 had not yet occurred. (source: run.log step 21000 line and evals through epoch 61)
- The second LR drop fired at `step 36000` with `lr: 0.0010`, so the experiment is testing the planned reachable late-refinement phase. After the second drop, `best_test_acc` rose to 93.13% by epoch 103, still below the 93.33% threshold. (source: run.log step 36000 line and evals through epoch 103)
- The run completed normally with exit code 0 and printed summary metrics. Final `best_test_acc=93.13%`, below both the 93.23% baseline and the 93.33% improvement threshold, so EXP-024 is no-improvement. (source: run.log summary metrics)

Key Metrics:
- `best_test_acc`: 93.13%
- `final_test_acc`: 92.92%
- `final_test_loss`: 0.2986
- `training_seconds`: 300.0
- `total_seconds`: 390.5
- `startup_seconds`: 1.9
- `peak_vram_mb`: 660.4
- `num_epochs`: 109
- `num_steps`: 42306
- `num_params`: 822,790

## Verification Results

### Conditions Checked
- Baseline and threshold: PASS. Baseline query reported `baseline=93.23`; EXP-024 threshold is `93.33%`. (source: exp-index.sh baseline output)
- Scope before launch/result: PASS. `git diff --name-only` reported only `train.py`; `git status --short --branch` showed branch `autoresearch/exp-024`, modified `train.py`, and untracked `data/`.
- Syntax and lint: PASS. `python3 -m py_compile train.py` exited 0 and `uv run ruff check train.py` reported all checks passed.
- Validation cadence: PASS. `rg -n "evaluator\\.evaluate|Eval\\(" train.py` reported one `Eval()` construction and one `evaluator.evaluate(...)` call.
- Anchor schedule: PASS. `grep "step 21000\\|step 36000" run.log` showed step 21000 with `lr: 0.0100` and step 36000 with `lr: 0.0010`.
- Experiment completion: PASS. The process exited 0, printed numeric summary metrics, and `total_seconds=390.5`, under the 10-minute wall-clock cap.
- Metric improvement: FAIL. `best_test_acc=93.13%`, below the required `93.33%` threshold. Under the goal's +0.10 percentage-point rule, this is no-improvement.
- Hard constraints: PASS. Only the planned `LR_MILESTONES = [21000, 36000]` diff was present during the run, `training_seconds=300.0`, `total_seconds=390.5`, and no protected files changed.

### Informational Metrics
- `final_test_acc`: 92.92%
- `final_test_loss`: 0.2986
- `training_seconds`: 300.0
- `total_seconds`: 390.5
- `startup_seconds`: 1.9
- `peak_vram_mb`: 660.4
- `num_epochs`: 109
- `num_steps`: 42306
- `num_params`: 822,790

## Errors & Dead Ends

## Human Notes

> No human intervention during autopilot execution.
