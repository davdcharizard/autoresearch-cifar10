# EXP-038: Increase Weight Decay to 2e-4

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-038.md
- **Plan**: plans/plan-038.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-038
- **Commit**: 755be2c
- **PR**: skipped - no git remote configured
- **Outcome**: completed - improvement

## Implementation Notes

### Summary

Implemented EXP-038 exactly as planned by changing only `WEIGHT_DECAY` in `train.py` from `1e-4` to `2e-4`. The reflection-padding 28/56/112 ResNet-20 anchor, batch size 128, `LR_MILESTONES = [21000, 64000]`, optimizer class, momentum, `label_smoothing=0.05`, FP32 channels-last compile path, seed, fixed training budget, and once-per-epoch validation path were preserved. Preflight checks passed for Python syntax, ruff, diff scope, stronger weight decay, anchor settings, and validation cadence.

### Surprises & Discoveries

No implementation surprises. Weight decay is controlled by a single top-level constant used directly in the SGD optimizer construction, so the experiment is a one-scalar change.

### Decisions

No deviations from the plan were needed. The schedule, smoothing, and batch geometry are intentionally preserved to isolate stronger weight decay.

## Experimental Adjustments
- No PR was created because this repository has no configured git remote. The successful code change was committed on `autoresearch/exp-038` and fast-forward merged into `autoresearch/dev`.

## Run Log

### Run 1

Metadata:
- **Job ID**: local command session 93017; shell PID 2438355
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-09 07:49 UTC
- **Ended**: 2026-06-09 07:56 UTC

Description:
- Run the reflection-padding 28/56/112 ResNet-20 anchor with only `WEIGHT_DECAY` changed from `1e-4` to `2e-4`. The experiment tests whether stronger shrinkage improves generalization without changing throughput, schedule reachability, architecture, augmentation, loss smoothing, optimizer class, or evaluation cadence. It must complete under the fixed 300s training budget and under 10 minutes total wall-clock time. It must reach `best_test_acc >= 93.80%` to count as an improvement under the goal's +0.10 percentage-point rule.

Observations:
- Preflight checks passed: `python3 -m py_compile train.py`, `uv run ruff check train.py`, diff scope, `WEIGHT_DECAY = 2e-4`, preserved anchor grep, and validation-cadence grep all succeeded before launch. (source: command outputs, 2026-06-09 07:47 UTC)
- Baseline check reported `baseline=93.70`, making the concrete improvement threshold `best_test_acc >= 93.80`; both visible H20 GPUs were idle, GPU 0 was selected, and CUDA isolation reported one visible `NVIDIA H20`. (source: exp-index, nvidia-smi, CUDA smoke test, 2026-06-09 07:47 UTC)
- Startup confirms CUDA execution, `ResNet-20 | params: 822,790`, `Time budget: 300s`, and `Batches per epoch: 390`, matching the preserved batch-128 anchor. (source: run.log L1-L4)
- The planned first LR drop fired at step 21000 with `lr: 0.0100`; post-drop accuracy reached 93.67% by epoch 60, still below the 93.80% improvement threshold. (source: `grep "step 21000" run.log`; `grep "eval ep" run.log | tail -15`)
- The late post-drop plateau crossed the improvement bar at epoch 69 and peaked at 93.97% by epoch 74; no `step 64000` line is present. (source: `grep "eval ep" run.log | tail -20`; `grep "step 64000" run.log`)
- The run exited cleanly with `best_test_acc: 93.97%`, `final_test_acc: 93.54%`, `final_test_loss: 0.2447`, and `total_seconds: 403.1`. This is +0.27 over the 93.70% baseline and clears the 93.80% improvement threshold. (source: final summary in run.log)

Key Metrics:
- `best_test_acc`: 93.97%
- `final_test_acc`: 93.54%
- `final_test_loss`: 0.2447
- `training_seconds`: 300.0
- `total_seconds`: 403.1
- `startup_seconds`: 3.7
- `peak_vram_mb`: 660.4
- `num_epochs`: 107
- `num_steps`: 41389
- `num_params`: 822,790

## Verification Results

### Conditions Checked
- Passed: Process exited with code 0 and completed before the 10-minute wall-clock limit (`total_seconds: 403.1`).
- Passed: The run reported a numeric `best_test_acc` of 93.97%.
- Passed: The current baseline is 93.70%, so the goal requires at least 93.80% for improvement; EXP-038 reached 93.97%.
- Passed: Fixed training budget was preserved (`training_seconds: 300.0`).
- Passed: Architecture stayed unchanged (`num_params: 822,790`).
- Passed: Batch size stayed unchanged (`Batches per epoch: 390`).
- Passed: First LR drop occurred at step 21000 with `lr: 0.0100`.
- Passed: No second LR drop occurred; `step 64000` was absent and the run ended at 41,389 steps.
- Passed: The tracked source diff during the run was limited to the planned `train.py` weight-decay scalar change.
- Passed: No error, exception, CUDA OOM, NaN, or Inf signatures were found in `run.log`.

### Informational Metrics
- Stronger weight decay reached 93.97%, a +0.27 point lift over baseline and a +0.17 point margin over the +0.10 improvement threshold.
- Final accuracy was 0.43 points below the peak, indicating the best value came from the late post-drop plateau rather than the final epoch.

## Errors & Dead Ends

## Human Notes

> No human notes yet.
