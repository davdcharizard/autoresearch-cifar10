# EXP-049: Decoupled SGD Weight Decay at 2e-4

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-049.md
- **Plan**: plans/plan-049.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-049
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary

Implemented the planned decoupled weight-decay test in `train.py`. The SGD optimizer now uses `weight_decay=0.0`, while a new `apply_decoupled_weight_decay(params, lr)` helper applies multiplicative shrinkage with the existing `WEIGHT_DECAY = 2e-4` value after each optimizer step. All architecture, data augmentation, label smoothing, LR milestones, compile, channels-last, batch size, and fixed-harness settings were preserved.

### Surprises & Discoveries

No implementation surprises. `model.parameters()` remains available after optional `torch.compile(model)`, so `decay_params` can be captured after compile and passed to the same optimizer parameter iterator style used by the anchor.

### Decisions

- Applied decay to every trainable parameter, matching the current global `optim.SGD(..., weight_decay=WEIGHT_DECAY)` parameter set rather than introducing a new BN/bias exclusion policy.
- Captured `step_lr` before `optimizer.step()` and applied manual decay before `scheduler.step()`, so the decay scale uses the LR active for the just-completed SGD update while the printed LR milestone behavior remains unchanged.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local foreground process on GPU1, shell PID 3208157, uv PID 3208158, main python PID 3208161
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log`
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-09 15:47:38 UTC
- **Ended**: 2026-06-09 15:54:33 UTC

Description:
- Run the EXP-049 decoupled SGD weight-decay recipe on a single local GPU with output captured to `run.log`. The experiment preserves the EXP-038 anchor but changes weight decay semantics from optimizer-coupled L2 to manual multiplicative shrinkage. Expected behavior is near-anchor throughput, a first LR drop at step 21000, and final `best_test_acc` compared against the 94.07% threshold.

Observations:
- Preflight passed: `python3 -m py_compile train.py` exited 0 and `uv run ruff check train.py` reported `All checks passed!`.
- Diff scope confirmed: tracked code changes are limited to `train.py`; the diff disables SGD-coupled weight decay and adds manual decoupled shrinkage using the existing `WEIGHT_DECAY = 2e-4`.
- 2026-06-09 15:47 UTC: GPU1 selected because GPU0 had an unrelated active run in another workspace and GPU1 was idle at launch. Stale `run.log` was removed immediately before launch.
- 2026-06-09 15:47 UTC: Foreground training process launched; startup log confirms CUDA, expected parameter count, 300s budget, and 390 batches per epoch.
- 2026-06-09 15:49 UTC: Early training healthy through epoch 19 with best test accuracy 87.71%, no traceback/OOM patterns, and throughput sufficient to reach the step-21000 LR drop.
- 2026-06-09 15:52 UTC: First LR drop confirmed at step 21000 (`lr: 0.0100`). Post-drop accuracy climbed to 92.95% by epoch 78, still below the 94.07% threshold.
- 2026-06-09 15:54 UTC: Run exited cleanly with `best_test_acc=93.06%`, `final_test_acc=92.86%`, and `total_seconds=404.0`. The comparison is valid, but the result is below the 94.07% improvement threshold, so EXP-049 is a no-improvement.

Key Metrics:
- `best_test_acc`: 93.06%
- `final_test_acc`: 92.86%
- `final_test_loss`: 0.2655
- `training_seconds`: 300.0
- `total_seconds`: 404.0
- `startup_seconds`: 3.0
- `peak_vram_mb`: 660.4
- `num_epochs`: 104
- `num_steps`: 40437
- `num_params`: 822,790

## Verification Results

### Conditions Checked
- [x] Baseline check passed: experiment index reports baseline `93.97%` at commit `755be2c`; improvement threshold is `94.07%`.
- [x] Diff scope passed: tracked code changes were limited to `train.py`.
- [x] Compile passed: `python3 -m py_compile train.py` exited 0.
- [x] Style passed: `uv run ruff check train.py` reported `All checks passed!`.
- [x] Completion passed: training exited 0 with numeric `best_test_acc=93.06%` and `total_seconds=404.0`, below the 600s limit.
- [x] First LR drop passed: `run.log` contains a step 21000 progress line with `lr: 0.0100`.
- [x] Metric extraction passed: all expected final summary metrics were present in `run.log`.
- [x] Classification: valid no-improvement because `93.06% < 94.07%`.

### Informational Metrics
- `final_test_acc`: 92.86%
- `final_test_loss`: 0.2655
- `training_seconds`: 300.0
- `total_seconds`: 404.0
- `startup_seconds`: 3.0
- `peak_vram_mb`: 660.4
- `num_epochs`: 104
- `num_steps`: 40437
- `num_params`: 822,790

## Errors & Dead Ends

## Human Notes

> No human interventions during autopilot execution.
