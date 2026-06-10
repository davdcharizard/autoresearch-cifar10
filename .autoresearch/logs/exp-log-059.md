# EXP-059: Average-Pool Option-A Downsample Shortcut

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-059.md
- **Plan**: plans/plan-059.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-059
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: no-improvement

## Implementation Notes

### Summary

Implemented the planned average-pool option-A shortcut probe in `train.py` only. The change adds a startup-visible `SHORTCUT_DOWNSAMPLE = "avg_pool_option_a"` constant and replaces stride-2 shortcut slicing with `F.avg_pool2d(..., kernel_size=self.stride, stride=self.stride)` inside the existing downsample branch. The zero-channel padding path, model widths/depth, optimizer, schedule, augmentation, label smoothing, compile path, and validation cadence remain unchanged.

### Surprises & Discoveries

No code-structure surprises. The existing shortcut branch was localized and already used `F.pad`, so replacing the spatial downsample operation required no new imports.

### Decisions

Kept the change inside `if self.need_pad:` exactly as planned, so non-transition residual blocks are untouched. Used `kernel_size=self.stride` instead of a hard-coded `2` to keep the code consistent with the block's stride value.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local session 7763; shell PID 3503102; uv PID 3503103; python PID 3503106
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-09 18:16
- **Ended**: 2026-06-09 18:23 UTC

Description:
- Local foreground run of EXP-059 on one selected GPU. This run tests whether replacing strided shortcut slicing with average pooling before option-A zero padding improves downsample transition quality without changing trainable parameter count or the current training recipe. Expected behavior is a clean run within 10 minutes, `Shortcut downsample: avg_pool_option_a` in startup logs, `num_params=822,790`, the step-21000 LR drop, and classification against the `94.07%` improvement threshold.

Observations:
- Startup succeeded on CUDA with `ResNet-20 | params: 822,790`, `Shortcut downsample: avg_pool_option_a`, and unchanged `Batches per epoch: 390`. This confirms no trainable parameter-count change and expected batch geometry. (source: run.log L1-L5)
- The first LR drop was reached cleanly at step 21000 in epoch 54, with the progress line reporting `lr: 0.0100`; the comparison is not confounded by a missed schedule milestone.
- Post-drop accuracy rose from 91.80% at epoch 54 to 93.42% by epoch 66, then plateaued around 93.0-93.4% through the end of the 300s training budget. The final epoch reported 93.20% with best 93.42%.
- The run exited normally with final summary metrics present, total wall time 403.3s, and peak VRAM 660.4 MB.

Key Metrics:
- `best_test_acc`: 93.42%
- `final_test_acc`: 93.20%
- `final_test_loss`: 0.2566
- `training_seconds`: 300.0
- `total_seconds`: 403.3
- `startup_seconds`: 2.4
- `peak_vram_mb`: 660.4
- `num_epochs`: 102
- `num_steps`: 39,682
- `num_params`: 822,790
- Baseline: 93.97%
- Improvement threshold: 94.07%
- Delta vs baseline: -0.55 percentage points
- Verdict: no-improvement

## Verification Results

### Conditions Checked
- `python3 -m py_compile train.py`: passed before launch.
- `uv run ruff check train.py`: passed before launch.
- Tracked code diff: `train.py` only.
- Startup shortcut setting: passed, `Shortcut downsample: avg_pool_option_a`.
- Batch geometry: passed, `Batches per epoch: 390`.
- First LR drop: passed, step 21000 reported `lr: 0.0100`.
- Final summary metrics: passed, numeric `best_test_acc` present.
- Parameter count: passed, `num_params=822,790`.
- Runtime cap: passed, total wall time 403.3s < 600s.
- Improvement classification: no-improvement because 93.42% < 94.07%.

### Informational Metrics
- Final epoch: 102
- Final test accuracy/loss: 93.20% / 0.2566
- Best epoch region: first reached 93.42% at epoch 66 and matched at epoch 72; no later improvement above 93.42%.

## Errors & Dead Ends
- None. The run completed cleanly and produced valid final metrics.

## Human Notes

> Autopilot mode; no human approval or intervention requested during execution.
