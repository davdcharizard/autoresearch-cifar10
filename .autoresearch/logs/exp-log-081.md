# EXP-081: Reflection Crop Padding 3

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-081.md
- **Plan**: plans/plan-081.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-081
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary

EXP-081 implements the approved transform-only change in `train.py`: the training `RandomCrop` padding is reduced from 4 to 3 while preserving `padding_mode="reflect"`. A startup marker, `RandomCrop padding: 3 reflect`, was added so `run.log` can prove the intended augmentation setting was used. All CutMix settings, clean label smoothing, model architecture, optimizer, LR schedule, batch size, seed, compile/channels-last behavior, and validation cadence are unchanged.

### Surprises & Discoveries

No implementation surprises. The transform pipeline exposed the padding value directly, so the planned change was a single parameter edit plus the explicit log marker.

### Decisions

The marker is printed immediately after constructing the training transform. This keeps it near the modified configuration and ensures startup verification does not depend on parsing Python source after launch.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local foreground session 26668 on GPU0
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-09 23:30 UTC
- **Ended**: 2026-06-09 23:38 UTC

Description:
- This run tests whether the current CutMix anchor is slightly over-regularized by 4-pixel reflection crop jitter. It reduces only the `RandomCrop` padding from 4 to 3 while preserving reflection padding mode and all model, optimizer, schedule, CutMix, normalization, seed, and evaluation settings. The active baseline is 94.11%, so the required improvement threshold is 94.21%.

Observations:
- Preflight scope check passed: `git diff --name-only` listed only `train.py`.
- Preflight syntax check passed: `python3 -m py_compile train.py` exited 0.
- Preflight style check passed: `uv run ruff check train.py` reported `All checks passed!`.
- GPU availability check found GPU0 idle and GPU1 occupied by another local run; Run 1 selected GPU0 with `CUDA_VISIBLE_DEVICES=0`.
- Startup confirmed `Device: cuda`, `RandomCrop padding: 3 reflect`, `ResNet-20 | params: 822,790`, unchanged `CutMix alpha: 1.0, prob: 0.5, label smoothing: 0.05`, and `Batches per epoch: 390`.
- Early training was healthy: best reached 81.89% at epoch 12, 87.37% at epoch 27, and 89.02% by epoch 42 with no error signatures in `run.log`.
- The first LR drop was reached at step 21000 in epoch 54, switching to `lr: 0.0100` with about 138s remaining. Post-drop accuracy reached 91.56% at epoch 54 and 93.68% by epoch 61.
- The run peaked at 94.18% in epoch 81, then never exceeded it through epoch 103. This is +0.07pp above the 94.11% baseline but below the required 94.21% improvement threshold, so EXP-081 is `no-improvement`.

Key Metrics:
- `best_test_acc`: 94.18%
- `final_test_acc`: 93.31%
- `final_test_loss`: 0.3047
- `training_seconds`: 300.0
- `total_seconds`: 395.3
- `startup_seconds`: 1.8
- `peak_vram_mb`: 660.4
- `num_epochs`: 103
- `num_steps`: 39,856
- `num_params`: 822,790

## Verification Results

### Conditions Checked
- Code-scope constraint: passed. `git diff --name-only` listed only `train.py`.
- Syntax and style: passed. `python3 -m py_compile train.py` exited 0 and `uv run ruff check train.py` reported `All checks passed!`.
- Implementation from code/log: passed. The diff changes `RandomCrop` padding to 3, `run.log` line 2 confirms `RandomCrop padding: 3 reflect`, and line 4 confirms unchanged CutMix alpha/probability/label smoothing.
- Scheduler behavior: passed. `run.log` line 113 shows step 21000 switched to `lr: 0.0100`.
- Run completion and primary metric: passed. `run.log` lines 214-223 report final metrics including numeric `best_test_acc: 94.18%`.
- Hard constraints: passed. Only `train.py` changed; parameter count stayed 822,790; validation remained once per epoch; fixed 300s training budget was used.
- Improvement threshold: failed for improvement classification. The active baseline is 94.11%, and the required threshold is 94.21%; EXP-081 reached 94.18%, so it is below the +0.10pp noise guard and is classified as `no-improvement`.

### Informational Metrics
- `final_test_acc`: 93.31%
- `final_test_loss`: 0.3047
- `training_seconds`: 300.0
- `total_seconds`: 395.3
- `startup_seconds`: 1.8
- `peak_vram_mb`: 660.4
- `num_epochs`: 103
- `num_steps`: 39,856
- `num_params`: 822,790

## Errors & Dead Ends

## Human Notes

> Autopilot execution; no human intervention during implementation.
