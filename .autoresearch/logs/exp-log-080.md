# EXP-080: Very Short Linear LR Warmup

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-080.md
- **Plan**: plans/plan-080.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-080
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary

EXP-080 implements the approved very short LR warmup in `train.py` only. It adds `LR_WARMUP_START = 0.02`, `LR_WARMUP_STEPS = 500`, and `current_warmup_lr(step)`, then sets each optimizer param group's LR before the batch update while the helper returns a value. The existing `LR=0.1`, `MultiStepLR` milestones, `optimizer.step()` / `scheduler.step()` order, CutMix anchor, architecture, transforms, batch size, seed, and evaluation harness are unchanged.

### Surprises & Discoveries

No implementation surprises. Because PyTorch `MultiStepLR` preserves the current group LR on non-milestone steps, setting the warmup LR before each early batch is compatible with the existing scheduler as long as `step == 500` explicitly restores `LR=0.1`.

### Decisions

The helper applies through `step == LR_WARMUP_STEPS` rather than stopping at `step < LR_WARMUP_STEPS`; this ensures the optimizer param group is restored to exactly 0.1 before normal scheduler ownership resumes. After step 500, no manual LR assignment occurs, so the 21k milestone should still drop from 0.1 to 0.01.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local foreground session 37567 on GPU0
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-09 23:16 UTC
- **Ended**: 2026-06-09 23:23 UTC

Description:
- This run tests whether reducing the earliest optimizer impulse improves the current probabilistic CutMix anchor. It warms LR from 0.02 to 0.1 over the first 500 steps while preserving the validated CutMix, architecture, optimizer, milestones, transforms, and evaluation harness. The active baseline is 94.11%, so the required improvement threshold is 94.21%.

Observations:
- Preflight scope check passed: `git diff --name-only` listed only `train.py`.
- Preflight syntax check passed: `python3 -m py_compile train.py` exited 0.
- Preflight style check passed: `uv run ruff check train.py` reported `All checks passed!`.
- GPU availability check found both H20 GPUs idle; Run 1 selected GPU0 with `CUDA_VISIBLE_DEVICES=0`.
- Startup confirmed `Device: cuda`, `ResNet-20 | params: 822,790`, unchanged `CutMix alpha: 1.0, prob: 0.5, label smoothing: 0.05`, and `LR warmup: 0.02 -> 0.1 over 500 steps`.
- Early LR warmup behavior is confirmed: progress lines showed `lr: 0.0278` at step 50, `lr: 0.0998` at step 500, and `lr: 0.1000` at step 550. Early evaluations are healthy, reaching 82.04% best at epoch 10.
- Pre-drop best reached 87.96% at epoch 33. The first LR drop was reached at step 21000 in epoch 54, switching to `lr: 0.0100` with about 135s remaining. Post-drop accuracy reached 91.68% at epoch 54 and 93.59% by epoch 59.
- The best post-drop value was `94.08%` at epoch 93. This is below the current baseline of `94.11%` and below the required noise-guard threshold of `94.21%`, so EXP-080 is classified as `no-improvement`.

Key Metrics:
- `best_test_acc`: 94.08%
- `final_test_acc`: 93.37%
- `final_test_loss`: 0.2708
- `training_seconds`: 300.0
- `total_seconds`: 392.7
- `startup_seconds`: 1.9
- `peak_vram_mb`: 660.4
- `num_epochs`: 102
- `num_steps`: 39,466
- `num_params`: 822,790

## Verification Results

### Conditions Checked
- Scope constraint: passed. `git diff --name-only` lists only `train.py`.
- Syntax/style: passed. `python3 -m py_compile train.py` exited 0 before launch; `uv run ruff check train.py` reported all checks passed.
- Implementation/log marker: passed. Startup log confirms `LR warmup: 0.02 -> 0.1 over 500 steps` and unchanged `CutMix alpha: 1.0, prob: 0.5, label smoothing: 0.05`.
- Scheduler behavior: passed. The log shows LR below 0.1 early, near-restored by step 500, exactly `0.1000` after warmup, and `0.0100` at step 21000.
- Completed run and primary metric: passed. `run.log` contains numeric `best_test_acc: 94.08%`.
- Hard constraints: passed. Only `train.py` changed; parameter count remained 822,790; fixed 300s training budget and once-per-epoch validation remained intact.
- Improvement threshold: failed for improvement classification. `best_test_acc=94.08%` is below the 94.21% required threshold, so verdict is `no-improvement`.

### Informational Metrics
- `final_test_acc`: 93.37%
- `final_test_loss`: 0.2708
- `training_seconds`: 300.0
- `total_seconds`: 392.7
- `startup_seconds`: 1.9
- `peak_vram_mb`: 660.4
- `num_epochs`: 102
- `num_steps`: 39,466
- `num_params`: 822,790

## Errors & Dead Ends

## Human Notes

> Autopilot execution; no human intervention during implementation.
