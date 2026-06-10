# EXP-082: Horizontal Flip Probability 0.4

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-082.md
- **Plan**: plans/plan-082.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-082
- **Commit**: e859ac5
- **PR**: N/A — no git remote configured in this local repo
- **Outcome**: completed

## Implementation Notes

### Summary

EXP-082 implements the approved transform-only change in `train.py`: the training `RandomHorizontalFlip` probability is changed from the torchvision default 0.5 to explicit `p=0.4`. A startup marker, `RandomHorizontalFlip p: 0.4`, was added so `run.log` can prove the intended augmentation setting was used. Reflection crop padding 4, CutMix settings, clean label smoothing, model architecture, optimizer, LR schedule, batch size, seed, compile/channels-last behavior, and validation cadence are unchanged.

### Surprises & Discoveries

No implementation surprises. The transform pipeline exposed the horizontal flip transform directly, so the planned change was a one-argument edit plus the explicit log marker.

### Decisions

The marker is printed immediately after constructing the training transform. This keeps the verification marker near the modified configuration and avoids relying only on source diff inspection after launch.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local foreground session 3977 on GPU0
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log
- **WandB**: N/A
- **Status**: running
- **Started**: 2026-06-09 23:45 UTC
- **Ended**: 2026-06-09 23:51 UTC

Description:
- This run tests whether the current CutMix anchor is slightly over-regularized by the default horizontal flip probability. It reduces only `RandomHorizontalFlip` from the default 0.5 probability to 0.4 while preserving reflection crop padding, CutMix, clean label smoothing, model, optimizer, schedule, batch size, seed, and evaluation settings. The active baseline is 94.11%, so the required improvement threshold is 94.21%.

Observations:
- Preflight scope check passed: `git diff --name-only` listed only `train.py`.
- Preflight syntax check passed: `python3 -m py_compile train.py` exited 0.
- Preflight style check passed: `uv run ruff check train.py` reported `All checks passed!`.
- GPU availability check found both GPU0 and GPU1 idle; Run 1 selected GPU0 with `CUDA_VISIBLE_DEVICES=0`.
- Startup confirmed `Device: cuda`, `RandomHorizontalFlip p: 0.4`, `ResNet-20 | params: 822,790`, unchanged `CutMix alpha: 1.0, prob: 0.5, label smoothing: 0.05`, and `Batches per epoch: 390` in `run.log` lines 1-6.
- Early training was healthy: best reached 84.89% at epoch 16, 87.23% at epoch 29, and 88.97% by epoch 41 with no error signatures in `run.log`.
- The first LR drop was reached at step 21000 in epoch 54, switching to `lr: 0.0100` with about 135s remaining. Post-drop accuracy reached 91.68% at epoch 54 and 92.91% at epoch 55.
- The run crossed the improvement threshold at epoch 75 with 94.22%, improved again to 94.26% at epoch 81, and peaked at 94.36% at epoch 88. This clears both the 94.11% baseline and the 94.21% noise-guard threshold.

Key Metrics:
- `best_test_acc`: 94.36%
- `final_test_acc`: 93.86%
- `final_test_loss`: 0.2479
- `training_seconds`: 300.0
- `total_seconds`: 393.3
- `startup_seconds`: 1.9
- `peak_vram_mb`: 660.4
- `num_epochs`: 101
- `num_steps`: 39,034
- `num_params`: 822,790

## Verification Results

### Conditions Checked
- Code-scope constraint: passed. `git diff --name-only` listed only `train.py`.
- Syntax and style: passed. `python3 -m py_compile train.py` exited 0 and `uv run ruff check train.py` reported `All checks passed!`.
- Implementation from code/log: passed. The diff changes `RandomHorizontalFlip` to `p=0.4`, `run.log` line 2 confirms `RandomHorizontalFlip p: 0.4`, and line 4 confirms unchanged CutMix alpha/probability/label smoothing.
- Scheduler behavior: passed. `run.log` line 113 shows step 21000 switched to `lr: 0.0100`.
- Run completion and primary metric: passed. `run.log` lines 210-219 report final metrics including numeric `best_test_acc: 94.36%`.
- Hard constraints: passed. Only `train.py` changed; parameter count stayed 822,790; validation remained once per epoch; fixed 300s training budget was used.
- Improvement threshold: passed. The active baseline is 94.11%, and the required threshold is 94.21%; EXP-082 reached 94.36%, so it is an `improvement`.

### Informational Metrics
- `final_test_acc`: 93.86%
- `final_test_loss`: 0.2479
- `training_seconds`: 300.0
- `total_seconds`: 393.3
- `startup_seconds`: 1.9
- `peak_vram_mb`: 660.4
- `num_epochs`: 101
- `num_steps`: 39,034
- `num_params`: 822,790

## Errors & Dead Ends

## Human Notes

> Autopilot execution; no human intervention during implementation.
