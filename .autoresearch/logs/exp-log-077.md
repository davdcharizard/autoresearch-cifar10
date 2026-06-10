# EXP-077: Anti-Aliased Residual Downsample

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-077.md
- **Plan**: plans/plan-077.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-077
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: no-improvement

## Implementation Notes

### Summary

EXP-077 implements the approved anti-aliased residual downsample variant. The branch `autoresearch/exp-077` was created from `autoresearch/dev`, and `train.py` is the only tracked file changed. `BasicBlock` now records whether a block is a residual transition, uses a stride-1 first convolution for transition blocks, and average-pools the residual branch input before that convolution. The option-A shortcut path remains unchanged, and a startup marker was added to make the variant visible in `run.log`.

### Surprises & Discoveries

No implementation surprises. The existing block already stores `self.stride`, so the pooling path can reuse that value without adding another hyperparameter or changing non-transition blocks.

### Decisions

Kept the shortcut path exactly on the anchor implementation to isolate residual-branch downsampling and avoid repeating EXP-059's shortcut average-pooling change. Used `F.avg_pool2d` in `forward` rather than a module attribute so the patch stays minimal and uses the already imported `torch.nn.functional as F`.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local foreground session 51764 on GPU0
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-09 22:35 UTC
- **Ended**: 2026-06-09 22:41 UTC

Description:
- This run tests whether smoothing only the learned residual branch's stride-2 downsampling improves the current CutMix anchor. It preserves the option-A shortcut, CutMix alpha/probability, label smoothing, architecture width/depth, optimizer, schedule, transforms, batch size, compile/channels-last, seed, and evaluation harness. The active baseline is 94.11%, so the required improvement threshold is 94.21%.

Observations:
- Preflight scope check passed: `git diff --name-only` listed only `train.py`.
- Preflight syntax check passed: `python3 -m py_compile train.py` exited 0.
- Preflight style check passed: `uv run ruff check train.py` reported `All checks passed!`.
- GPU availability check found GPU0 idle and GPU1 partially active; Run 1 selected GPU0 with `CUDA_VISIBLE_DEVICES=0`.
- Startup confirmed the expected residual-downsample variant: `Device: cuda`, unchanged `ResNet-20 | params: 822,790`, `Residual downsample: avgpool before stride-2 conv`, and `CutMix alpha: 1.0, prob: 0.5, label smoothing: 0.05` (source: run.log L1-L4).
- Early output reached epoch 9 with no error signatures; `test_acc` improved from 47.46% to a current best of 79.32% (source: run.log L10-L24).
- The first LR drop was reached at step 21000 in epoch 54, switching to `lr: 0.0100` with about 120s remaining. Post-drop accuracy jumped from a pre-drop best of 88.05% at epoch 53 to 91.97% at epoch 54, 93.35% at epoch 56, and 93.91% at epoch 71; this is healthy convergence but still below the 94.21% improvement threshold.
- The final late spike reached 93.99% at epoch 93, then the final checkpoint ended at 93.90%. The run completed cleanly and produced a numeric final metric block.

Key Metrics:
- `best_test_acc`: 93.99%
- `final_test_acc`: 93.90%
- `final_test_loss`: 0.2731
- `training_seconds`: 300.0
- `total_seconds`: 395.6
- `startup_seconds`: 2.0
- `peak_vram_mb`: 577.4
- `num_epochs`: 95
- `num_steps`: 36,813
- `num_params`: 822,790

## Verification Results

### Conditions Checked
- Code scope: `git diff --name-only` listed only `train.py`.
- Syntax: `python3 -m py_compile train.py` exited 0.
- Style: `uv run ruff check train.py` reported `All checks passed!`.
- Startup markers: `run.log` confirmed `Device: cuda`, `ResNet-20 | params: 822,790`, the residual-downsample marker, and unchanged CutMix alpha/probability/label smoothing.
- Schedule: first LR drop at step 21000 observed with `lr: 0.0100`.
- Error scan: no `Traceback`, CUDA, error, NaN, or non-finite signatures were found in `run.log`.
- Improvement test: baseline is 94.11% and the +0.10pp threshold is 94.21%; EXP-077 reached 93.99%, so it is `no-improvement`.

### Informational Metrics
- Best accuracy was 0.12pp below the current 94.11% baseline and 0.22pp below the 94.21% improvement threshold.
- Parameter count remained unchanged at 822,790, confirming this tested only residual downsampling behavior.

## Errors & Dead Ends
- No infrastructure, CUDA, compile, shape, or runtime errors occurred.
- The isolated anti-aliased residual downsample did not clear the active baseline or the required +0.10pp improvement threshold.

## Human Notes

> Autopilot execution; no human intervention during implementation.
