# EXP-018: Learned Projection Shortcuts at Downsample Transitions

## Execution

Overall Status & Info:
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-018.md
- **Plan**: plans/plan-018.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-018
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: failed (verification: no-improvement)

## Implementation Notes

### Summary

EXP-018 implements the planned projection-shortcut architecture change in `train.py` only. `BasicBlock` now uses an identity shortcut for same-shape residual blocks and a learned 1x1 convolution plus BatchNorm projection for stride/channel-change transition blocks. The current best EXP-016 width `(28, 56, 112)`, 21k first LR drop, optimizer, augmentation, seed, compile/channels-last path, batch size, and validation cadence were preserved.

### Surprises & Discoveries

The existing block implementation kept only three attributes for zero-padding (`stride`, `need_pad`, and `pad_channels`), so replacing it with `self.proj` was mechanically small. `torch.nn.functional as F` still remains required for ReLU and cross-entropy, so no import cleanup was needed.

### Decisions

The shortcut projection uses `nn.Sequential(Conv2d(..., kernel_size=1, stride=stride, bias=False), BatchNorm2d(...))` only when stride or channel count changes. Same-shape residual blocks remain pure identity shortcuts to keep the experiment targeted to transition mappings and minimize throughput impact.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local session 63229; shell PID 1006822; uv PID 1006823; main Python PID 1006826
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-08 17:03 UTC
- **Ended**: 2026-06-08 17:10 UTC

Description:
- Run the 28/56/112 ResNet-20 recipe with learned projection shortcuts at the two downsample/channel-transition blocks. This tests whether learned stage-transition mappings improve CIFAR-10 `best_test_acc` without the broad throughput loss seen in EXP-017. The run must complete on one GPU, keep validation to once per epoch, and reach at least `93.33%` `best_test_acc` to count as an improvement over the 93.23% baseline.

Observations:
- Preflight passed: `python3 -m py_compile train.py` exited 0, `uv run ruff check train.py` reported all checks passed, `git diff --name-only` showed only `train.py`, and `rg -n "evaluator\\.evaluate|Eval\\(" train.py` showed one `Eval()` construction plus one epoch-level evaluate call.
- Startup confirmed on CUDA with `ResNet-20 | params: 830,966`, 300s training budget, and 390 batches per epoch. The parameter count is a small increase over EXP-016's 822,790, consistent with two learned projection shortcuts. (source: run.log L1-L4)
- The first LR drop was reached at step 21000 during epoch 54; accuracy rose from a pre-drop best of 89.26% to 90.97% immediately after the drop. By epoch 76 the run had reached 92.97%, still below the 93.33% improvement threshold. (source: run.log L98-L158)
- The run completed without crash before the 10-minute cap. Final summary reported `best_test_acc: 92.97%`, which is below the 93.23% baseline and therefore below the 93.33% required improvement threshold. (source: run.log final summary)

Key Metrics:
- best_test_acc: 92.97% (source: run.log final summary)
- final_test_acc: 92.90% (source: run.log final summary)
- final_test_loss: 0.3227 (source: run.log final summary)
- training_seconds: 300.0 (source: run.log final summary)
- total_seconds: 395.8 (source: run.log final summary)
- startup_seconds: 2.9 (source: run.log final summary)
- peak_vram_mb: 670.9 (source: run.log final summary)
- num_epochs: 99 (source: run.log final summary)
- num_steps: 38322 (source: run.log final summary)
- num_params: 830,966 (source: run.log final summary)

## Verification Results

### Conditions Checked
- Baseline and threshold: passed. Experiment-index baseline remained 93.23%, so the EXP-018 success threshold was 93.33%. (source: `exp-index.sh baseline`)
- Scope before launch: passed. `git diff --name-only` showed only `train.py`; `data/` remained untracked. (source: command output during execution)
- Syntax and lint: passed. `python3 -m py_compile train.py` exited 0 and `uv run ruff check train.py` reported all checks passed. (source: command output during execution)
- Validation cadence: passed. `rg -n "evaluator\\.evaluate|Eval\\(" train.py` showed one `Eval()` construction and one epoch-level `evaluator.evaluate(...)` call. (source: command output during execution)
- Experiment completion: passed. `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` exited 0 before 10 minutes total wall-clock and reported numeric metrics. (source: run.log final summary)
- Metric improvement: failed. `best_test_acc` was 92.97%, below baseline 93.23% and below the required 93.33% threshold. Remaining hard-constraint checks were not used to overturn this failed necessary condition. (source: run.log final summary)

### Informational Metrics
- Not collected as passing metrics because the necessary improvement condition failed. Metrics are still recorded in Run 1 Key Metrics for analysis.

## Errors & Dead Ends

## Human Notes

> No human intervention during autopilot execution.
