# EXP-068: CutMix Alpha 2.0

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-068.md
- **Plan**: plans/plan-068.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-068
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary

Created `autoresearch/exp-068` from `autoresearch/dev` and implemented the planned opposite alpha bracket by changing only `CUTMIX_ALPHA` in `train.py` from `1.0` to `2.0`. The validated `CUTMIX_PROB=0.5` setting, endpoint label smoothing, architecture, optimizer, LR schedule, transform stack, compile/channels-last path, and evaluation cadence were left unchanged.

### Surprises & Discoveries

No implementation surprises. The current CutMix implementation exposes alpha as a top-level constant consumed by the beta distribution, so the planned change is isolated.

### Decisions

Kept the experiment to a one-line hyperparameter change to close the alpha bracket after EXP-067 showed `alpha=0.5` did not beat the anchor. Preflight confirmed the tracked diff is limited to `train.py`; `python3 -m py_compile train.py` and `uv run ruff check train.py` both passed.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local session 8608; shell PID 3712751, uv PID 3712752, main Python PID 3712755
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-09 20:28 UTC
- **Ended**: 2026-06-09 20:35 UTC

Description:
- Run the one-scalar `CUTMIX_ALPHA=2.0` bracket locally on a single available GPU using the fixed CIFAR-10 training harness. This tests whether a less-variable CutMix patch-area distribution improves the current `alpha=1.0, p=0.5` CutMix anchor. The required improvement threshold is `best_test_acc >= 94.21%`; output is captured to `run.log` for startup markers, LR milestone checks, final metrics, and error diagnosis.

Observations:
- Startup markers confirmed CUDA execution, ResNet-20 with 822,790 parameters, `CutMix alpha: 2.0, prob: 0.5, label smoothing: 0.05`, 300s budget, and 390 batches per epoch. GPU0 was selected; `/proc/3712755/cwd` confirmed the main Python process is in this project root. (source: run.log L1-L5; `/proc/3712755/cwd`)
- First LR drop reached as expected at `step 21000 ep 54` with `lr: 0.0100`; no traceback, OOM, or non-finite-loss markers were present during monitoring. Post-drop accuracy climbed from 91.77% at epoch 54 to 93.26% by epoch 57. (source: run.log around `step 21000`; eval lines ep 54-57)
- The run completed cleanly within the 10-minute cap. Peak accuracy reached 94.00% at epoch 79, then final accuracy settled at 93.66% by epoch 102. This is below both the 94.11% current baseline and the 94.21% improvement threshold. (source: run.log L163, L209-L220)

Key Metrics:
- `best_test_acc`: 94.00%
- `final_test_acc`: 93.66%
- `final_test_loss`: 0.2638
- `training_seconds`: 300.0
- `total_seconds`: 395.2
- `startup_seconds`: 2.4
- `peak_vram_mb`: 660.4
- `num_epochs`: 102
- `num_steps`: 39747
- `num_params`: 822,790

## Verification Results

### Conditions Checked
- Branch and tracked diff: passed. Branch is `autoresearch/exp-068`; tracked diff is limited to `train.py`, with the intended `CUTMIX_ALPHA = 2.0` change. (source: `git status --short --branch`; `git diff --name-only`)
- Syntax and lint: passed. `python3 -m py_compile train.py` exited 0 and `uv run ruff check train.py` reported all checks passed before launch.
- Experiment completion: passed. Foreground local run exited cleanly and produced final summary metrics in `run.log` within the 10-minute total cap. (source: run.log L211-L220)
- Startup and scope markers: passed. `run.log` showed CUDA, ResNet-20 with 822,790 parameters, `CutMix alpha: 2.0, prob: 0.5, label smoothing: 0.05`, 300s budget, and 390 batches per epoch. (source: run.log L1-L5)
- LR milestone: passed. The first LR drop appeared at `step 21000 ep 54` with `lr: 0.0100`. (source: run.log around `step 21000`)
- Primary metric extraction: passed. `best_test_acc: 94.00%` was present. (source: run.log L211)
- Improvement threshold: failed for improvement, valid for no-improvement. Current baseline is 94.11%, required threshold is 94.21%, and EXP-068 reached 94.00%.
- Hard constraints: passed. The only tracked code file modified during the experiment was `train.py`; no benchmark harness or dependency files were touched.

### Informational Metrics
- See Key Metrics above.

## Errors & Dead Ends

## Human Notes

> Autopilot execution; no human intervention during implementation.
