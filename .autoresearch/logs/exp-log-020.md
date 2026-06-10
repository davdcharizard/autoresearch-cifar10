# EXP-020: Final-Stage Width 128 with 20k First LR Drop

## Execution

Overall Status & Info:
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-020.md
- **Plan**: plans/plan-020.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-020
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: failed

## Implementation Notes

### Summary

EXP-020 implements the planned final-stage-only capacity test in `train.py` only. `STAGE_WIDTHS` changed from `(28, 56, 112)` to `(28, 56, 128)`, and `LR_MILESTONES` changed from `[21000, 64000]` to `[20000, 64000]`. The rest of the training recipe, including depth, optimizer, augmentation, seed, batch size, compile/channels-last settings, fixed time budget, and once-per-epoch validation, was preserved.

### Surprises & Discoveries

The implementation was a two-constant diff. Preflight confirmed no incidental changes to evaluation cadence, imports, or training flow.

### Decisions

No deviations from the plan were needed. The first LR drop remains at 20k to compensate for expected final-stage throughput loss while preserving a substantial LR 0.01 refinement window.

## Experimental Adjustments

- **Use physical GPU 1 for launch**: Pre-launch `nvidia-smi` showed GPU 0 active and GPU 1 idle, so the run uses `CUDA_VISIBLE_DEVICES=1` to preserve single-GPU isolation without sharing GPU 0. (ref: pre-launch GPU check, 2026-06-08 17:34 UTC)

## Run Log

### Run 1

Metadata:
- **Job ID**: local session 88842; shell PID 1069111; uv PID 1069112; main Python PID 1069125
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-08 17:34 UTC
- **Ended**: 2026-06-08 17:41 UTC

Description:
- Run a 28/56/128 ResNet-20 with a 20k first LR drop under the fixed 300s training budget. This tests whether adding capacity only in the final 8x8 stage can improve over the 28/56/112 baseline without repeating the broad proportional-widening failures. The run must complete on one GPU, keep validation once per epoch, reach the 20k drop, and achieve at least 93.33% `best_test_acc` to count as an improvement.

Observations:
- Preflight passed: `python3 -m py_compile train.py` exited 0, `uv run ruff check train.py` reported all checks passed, `git diff -- train.py` showed only the `STAGE_WIDTHS` and `LR_MILESTONES` constant changes, and `rg -n "evaluator\\.evaluate|Eval\\(" train.py` showed one `Eval()` construction plus one epoch-level evaluate call.
- Startup confirmed on CUDA with `ResNet-20 | params: 1,004,006`, 300s training budget, and 390 batches per epoch. (source: run.log L1-L4)
- Early training is healthy with no traceback/OOM/NaN patterns found. The run reached epoch 13 with `best_test_acc=82.35%`, and GPU 1 showed active memory allocation for this run. (source: run.log L7-L32)
- The first LR drop was reached at step 20000 during epoch 52 with about 143s training budget remaining. Immediate post-drop evaluations reached 91.77% at epoch 52 and 91.95% at epoch 53, below the 93.33% threshold but with the refinement window still in progress. (source: run.log L109-L112)
- The run completed without traceback/OOM/NaN patterns. Accuracy plateaued after the first LR drop and peaked at `best_test_acc=92.60%` by epoch 76, then ended at `final_test_acc=92.35%`; this is below the required EXP-020 threshold of 93.33%. (source: run.log summary)

Key Metrics:
- best_test_acc: 92.60%
- final_test_acc: 92.35%
- final_test_loss: 0.3503
- training_seconds: 300.0
- total_seconds: 399.6
- startup_seconds: 3.0
- peak_vram_mb: 686.7
- num_epochs: 106
- num_steps: 40989
- num_params: 1,004,006

## Verification Results

### Conditions Checked
- Baseline and threshold: passed. Experiment-index baseline remains `93.23%`; EXP-020 requires `best_test_acc >= 93.33%` under the goal's +0.10 percentage-point rule.
- Scope: passed. `git diff --name-only` reports only `train.py`; `.autoresearch/` remains local-only and `data/` remains untracked.
- Syntax and lint: passed. `python3 -m py_compile train.py` exited 0, and `uv run ruff check train.py` reported all checks passed.
- Validation cadence: passed. `rg -n "evaluator\\.evaluate|Eval\\(" train.py` reports one `Eval()` construction and one epoch-level `evaluator.evaluate(...)` call.
- Experiment completion: passed. The process exited 0, reported numeric metrics, used `training_seconds=300.0`, and stayed under 10 minutes total wall-clock with `total_seconds=399.6`.
- Schedule and hard constraints: passed. `grep "lr: 0.0100" run.log` confirms the first LR drop at step 20000, and only the planned `train.py` constants changed during execution.
- Metric improvement: failed. `best_test_acc=92.60%` is below the required `93.33%`; classify as no-improvement in analysis.

### Informational Metrics
- final_test_acc: 92.35%
- final_test_loss: 0.3503
- training_seconds: 300.0
- total_seconds: 399.6
- startup_seconds: 3.0
- peak_vram_mb: 686.7
- num_epochs: 106
- num_steps: 40989
- num_params: 1,004,006

## Errors & Dead Ends

## Human Notes

> No human intervention during autopilot execution.
