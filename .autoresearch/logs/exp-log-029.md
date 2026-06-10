# EXP-029: Reflection Padding for RandomCrop

## Execution

Overall Status & Info:
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-029.md
- **Plan**: plans/plan-029.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-029
- **Commit**: 2928652
- **PR**: skipped — no git remote configured
- **Outcome**: completed

## Implementation Notes

### Summary

Implemented EXP-029 exactly as planned by changing only the training transform in `train.py`: the CIFAR-10 random crop now uses `padding_mode="reflect"` while preserving crop size, padding amount, horizontal flip, normalization, model width, optimizer, schedule, FP32 channels-last compile path, seed, and once-per-epoch evaluation. Preflight checks passed for Python syntax, ruff, diff scope, validation cadence, and transform presence.

### Surprises & Discoveries

No implementation surprises. The local torchvision `RandomCrop` signature includes `padding_mode`, matching the official documentation and allowing the experiment to remain a one-line augmentation-boundary change.

### Decisions

Kept all anchor settings unchanged to isolate the effect of reflected crop padding. No deviations from the plan were needed.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local command session 7433
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-08 19:51 UTC
- **Ended**: 2026-06-08 19:58 UTC

Description:
- Run the 28/56/112 ResNet-20 anchor with only the training crop padding changed from constant to reflected padding. The expectation is that reflected crop margins may reduce artificial zero-border artifacts without changing throughput, model capacity, optimizer dynamics, or validation cadence. The run must complete under the fixed 300s training budget and under 10 minutes total wall-clock time. It must reach `best_test_acc >= 93.33%` to count as an improvement over the current `93.23%` baseline.

Observations:
- Initial shell-background launch returned PID 1843913 but produced an empty `run.log` and no live process, so it was discarded before training started. The managed command-session launch started cleanly on GPU 0. (source: local process checks before Run 1)
- Startup confirms CUDA execution, `ResNet-20 | params: 822,790`, `Time budget: 300s`, and `Batches per epoch: 390`. (source: run.log L1-L4)
- Early training is healthy with no error patterns; best test accuracy reached 85.39% by epoch 13 while GPU 0 was active. (source: run.log L6-L30)
- Mid-run progress remains healthy with best test accuracy reaching 88.45% by epoch 22 and holding through epoch 28; no LR drop yet. (source: run.log L32-L60)
- Pre-drop accuracy reached 89.26% by epoch 43 and the run remained stable through epoch 45, still at LR 0.1. (source: run.log L80-L94)
- The first LR drop fired at step 21000 with `lr: 0.0100`; post-drop accuracy rose to 93.12% by epoch 57, below but near the 93.33% improvement threshold. (source: run.log L111-L118)
- Late refinement crossed the improvement threshold: epoch 72 reached 93.35%, epoch 73 reached 93.34%, and epoch 74 reached a new best of 93.58%. (source: run.log L148-L154)
- The run completed cleanly after 300.0 training seconds and 396.2 total seconds, reporting `best_test_acc: 93.58%`. (source: run.log L228-L237)

Key Metrics:
- best_test_acc: 93.58% (source: run.log L228)
- final_test_acc: 93.35% (source: run.log L229)
- final_test_loss: 0.2834 (source: run.log L230)
- training_seconds: 300.0 (source: run.log L231)
- total_seconds: 396.2 (source: run.log L232)
- peak_vram_mb: 660.4 (source: run.log L234)
- num_epochs: 111 (source: run.log L235)
- num_steps: 43,112 (source: run.log L236)
- num_params: 822,790 (source: run.log L237)

## Verification Results

### Conditions Checked
- Completion without crash: passed; `uv run train.py` exited 0 and printed a final summary. (source: command session 7433; run.log L228-L237)
- Numeric primary metric: passed; `best_test_acc: 93.58%` was reported. (source: run.log L228)
- Improvement threshold: passed; baseline is 93.23%, required threshold is 93.33%, and EXP-029 reached 93.58%. (source: experiment index baseline query; run.log L228)
- Scope: passed; the only tracked source diff is `train.py`, changing `RandomCrop(32, padding=4)` to `RandomCrop(32, padding=4, padding_mode="reflect")`. (source: `git diff -- train.py`)
- Validation cadence: passed; one `Eval()` construction and one epoch-level `evaluator.evaluate(...)` call remain. (source: train.py L30, L222)
- Preserved anchor checks: passed; batches per epoch is 390, the first LR drop fired at step 21000 with `lr: 0.0100`, and `num_params` is 822,790. (source: run.log L4, L111, L237)
- Runtime bound: passed; `total_seconds` is 396.2, below the 10-minute cap. (source: run.log L232)

### Informational Metrics
- final_test_acc: 93.35% (source: run.log L229)
- final_test_loss: 0.2834 (source: run.log L230)
- training_seconds: 300.0 (source: run.log L231)
- total_seconds: 396.2 (source: run.log L232)
- startup_seconds: 1.9 (source: run.log L233)
- peak_vram_mb: 660.4 (source: run.log L234)
- num_epochs: 111 (source: run.log L235)
- num_steps: 43,112 (source: run.log L236)
- num_params: 822,790 (source: run.log L237)

## Errors & Dead Ends

## Human Notes

> No human notes yet.
