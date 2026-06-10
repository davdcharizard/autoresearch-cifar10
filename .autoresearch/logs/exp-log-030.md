# EXP-030: Reflection Anchor With 32k Second LR Drop

## Execution

Overall Status & Info:
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-030.md
- **Plan**: plans/plan-030.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-030
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed - no-improvement

## Implementation Notes

### Summary

Implemented EXP-030 exactly as planned by changing only `LR_MILESTONES` in `train.py` from `[21000, 64000]` to `[21000, 32000]`. The reflected `RandomCrop`, 28/56/112 ResNet-20 width, batch size, optimizer settings, FP32 channels-last compile path, seed, and once-per-epoch validation path were preserved. Preflight checks passed for Python syntax, ruff, diff scope, validation cadence, reflected crop padding, and schedule presence.

### Surprises & Discoveries

No implementation surprises. The current EXP-029 anchor already had reflected crop padding and the expected `LR_MILESTONES = [21000, 64000]`, so the EXP-030 implementation remained a one-line schedule change.

### Decisions

No deviations from the plan were needed. Kept all non-schedule settings unchanged to isolate whether a reachable second LR drop improves the reflection-padding anchor.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local command session 33356
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-08 20:08 UTC
- **Ended**: 2026-06-08 20:15 UTC

Description:
- Run the EXP-029 reflection-padding 28/56/112 ResNet-20 anchor with a reachable second LR drop at step 32000. The experiment tests whether converting late LR 0.01 oscillation into an LR 0.001 refinement phase improves `best_test_acc` without changing throughput-critical settings, capacity, augmentation, optimizer, or validation cadence. The run must complete under the fixed 300s training budget and under 10 minutes total wall-clock time. It must reach `best_test_acc >= 93.68%` to count as an improvement over the current `93.58%` baseline.

Observations:
- Startup confirms CUDA execution, `ResNet-20 | params: 822,790`, `Time budget: 300s`, and `Batches per epoch: 390`. (source: run.log L1-L4)
- Early training is healthy with no error patterns; best test accuracy reached 83.09% by epoch 7 while GPU 1 was active. (source: run.log L6-L47)
- The first LR drop fired at step 21000 with `lr: 0.0100`; post-drop accuracy rose to 93.27% by epoch 60, still below the 93.68% improvement threshold before the planned second drop. (source: run.log L111-L124)
- The second LR drop fired at step 32000 with `lr: 0.0010`; early post-second-drop evaluations stayed flat, with best accuracy remaining 93.28% through epoch 90. (source: run.log L169-L184)
- Late LR 0.001 refinement produced only a small lift to 93.33% by epoch 96, then plateaued. The run completed normally with `best_test_acc=93.33%`, below the 93.58% baseline and the 93.68% improvement threshold, so EXP-030 is no-improvement. (source: run.log L196-L227)

Key Metrics:
- best_test_acc: 93.33% (source: run.log L218)
- final_test_acc: 93.28% (source: run.log L219)
- final_test_loss: 0.2728 (source: run.log L220)
- training_seconds: 300.0 (source: run.log L221)
- total_seconds: 398.9 (source: run.log L222)
- startup_seconds: 2.4 (source: run.log L223)
- peak_vram_mb: 660.4 (source: run.log L224)
- num_epochs: 111 (source: run.log L225)
- num_steps: 43,208 (source: run.log L226)
- num_params: 822,790 (source: run.log L227)

## Verification Results

### Conditions Checked
- Baseline and threshold: PASS. Baseline query reported `baseline=93.58`; EXP-030 threshold is `93.68%` under the goal's +0.10 percentage-point rule. (source: exp-index.sh baseline output)
- Scope before launch/result: PASS. `git diff --name-only` reported only `train.py`; `git status --short --branch` showed branch `autoresearch/exp-030`, modified `train.py`, and untracked `data/`.
- Syntax and lint: PASS. `python3 -m py_compile train.py` exited 0 and `uv run ruff check train.py` reported all checks passed before launch.
- Validation cadence and augmentation: PASS. One `Eval()` construction and one epoch-level `evaluator.evaluate(...)` call remain; reflected crop padding is preserved. (source: train.py L30, L127, L222)
- Schedule implementation: PASS. `LR_MILESTONES = [21000, 32000]` was present in `train.py`, and the log shows step 21000 at `lr: 0.0100` and step 32000 at `lr: 0.0010`. (source: train.py L25; run.log L111, L169)
- Preserved batch size and parameter count: PASS. `Batches per epoch: 390` confirmed `BATCH_SIZE=128`, and `num_params=822,790`. (source: run.log L4, L227)
- Experiment completion: PASS. The process exited 0, printed numeric summary metrics, and `total_seconds=398.9`, under the 10-minute wall-clock cap. (source: command session 33356; run.log L218-L227)
- Metric improvement: FAIL. `best_test_acc=93.33%`, below the required `93.68%` threshold. Under the goal's +0.10 percentage-point rule, this is no-improvement. (source: run.log L218)
- Hard constraints: PASS. Only the planned schedule diff was present during the run, `training_seconds=300.0`, `total_seconds=398.9`, and no protected files changed. (source: `git diff -- train.py`; run.log L221-L222)

### Informational Metrics
- final_test_acc: 93.28% (source: run.log L219)
- final_test_loss: 0.2728 (source: run.log L220)
- training_seconds: 300.0 (source: run.log L221)
- total_seconds: 398.9 (source: run.log L222)
- startup_seconds: 2.4 (source: run.log L223)
- peak_vram_mb: 660.4 (source: run.log L224)
- num_epochs: 111 (source: run.log L225)
- num_steps: 43,208 (source: run.log L226)
- num_params: 822,790 (source: run.log L227)

## Errors & Dead Ends

## Human Notes

> No human intervention during autopilot execution.
