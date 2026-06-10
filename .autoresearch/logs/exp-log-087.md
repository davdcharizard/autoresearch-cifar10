# EXP-087: Fine Upper Flip Bracket p=0.425 Under Padding 3

## Execution

Overall Status & Info:
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-087.md
- **Plan**: plans/plan-087.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-087
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary

Implemented the approved fine upper flip bracket on the EXP-085 spatial anchor. `train.py` changes only `RandomHorizontalFlip(p=0.4)` to `RandomHorizontalFlip(p=0.425)` while preserving reflection crop padding 3, and updates the startup marker to `RandomHorizontalFlip p: 0.425`. All other augmentation, CutMix, architecture, optimizer, schedule, seed, compile/channels-last, validation cadence, and fixed-budget behavior were preserved.

### Surprises & Discoveries

No implementation surprises. The target transform and existing marker were explicit and localized.

### Decisions

Kept EXP-087 isolated to flip probability on the validated padding-3 spatial anchor, as planned. No coupled changes to crop padding, CutMix, schedule, optimizer, normalization, or architecture were added, preserving a clean comparison against the 94.51% baseline and the 94.61% improvement threshold.

## Experimental Adjustments

None so far.

## Run Log

### Run 1

Metadata:
- **Job ID**: local attached session 28871; launcher PID 4130071; uv PID 4130072; train PID 4130075
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-10 01:00 UTC
- **Ended**: 2026-06-10 01:09 UTC

Description:
- Local single-GPU foreground run of EXP-087 using the current CutMix anchor with reflection crop padding 3 preserved and horizontal flip probability increased from 0.4 to 0.425. This tests whether the EXP-085 padding-3 anchor benefits from a small amount of restored horizontal invariance after EXP-086 showed that further reducing crop jitter is harmful. The run is expected to preserve throughput, parameter count, the step-21000 LR drop, and fixed wall-clock training behavior while attempting to reach at least 94.61% `best_test_acc`.

Observations:
- GPU0 was selected because `nvidia-smi` showed GPU0 at 0 MiB / 0% utilization while GPU1 had active load from another workspace. (source: `nvidia-smi` pre-launch output, 2026-06-10 01:00 UTC)
- Startup confirmed CUDA execution, `RandomCrop padding: 3 reflect`, `RandomHorizontalFlip p: 0.425`, `ResNet-20 | params: 822,790`, unchanged CutMix alpha/probability/label smoothing, and a 300s time budget. (source: run.log L1-L7)
- The first LR drop was reached on schedule at step 21000 with `lr: 0.0100`; pre-drop best was 89.01% at epoch 44 and stayed 89.01% through epoch 53. (source: run.log L91, L113-L115)
- Early post-drop convergence reached 93.45% by epoch 59, still below the 94.61% improvement threshold. (source: run.log L115-L125)
- The run peaked at 94.34% on epoch 75 and never exceeded that value through epoch 102. This is below both the 94.51% baseline and the 94.61% noise-guard improvement threshold. (source: run.log L157-L211)

Key Metrics:
- `best_test_acc`: 94.34%
- `final_test_acc`: 93.20%
- `final_test_loss`: 0.2988
- `training_seconds`: 300.0
- `total_seconds`: 395.8
- `startup_seconds`: 2.5
- `peak_vram_mb`: 660.4
- `num_epochs`: 102
- `num_steps`: 39,425
- `num_params`: 822,790

## Verification Results

### Conditions Checked
- **Code-scope constraint**: passed. `git diff --name-only` listed only `train.py`; untracked `data/` was unrelated and preserved.
- **Syntax and style**: passed. `python3 -m py_compile train.py` exited 0, and `uv run ruff check train.py` reported `All checks passed!`.
- **Implementation intent**: passed. `git diff train.py` changed only `RandomHorizontalFlip(p=0.4)` to `p=0.425` and the matching startup marker; `RandomCrop padding: 3 reflect` and CutMix settings were unchanged.
- **Startup markers**: passed. `run.log` confirmed CUDA execution, `RandomCrop padding: 3 reflect`, `RandomHorizontalFlip p: 0.425`, `ResNet-20 | params: 822,790`, `CutMix alpha: 1.0, prob: 0.5, label smoothing: 0.05`, and `Time budget: 300s`.
- **Scheduler integrity**: passed. The first LR drop was reached at step 21000 with `lr: 0.0100`.
- **Primary metric availability**: passed. `run.log` reported numeric `best_test_acc: 94.34%`.
- **Hard constraints**: passed. Only `train.py` changed; no dependency, data, `prepare.py`, evaluation-harness, seed, validation cadence, architecture, optimizer, LR milestone, normalization, CutMix, or fixed-budget behavior changes were made.
- **Improvement threshold**: failed for improvement classification. Baseline was 94.51%, the +0.10 percentage-point threshold required 94.61%, and EXP-087 reached 94.34%, so the verdict is `no-improvement`.

### Informational Metrics
- `final_test_acc`: 93.20%
- `final_test_loss`: 0.2988
- `training_seconds`: 300.0
- `total_seconds`: 395.8
- `startup_seconds`: 2.5
- `peak_vram_mb`: 660.4
- `num_epochs`: 102
- `num_steps`: 39,425
- `num_params`: 822,790

## Errors & Dead Ends
- No crash, OOM, NaN, non-finite-loss, or infrastructure error occurred. The approach failed because the primary metric did not exceed the baseline or threshold.

## Human Notes

> Researcher requested autopilot continuation; no execution-phase intervention was requested.
