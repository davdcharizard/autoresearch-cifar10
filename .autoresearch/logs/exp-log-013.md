# EXP-013: ResNet-20 Width 1.5x with Proven 24k First Drop

## Execution

Overall Status & Info:
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-013.md
- **Plan**: plans/plan-013.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-013
- **Commit**: 3134138
- **PR**: skipped — no git remote configured
- **Outcome**: completed

## Implementation Notes

### Summary
Implemented the planned isolated capacity increase on top of the EXP-011 widened baseline. `train.py` now changes only `STAGE_WIDTHS` from `(20, 40, 80)` to `(24, 48, 96)`, while preserving the validated `[24000, 64000]` LR milestones and all optimizer, augmentation, precision, compilation, and evaluation settings.

### Surprises & Discoveries
No implementation surprises. The ResNet wiring already derives the stem, residual stages, and classifier width from `STAGE_WIDTHS`, so the 1.5x width experiment required only a single constant change.

### Decisions
Kept the first LR drop at step 24000 instead of moving it earlier because EXP-012 showed 22k was too early for the 20/40/80 widened model. This isolates whether more capacity, not scheduler retuning, can clear the 92.22% threshold.

## Experimental Adjustments
- PR creation was skipped because `git remote -v` returned no configured remotes. The successful experiment was still committed and fast-forward merged into `autoresearch/dev` locally.

## Run Log

### Run 1

Metadata:
- **Job ID**: local shell PID 277989
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-08 15:34 UTC
- **Ended**: 2026-06-08 15:41 UTC

Description:
- Run a 24/48/96 ResNet-20 with the proven `[24000, 64000]` milestone schedule. This tests whether a second moderate width increase raises the fixed-budget accuracy ceiling while retaining enough throughput to reach the 24k first LR drop. Success requires `best_test_acc >= 92.22%` against the current 92.12% baseline.

Observations:
- Physical GPU 0 was idle before launch, stale `run.log` was absent, and the single-GPU CUDA check reported one visible NVIDIA H20. EXP-013 launched on physical GPU 0 via `CUDA_VISIBLE_DEVICES=0`. (source: `nvidia-smi`, CUDA check, `ls run.log`)
- Startup log is being written and reports `Device: cuda`, `ResNet-20 | params: 605,026`, `Time budget: 300s`, and `Batches per epoch: 390`. Physical GPU 0 shows 561 MiB allocated during startup. (source: `run.log` startup lines, `nvidia-smi`)
- Early monitoring is clean through epoch 17: no traceback, CUDA OOM, NaN/Inf, or compile failure patterns are present. Best early test accuracy is 85.50%, and the run is still before the planned step-24000 LR drop. (source: `run.log` early epoch output and error-pattern scan)
- The planned first LR drop occurred at step 24000 during epoch 62 with about 123s remaining. Post-drop accuracy jumped to 91.39% at epoch 62 and reached 92.34% by epoch 71, clearing the 92.22% improvement threshold while the run continues. (source: `run.log` LR-drop and eval lines)
- The run completed successfully and improved the primary metric. Final summary reported `best_test_acc: 92.49%`, `final_test_acc: 92.25%`, 108 epochs, 41,825 steps, and 564.0 MiB peak VRAM. (source: `run.log` final summary)

Key Metrics:
- best_test_acc: 92.49%
- final_test_acc: 92.25%
- final_test_loss: 0.3530
- training_seconds: 300.0
- total_seconds: 404.4
- startup_seconds: 2.6
- peak_vram_mb: 564.0
- num_epochs: 108
- num_steps: 41825
- num_params: 605,026

## Verification Results

### Conditions Checked
- Baseline and threshold: passed. Current baseline is 92.12%, so EXP-013 required `best_test_acc >= 92.22%`.
- Single-GPU CUDA execution: passed. The run used one visible NVIDIA H20 via `CUDA_VISIBLE_DEVICES=0`.
- Experiment completion: passed. `run.log` contains parseable final summary metrics including `best_test_acc: 92.49%` and `peak_vram_mb: 564.0`.
- Primary metric condition: passed. `best_test_acc=92.49%` is +0.37 percentage points over the 92.12% baseline, exceeding the required +0.10-point margin.
- Schedule/throughput sanity review: passed. The run reached the first LR drop at step 24000, stayed at LR 0.01 afterward, and completed 41,825 steps over 108 epochs.
- Scope review: passed. `git diff -- train.py` shows only the planned `STAGE_WIDTHS` change from `(20, 40, 80)` to `(24, 48, 96)`.
- Validation cadence review: passed. `train.py` contains one `evaluator.evaluate(model, device)` call inside the epoch loop and no additional validation loop.

### Informational Metrics
- final_test_acc: 92.25%
- final_test_loss: 0.3530
- training_seconds: 300.0
- total_seconds: 404.4
- startup_seconds: 2.6
- peak_vram_mb: 564.0
- num_epochs: 108
- num_steps: 41825
- num_params: 605,026

## Errors & Dead Ends

## Human Notes

> {Researcher can add comments, corrections, or context here}

<!-- NOTE: Human notes are high trust and privileged relative to other info in this document. -->
