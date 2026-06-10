# EXP-016: 28/56/112 First LR Drop at 21k

## Execution

Overall Status & Info:
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-016.md
- **Plan**: plans/plan-016.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-016
- **Commit**: f187edf
- **PR**: skipped — no git remote configured
- **Outcome**: completed

## Implementation Notes

### Summary
Implemented the planned isolated earlier schedule retune on top of the EXP-014 baseline. `train.py` keeps `STAGE_WIDTHS = (28, 56, 112)` and changes only `LR_MILESTONES` from `[22000, 64000]` to `[21000, 64000]`.

### Surprises & Discoveries
No implementation surprises. The scheduler remains centralized in `LR_MILESTONES`, so the experiment is a one-line configuration change.

### Decisions
Kept the second LR milestone at 64000 so the run remains in LR 0.01 after the first drop. This isolates whether the current 28/56/112 model benefits from 1k more low-LR refinement time after EXP-015 showed 23k is too late.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local shell PID 664192
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-08 16:32 UTC
- **Ended**: 2026-06-08 16:39 UTC

Description:
- Run the current 28/56/112 ResNet-20 with first LR milestone moved from step 22000 to 21000. This tests whether more LR 0.01 refinement can improve the 93.09% baseline to at least 93.19% after the 23k retune failed.

Observations:
- Physical GPU 0 was idle before launch, stale `run.log` was removed, and the single-GPU CUDA check reported one visible NVIDIA H20. EXP-016 will launch on physical GPU 0 via `CUDA_VISIBLE_DEVICES=0`. (source: `nvidia-smi`, CUDA check)
- Startup log is being written and reports `Device: cuda`, `ResNet-20 | params: 822,790`, `Time budget: 300s`, and `Batches per epoch: 390`. Physical GPU 0 shows 821 MiB allocated during startup. (source: `run.log` startup lines, `nvidia-smi`)
- Early monitoring is clean through epoch 24: no traceback, CUDA OOM, NaN/Inf, or compile failure patterns are present. Best early test accuracy is 88.14%, and the run is still before the planned step-21000 LR drop. (source: `run.log` early epoch output and error-pattern scan)
- The planned first LR drop occurred at step 21000 during epoch 54 with about 145s remaining. Post-drop accuracy reached 92.99% by epoch 59 and 93.14% by epoch 60, just below the 93.19% threshold while the run continues. (source: `run.log` LR-drop and eval lines)
- Run completed normally after the fixed 300s training budget. Final `best_test_acc` was 93.23%, which is +0.14 percentage points over the 93.09% baseline and clears the tightened 93.19% improvement threshold. (source: `run.log` final summary)

Key Metrics:
- best_test_acc: 93.23%
- final_test_acc: 93.03%
- final_test_loss: 0.3014
- training_seconds: 300.0
- total_seconds: 380.6
- startup_seconds: 1.9
- peak_vram_mb: 660.4
- num_epochs: 88
- num_steps: 34208
- num_params: 822,790

## Verification Results

### Conditions Checked
- [x] Baseline and threshold confirmed: current experiment-index baseline was 93.09%, so EXP-016 needed `best_test_acc >= 93.19%` under the +0.10 point rule.
- [x] Single-GPU CUDA execution confirmed before launch: one visible NVIDIA H20 via `CUDA_VISIBLE_DEVICES=0`.
- [x] Completion metric present: `run.log` contains numeric `best_test_acc: 93.23%` and final summary metrics.
- [x] Primary metric condition passed: 93.23% is >= 93.19%.
- [x] Schedule sanity passed: the run reached the planned first LR drop at step 21000, stayed at LR 0.01 through completion, and ended at 34,208 steps / 88 epochs with 822,790 parameters.
- [x] Error scan passed: no traceback, CUDA OOM, NaN/Inf, or compile-failure patterns were found in `run.log`.
- [x] Scope review passed: `git diff -- train.py` shows only `LR_MILESTONES = [22000, 64000]` -> `[21000, 64000]`.
- [x] Validation cadence passed: `train.py` has one `evaluator.evaluate(model, device)` call in the epoch loop and no extra validation loop was added.

### Informational Metrics
- final_test_acc: 93.03%
- final_test_loss: 0.3014
- training_seconds: 300.0
- total_seconds: 380.6
- startup_seconds: 1.9
- peak_vram_mb: 660.4
- num_epochs: 88
- num_steps: 34208
- num_params: 822,790

## Errors & Dead Ends

## Human Notes

> {Researcher can add comments, corrections, or context here}

<!-- NOTE: Human notes are high trust and privileged relative to other info in this document. -->
