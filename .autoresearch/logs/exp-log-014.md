# EXP-014: ResNet-20 Width 1.75x with 22k First Drop

## Execution

Overall Status & Info:
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-014.md
- **Plan**: plans/plan-014.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-014
- **Commit**: 9b609e4
- **PR**: skipped — no git remote configured
- **Outcome**: completed

## Implementation Notes

### Summary
Implemented the planned schedule-calibrated width increase on top of the EXP-013 baseline. `train.py` now changes `STAGE_WIDTHS` from `(24, 48, 96)` to `(28, 56, 112)` and moves the first LR milestone from 24000 to 22000, while preserving all other training and evaluation settings.

### Surprises & Discoveries
No implementation surprises. The ResNet channel wiring still derives cleanly from `STAGE_WIDTHS`, and the scheduler remains centralized in `LR_MILESTONES`.

### Decisions
Kept the second LR milestone at 64000 so the run remains in LR 0.01 after the first drop and avoids the LR 0.001 phase that hurt EXP-003. The 22k first drop is intentional schedule calibration for the wider model's expected lower step budget.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local shell PID 328032
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-08 16:05 UTC
- **Ended**: 2026-06-08 16:12 UTC

Description:
- Run a 28/56/112 ResNet-20 with the first LR milestone moved to step 22000. This tests whether another width step can clear the new 92.59% threshold while preserving enough LR 0.01 refinement time under the fixed 300s budget.

Observations:
- Physical GPU 0 was idle before launch, stale `run.log` was absent, and the single-GPU CUDA check reported one visible NVIDIA H20. EXP-014 launched on physical GPU 0 via `CUDA_VISIBLE_DEVICES=0`. (source: `nvidia-smi`, CUDA check, `ls run.log`)
- Startup log is being written and reports `Device: cuda`, `ResNet-20 | params: 822,790`, `Time budget: 300s`, and `Batches per epoch: 390`. Physical GPU 0 shows 403 MiB allocated during startup. (source: `run.log` startup lines, `nvidia-smi`)
- Early monitoring is clean through epoch 15: no traceback, CUDA OOM, NaN/Inf, or compile failure patterns are present. Best early test accuracy is 84.29%, and the run is still before the planned step-22000 LR drop. (source: `run.log` early epoch output and error-pattern scan)
- The planned first LR drop occurred at step 22000 during epoch 57 with about 123s remaining. Post-drop accuracy jumped to 91.81% at epoch 57, reached 92.65% at epoch 59, and peaked at 93.06% by epoch 61, clearing the 92.59% improvement threshold while the run continues. (source: `run.log` LR-drop and eval lines)
- The run completed normally with final summary metrics present. Best accuracy improved to 93.09%, exceeding the 92.49% baseline by +0.60 points and clearing the tightened +0.10 point threshold. (source: `run.log` lines 184-193)

Key Metrics:
- best_test_acc: 93.09%
- final_test_acc: 92.92%
- final_test_loss: 0.2995
- training_seconds: 300.0
- total_seconds: 395.3
- startup_seconds: 3.5
- peak_vram_mb: 661.9
- num_epochs: 88
- num_steps: 34259
- num_params: 822,790

## Verification Results

### Conditions Checked
- Baseline and threshold: passed. `exp-index.sh baseline` reported baseline=92.49, so EXP-014 required best_test_acc >= 92.59.
- Single-GPU CUDA execution: passed. `CUDA_VISIBLE_DEVICES=0 uv run python ...` reported CUDA available, visible device count 1, and device name NVIDIA H20.
- Completion metric present: passed. `run.log` includes numeric `best_test_acc: 93.09%` and `peak_vram_mb: 661.9`.
- Primary metric condition: passed. 93.09 >= 92.59, so the run is an improvement under the tightened +0.10 point rule.
- Schedule/throughput sanity: passed. The first LR drop occurred at step 22000, the run reached 34259 steps over 88 epochs, and the model had 822,790 parameters.
- Error-pattern scan: passed. `grep -n "Traceback\|CUDA out of memory\|RuntimeError\|Error\|nan\|inf" run.log` returned no matches.
- Scope review: passed. `git diff -- train.py` only changes `STAGE_WIDTHS` from `(24, 48, 96)` to `(28, 56, 112)` and `LR_MILESTONES` from `[24000, 64000]` to `[22000, 64000]`.
- Validation cadence review: passed. `train.py` has one `evaluator = Eval()` construction and one `evaluator.evaluate(model, device)` call inside the epoch path; no extra validation loop was added.

### Informational Metrics
- final_test_acc: 92.92%
- final_test_loss: 0.2995
- training_seconds: 300.0
- total_seconds: 395.3
- startup_seconds: 3.5
- peak_vram_mb: 661.9
- num_epochs: 88
- num_steps: 34259
- num_params: 822,790

## Errors & Dead Ends

## Human Notes

> {Researcher can add comments, corrections, or context here}

<!-- NOTE: Human notes are high trust and privileged relative to other info in this document. -->
