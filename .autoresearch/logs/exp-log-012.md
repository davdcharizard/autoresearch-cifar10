# EXP-012: Earlier First LR Drop on Widened ResNet-20

## Execution

Overall Status & Info:
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-012.md
- **Plan**: plans/plan-012.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-012
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: failed

## Implementation Notes

### Summary
Implemented the planned isolated scheduler retune on top of the successful EXP-011 widened ResNet-20 baseline. `train.py` now changes only `LR_MILESTONES` from `[24000, 64000]` to `[22000, 64000]`, preserving the 20/40/80 stage widths and all other training settings.

### Surprises & Discoveries
No implementation surprises. The scheduler is already centralized in the `LR_MILESTONES` constant, so the experiment required only a one-line change.

### Decisions
Kept the second milestone at 64000 so the run remains in LR 0.01 after the first drop and avoids the LR 0.001 phase that hurt EXP-003. No other schedule, optimizer, or architecture knobs were changed so the run isolates the first-drop timing.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local shell PID 221156
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-08 14:57 UTC
- **Ended**: 2026-06-08 15:04 UTC

Description:
- Run the EXP-011 widened ResNet-20 recipe with only the first LR milestone moved from 24000 to 22000. This tests whether the final-epoch EXP-011 peak was limited by too little LR 0.01 refinement time. Success requires `best_test_acc >= 92.22%` against the current 92.12% baseline.

Observations:
- Physical GPU 0 was occupied by an unrelated run in `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-opus-4-8`, so EXP-012 launched on physical GPU 1 via `CUDA_VISIBLE_DEVICES=1`. Stale `run.log` was absent before launch. (source: `nvidia-smi`, `pgrep`, `ls run.log`)
- Startup log is being written and reports `Device: cuda`, `ResNet-20 | params: 420,670`, `Time budget: 300s`, and `Batches per epoch: 390`. Physical GPU 1 shows 401 MiB allocated during startup. (source: `run.log` startup lines, `nvidia-smi`)
- Early monitoring is clean through epoch 31: no traceback, CUDA OOM, NaN/Inf, or compile failure patterns are present. The best early test accuracy is 86.47% and the run is still before the planned step-22000 LR drop. (source: `run.log` lines 6-66)
- The planned first LR drop occurred at step 22000 during epoch 57, switching to `lr: 0.0100`. Post-drop accuracy jumped to 91.96% by epoch 59, but remains below the 92.22% improvement threshold while the run continues. (source: `run.log` lines 117-124)
- The run completed successfully but did not meet the primary metric condition. Final summary reported `best_test_acc: 92.16%`, which is below the required 92.22% threshold, with final accuracy 91.66% after 117 epochs and 45478 steps. (source: `run.log` lines 240-249)

Key Metrics:
- best_test_acc: 92.16%
- final_test_acc: 91.66%
- final_test_loss: 0.3880
- training_seconds: 300.0
- total_seconds: 405.5
- startup_seconds: 2.9
- peak_vram_mb: 468.3
- num_epochs: 117
- num_steps: 45478
- num_params: 420,670

## Verification Results

### Conditions Checked
- Baseline and threshold: passed. Current baseline is 92.12%, so EXP-012 required `best_test_acc >= 92.22%`.
- Single-GPU CUDA execution: passed. The run used one visible NVIDIA H20 via `CUDA_VISIBLE_DEVICES=1` because physical GPU 0 was occupied by an unrelated run.
- Experiment completion: passed. `run.log` contains parseable final summary metrics including `best_test_acc: 92.16%` and `peak_vram_mb: 468.3`.
- Primary metric condition: failed. `best_test_acc=92.16%` is below the required 92.22% threshold, so EXP-012 is a valid no-improvement despite being +0.04 points above the 92.12% baseline.
- Schedule/throughput sanity review: skipped — aborted verification after primary metric failure.
- Scope review: skipped — aborted verification after primary metric failure.
- Validation cadence review: skipped — aborted verification after primary metric failure.

### Informational Metrics

## Errors & Dead Ends

## Human Notes

> {Researcher can add comments, corrections, or context here}

<!-- NOTE: Human notes are high trust and privileged relative to other info in this document. -->
