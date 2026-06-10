# EXP-006: Schedule-Calibrated ResNet-32

## Execution

Overall Status & Info:
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-006.md
- **Plan**: plans/plan-006.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-006
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: failed

## Implementation Notes

### Summary
Implemented the planned capacity increase by changing the existing CIFAR ResNet depth from ResNet-20 to ResNet-32 (`NUM_BLOCKS=5`) and adding explicit schedule constants. The scheduler now uses `LR_MILESTONES = [26000, 39000]` while preserving optimizer, batch size, augmentation, seed, FP32 arithmetic, channels-last, `torch.compile`, and once-per-epoch evaluation.

### Surprises & Discoveries
No implementation surprises. The existing `ResNet` class already parameterizes depth through `NUM_BLOCKS`, so the capacity change is a narrow top-level hyperparameter change rather than a new model implementation.

### Decisions
Kept the architecture change modest and stayed within the local ResNet implementation to avoid introducing a new family of shape or shortcut behavior. The milestones are explicit top-level constants so the experiment diff makes the schedule calibration visible and reversible.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local PID 4160739
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-08 12:10 UTC
- **Ended**: 2026-06-08 12:17 UTC

Description:
- Run the schedule-calibrated ResNet-32 recipe locally on one GPU with output redirected to `run.log`. This tests whether modest additional capacity can clear the tightened `92.05%` threshold while preserving the EXP-002 FP32 compile/channels-last throughput path. The expected outcome is a valid completed run with numeric `best_test_acc`; success requires `best_test_acc >= 92.05%`.

Observations:
- Startup is clean under `CUDA_VISIBLE_DEVICES=0`: log reports `Device: cuda`, `ResNet-32 | params: 464,154`, `Time budget: 300s`, and `Batches per epoch: 390`. The known non-fatal TF32 performance warning appears again. (source: run.log startup lines)
- Early training is clean with no traceback, CUDA OOM, or compile failure. ResNet-32 reaches 83.88% by epoch 12 while GPU 0 is actively utilized. (source: run.log lines 8-30; `nvidia-smi` occupancy check)
- Mid-run throughput is much lower than planned: around step 12.9k at 56.8% training time, with best accuracy 86.75% by epoch 24. At this pace EXP-006 may not reach the planned first LR drop at step 26k. (source: run.log lines 54-72)
- The run completed without crashing but never reached the first LR milestone: final `num_steps` is 23,642, below the planned first drop at 26,000. Best accuracy peaked at 88.18% at epoch 60, still at LR 0.1. (source: run.log lines 126-139)

Key Metrics:
- best_test_acc: 88.18%
- final_test_acc: 84.61%
- final_test_loss: 0.5499
- training_seconds: 300.0
- total_seconds: 367.7
- startup_seconds: 1.9
- peak_vram_mb: 476.3
- num_epochs: 61
- num_steps: 23,642
- num_params: 464,154

## Verification Results

### Conditions Checked
- Baseline and threshold: PASS. Experiment index reports baseline `91.95`; with the +0.10 point margin, EXP-006 requires `best_test_acc >= 92.05`. (source: `exp-index.sh baseline`; goal file)
- Completion and numeric metric: PASS. `run.log` reports a numeric `best_test_acc: 88.18%`. (source: run.log line 130)
- Primary metric condition: FAIL. `88.18% < 92.05%`, so EXP-006 is classified as no-improvement/research failure under the tightened threshold. (source: run.log line 130)
- Scope review: skipped — aborted verification after primary metric failure per execution protocol.
- Validation cadence review: skipped — aborted verification after primary metric failure per execution protocol.

### Informational Metrics
- final_test_acc: 84.61% (source: run.log line 131)
- final_test_loss: 0.5499 (source: run.log line 132)
- training_seconds: 300.0 (source: run.log line 133)
- total_seconds: 367.7 (source: run.log line 134)
- startup_seconds: 1.9 (source: run.log line 135)
- peak_vram_mb: 476.3 (source: run.log line 136)
- num_epochs: 61 (source: run.log line 137)
- num_steps: 23,642 (source: run.log line 138)
- num_params: 464,154 (source: run.log line 139)

## Errors & Dead Ends

## Human Notes

> {Researcher can add comments, corrections, or context here}

<!-- NOTE: Human notes are high trust and privileged relative to other info in this document. -->
