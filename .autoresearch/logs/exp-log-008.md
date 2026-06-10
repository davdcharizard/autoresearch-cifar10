# EXP-008: Earlier First LR Drop Without Second Drop

## Execution

Overall Status & Info:
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-008.md
- **Plan**: plans/plan-008.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-008
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary
Implemented the planned schedule-only intervention on top of the EXP-002 FP32 compile/channels-last ResNet-20 baseline. `train.py` now uses `MultiStepLR` milestones `[30000, 64000]` instead of `[32000, 48000]`, moving the first LR drop modestly earlier and making the second drop unreachable under the expected fixed-budget step count.

### Surprises & Discoveries
No implementation surprises. The scheduler is constructed inline, so the experiment is a one-line change and local checks confirmed the diff is limited to that line.

### Decisions
Kept the intervention deliberately isolated: no architecture, optimizer, augmentation, precision, batch size, seed, throughput flag, dependency, or evaluation cadence changes. The purpose is to test whether more LR 0.01 refinement helps without repeating EXP-003's reachable LR 0.001 phase.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local shell PID 53323
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-08 13:00 UTC
- **Ended**: 2026-06-08 13:06 UTC

Description:
- Run the `[30000, 64000]` scheduler variant locally on one GPU with output redirected to `run.log`. This tests whether moving only the first LR drop earlier gives more LR 0.01 refinement while avoiding the failed LR 0.001 phase. Success requires `best_test_acc >= 92.05%`.

Observations:
- Startup is clean under `CUDA_VISIBLE_DEVICES=0`: CUDA sees one NVIDIA H20, log reports `Device: cuda`, `ResNet-20 | params: 269,722`, `Time budget: 300s`, and `Batches per epoch: 390`. Training progress is being written to `run.log` at LR 0.1000. (source: run.log startup lines)
- Early monitoring is clean: no traceback, CUDA OOM, NaN/inf, compile error, or scheduler error patterns are present. By epoch 27 the run reached `best: 85.37%` while still at LR 0.1000, and GPU 0 was active. (source: run.log lines 6-70 and live `nvidia-smi` check)
- Mid-run schedule behavior matches the plan: step 30,000 switched to `lr: 0.0100`, and no `lr: 0.0010` lines are present. By epoch 86 the run reached `best: 91.42%`, still below the 92.05% success threshold with roughly 80 seconds remaining. (source: run.log lines 157-179)
- Run completed cleanly with `best_test_acc: 91.65%`, below both the 91.95% baseline and the 92.05% threshold. It completed 46,331 steps and 119 epochs, stayed at LR 0.0100 after step 30,000, and never reached the second 64,000-step milestone. (source: run.log lines 244-253 and scheduler grep)

Key Metrics:
- best_test_acc: 91.65%
- final_test_acc: 90.94%
- final_test_loss: 0.3779
- training_seconds: 300.0
- total_seconds: 405.5
- startup_seconds: 2.5
- peak_vram_mb: 379.0
- num_epochs: 119
- num_steps: 46331
- num_params: 269,722

## Verification Results

### Conditions Checked
- Baseline and threshold: baseline script reports `baseline=91.95`; with the required +0.10 point margin, EXP-008 needed `best_test_acc >= 92.05%`. Result: pass for baseline confirmation, threshold established.
- Completion metrics: `run.log` includes numeric `best_test_acc: 91.65%` and `peak_vram_mb: 379.0`. Result: pass for completion metric extraction.
- Primary metric: `91.65% < 92.05%`. Result: fail; classify EXP-008 as no-improvement.
- Schedule behavior: `run.log` shows `step 30000 ... lr: 0.0100` and no `lr: 0.0010` lines. Result: pass.
- Scope review: `git diff -- train.py` shows only the planned scheduler milestone change. Result: pass.
- Validation cadence review: `train.py` has one `evaluator.evaluate(model, device)` call in the epoch loop and the log has one `eval ep` record per completed epoch. Result: pass.

### Informational Metrics
- Final test accuracy: 90.94%
- Final test loss: 0.3779
- Training seconds: 300.0
- Total seconds: 405.5
- Startup seconds: 2.5
- Peak VRAM: 379.0 MB
- Epochs: 119
- Steps: 46,331
- Parameters: 269,722

## Errors & Dead Ends

## Human Notes

> {Researcher can add comments, corrections, or context here}

<!-- NOTE: Human notes are high trust and privileged relative to other info in this document. -->
