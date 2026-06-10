# EXP-010: Isolated Nesterov Momentum

## Execution

Overall Status & Info:
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-010.md
- **Plan**: plans/plan-010.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-010
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: failed

## Implementation Notes

### Summary
Implemented the planned optimizer-only Nesterov ablation on top of the EXP-002 FP32 compile/channels-last ResNet-20 baseline. `train.py` now defines `USE_NESTEROV = True` and passes `nesterov=USE_NESTEROV` to the existing SGD optimizer while preserving all other model, data, loss, schedule, precision, and evaluation settings.

### Surprises & Discoveries
No implementation surprises. PyTorch's SGD already supports Nesterov with the existing nonzero momentum and zero dampening defaults, so no optimizer restructuring was needed.

### Decisions
Used an explicit boolean hyperparameter instead of an inline literal so the diff remains auditable and reversible in the same style as prior experiment flags. No deviations from the plan.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local shell PID 124124
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-08 13:39 UTC
- **Ended**: 2026-06-08 13:46 UTC

Description:
- Run the isolated Nesterov FP32 compile/channels-last ResNet-20 recipe locally on one GPU with output redirected to `run.log`. This tests whether the previously confounded Nesterov optimizer component improves optimization or late-stage refinement without changing throughput, model capacity, augmentation, loss, schedule, seed, or evaluator. Success requires `best_test_acc >= 92.05%`.

Observations:
- EXP-010 launched on physical GPU 0 via `CUDA_VISIBLE_DEVICES=0`; both H20 GPUs were idle before launch and no stale `run.log` existed. Startup was still pending immediately after launch, with GPU 0 memory allocated by the training process. (source: `nvidia-smi`, `pgrep`, and run.log tail check at 2026-06-08 13:39 UTC)
- Startup completed cleanly: `run.log` reports `Device: cuda`, `ResNet-20 | params: 269,722`, `Time budget: 300s`, and `Batches per epoch: 390`. No traceback, CUDA OOM, optimizer error, or NaN/inf pattern was found during early monitoring. Best accuracy reached 85.10% by epoch 23. (source: `run.log` lines 1-62)
- The LR drop at step 32000 happened in epoch 83, but post-drop accuracy plateaued around 91.3% instead of approaching the 92.05% target. Peak accuracy was 91.33% at epoch 96; final accuracy was 90.74%. (source: `run.log` lines 169-238)

Key Metrics:
- best_test_acc: 91.33%
- final_test_acc: 90.74%
- final_test_loss: 0.3637
- training_seconds: 300.0
- total_seconds: 403.6
- startup_seconds: 2.9
- peak_vram_mb: 379.0
- num_epochs: 116
- num_steps: 45163
- num_params: 269,722

## Verification Results

### Conditions Checked
- Baseline and threshold: pass. `exp-index.sh baseline` reports `baseline=91.95`, so the +0.10 point gate requires `best_test_acc >= 92.05%`.
- Single-GPU execution: pass. EXP-010 ran with `CUDA_VISIBLE_DEVICES=0` on one NVIDIA H20-class GPU.
- Completion and metric parse: pass. `run.log` includes numeric summary metrics, including `best_test_acc: 91.33%` and `peak_vram_mb: 379.0`.
- Primary metric condition: fail. `best_test_acc=91.33%` is -0.62 points below the 91.95% baseline and -0.72 points below the 92.05% improvement threshold, so EXP-010 is `no-improvement`.
- Remaining checks: skipped after primary metric failure per verification procedure. Scope was spot-checked during execution; `git diff -- train.py` shows only the planned `USE_NESTEROV` constant and SGD `nesterov` argument.

### Informational Metrics
- `final_test_acc=90.74%`, `final_test_loss=0.3637`, `training_seconds=300.0`, `total_seconds=403.6`, `startup_seconds=2.9`, `peak_vram_mb=379.0`, `num_epochs=116`, `num_steps=45163`, `num_params=269,722`.

## Errors & Dead Ends

## Human Notes

> {Researcher can add comments, corrections, or context here}

<!-- NOTE: Human notes are high trust and privileged relative to other info in this document. -->
