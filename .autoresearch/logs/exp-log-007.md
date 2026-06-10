# EXP-007: Enable TF32 Throughput

## Execution

Overall Status & Info:
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-007.md
- **Plan**: plans/plan-007.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-007
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary
Implemented the planned narrow TF32 throughput change on top of the EXP-002 FP32 compile/channels-last baseline. `train.py` now defines `USE_TF32 = True` and, during CUDA setup before model construction and `torch.compile`, enables high float32 matmul precision plus CUDA matmul and cuDNN TF32 flags.

### Surprises & Discoveries
Local API check showed `torch.set_float32_matmul_precision` exists, `torch.backends.cuda.matmul.allow_tf32` defaulted to `False`, and `torch.backends.cudnn.allow_tf32` defaulted to `True`. The experiment makes both backend flags explicit.

### Decisions
Kept this as a single-variable throughput experiment: no architecture, schedule, optimizer, batch size, augmentation, seed, autocast, dependency, or evaluation cadence changes. The goal is to test TF32 against the proven ResNet-20 baseline without mixing in schedule or capacity confounds.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local PID 21952
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-08 12:44 UTC
- **Ended**: 2026-06-08 12:51 UTC

Description:
- Run the TF32-enabled FP32 compile/channels-last ResNet-20 recipe locally on one GPU with output redirected to `run.log`. This tests whether the repeated PyTorch TF32 warning can be converted into useful optimizer-step throughput while preserving the successful EXP-002 recipe. Success requires `best_test_acc >= 92.05%`.

Observations:
- Startup is clean under `CUDA_VISIBLE_DEVICES=0`: log reports `Device: cuda`, `ResNet-20 | params: 269,722`, `Time budget: 300s`, and `Batches per epoch: 390`. The previous TF32 tensor-core warning is absent in the startup tail. (source: run.log startup lines)
- Mid-run check is clean: no traceback, CUDA OOM, NaN/inf, or TF32 API errors are present. By epoch 70, before the first LR drop, the run reached `best: 88.13%`; GPU 0 was active with about 2.1 GiB allocated. (source: run.log lines 6-146 and live `nvidia-smi` check)
- Run completed without crash, but TF32 reduced completed work relative to EXP-002: only 37,922 steps and 98 epochs were completed. The first LR drop happened at step 32,000 / epoch 83 and the run never reached the second 48,000-step drop. (source: run.log final summary lines 205-211 and LR transition near epoch 83)

Key Metrics:
- best_test_acc: 91.39%
- final_test_acc: 91.17%
- final_test_loss: 0.3098
- training_seconds: 300.0
- total_seconds: 397.5
- startup_seconds: 2.4
- peak_vram_mb: 379.0
- num_epochs: 98
- num_steps: 37922
- num_params: 269,722

## Verification Results

### Conditions Checked
- Baseline and threshold: baseline script reports `baseline=91.95`; with the required +0.10 point margin, EXP-007 needed `best_test_acc >= 92.05%`. Result: pass for baseline confirmation, threshold established.
- Completion metrics: `run.log` includes numeric `best_test_acc: 91.39%` and `peak_vram_mb: 379.0`. Result: pass for completion metric extraction.
- Primary metric: `91.39% < 92.05%`. Result: fail; classify EXP-007 as no-improvement.
- Scope review: `git diff -- train.py` shows only `USE_TF32 = True` plus the planned CUDA TF32 setup block. Result: pass.
- Validation cadence review: `train.py` has one `evaluator.evaluate(model, device)` call in the epoch loop and the log has one `eval ep` record per completed epoch. Result: pass.

### Informational Metrics
- Final test accuracy: 91.17%
- Final test loss: 0.3098
- Training seconds: 300.0
- Total seconds: 397.5
- Startup seconds: 2.4
- Peak VRAM: 379.0 MB
- Epochs: 98
- Steps: 37,922
- Parameters: 269,722

## Errors & Dead Ends

## Human Notes

> {Researcher can add comments, corrections, or context here}

<!-- NOTE: Human notes are high trust and privileged relative to other info in this document. -->
