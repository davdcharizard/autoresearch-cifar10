# EXP-009: Weak 8x8 Cutout

## Execution

Overall Status & Info:
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-009.md
- **Plan**: plans/plan-009.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-009
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: failed

## Implementation Notes

### Summary
Implemented the planned weak cutout augmentation on top of the EXP-002 FP32 compile/channels-last ResNet-20 baseline. `train.py` now defines explicit cutout constants and conditionally appends a fixed 8x8 `transforms.RandomErasing` operation after normalization in the training transform.

### Surprises & Discoveries
No implementation surprises. The existing transform pipeline is a simple `transforms.Compose`, and Python list expansion allowed the cutout transform to be gated by `USE_CUTOUT` without restructuring the pipeline.

### Decisions
Kept the augmentation placement after normalization to match EXP-005, so this experiment isolates mask strength rather than transform placement. All model, optimizer, LR schedule, precision, throughput, seed, and evaluation settings were preserved.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local shell PID 79905
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-08 13:14 UTC
- **Ended**: 2026-06-08 13:22 UTC

Description:
- Run the weak 8x8 cutout FP32 compile/channels-last ResNet-20 recipe locally on one GPU with output redirected to `run.log`. This tests whether a much smaller and lower-probability mask can add useful generalization without EXP-005's 16x16 convergence delay. Success requires `best_test_acc >= 92.05%`.

Observations:
- Physical GPU 0 was occupied by an unrelated training process in `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-opus-4-8`, so EXP-009 launched on physical GPU 1 via `CUDA_VISIBLE_DEVICES=1`. Startup is clean: log reports `Device: cuda`, `ResNet-20 | params: 269,722`, `Time budget: 300s`, and `Batches per epoch: 390`. (source: `nvidia-smi`, `pgrep`, and run.log startup lines)
- No Python traceback, CUDA OOM, transform error, or NaN/inf pattern was found in `run.log`. Early progress was clean, with best accuracy reaching 85.83% by epoch 27 and 87.63% by epoch 77 before the first LR drop. (source: `run.log` lines 58 and 158)
- After the step-32000 LR drop to 0.01, best accuracy improved to 91.78% at epoch 97 and peaked at 91.87% at epoch 110. The run completed normally with final test accuracy 91.62%. (source: `run.log` lines 198, 224, 244-253)

Key Metrics:
- best_test_acc: 91.87%
- final_test_acc: 91.62%
- final_test_loss: 0.3167
- training_seconds: 300.0
- total_seconds: 406.3
- startup_seconds: 2.4
- peak_vram_mb: 379.0
- num_epochs: 119
- num_steps: 46047
- num_params: 269,722

## Verification Results

### Conditions Checked
- Baseline and threshold: pass. `exp-index.sh baseline` reports `baseline=91.95`, so the +0.10 point gate requires `best_test_acc >= 92.05%`.
- Single-GPU execution: pass. EXP-009 ran with `CUDA_VISIBLE_DEVICES=1` on one NVIDIA H20-class GPU because physical GPU 0 was occupied by an unrelated run; this was recorded as an execution adjustment.
- Completion and metric parse: pass. `run.log` includes numeric summary metrics, including `best_test_acc: 91.87%` and `peak_vram_mb: 379.0`.
- Primary metric condition: fail. `best_test_acc=91.87%` is -0.08 points below the 91.95% baseline and -0.18 points below the 92.05% improvement threshold, so EXP-009 is `no-improvement`.
- Remaining checks: skipped after primary metric failure per verification procedure. Scope was spot-checked during execution; `git diff -- train.py` shows only the planned cutout constants and transform addition.

### Informational Metrics
- `final_test_acc=91.62%`, `final_test_loss=0.3167`, `training_seconds=300.0`, `total_seconds=406.3`, `startup_seconds=2.4`, `peak_vram_mb=379.0`, `num_epochs=119`, `num_steps=46047`, `num_params=269,722`.

## Errors & Dead Ends

## Human Notes

> {Researcher can add comments, corrections, or context here}

<!-- NOTE: Human notes are high trust and privileged relative to other info in this document. -->
