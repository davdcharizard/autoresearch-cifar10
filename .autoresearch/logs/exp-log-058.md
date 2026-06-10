# EXP-058: Squeeze-and-Excitation BasicBlocks

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-058.md
- **Plan**: plans/plan-058.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-058
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary

Implemented the planned SE residual-block probe in `train.py` only. The change adds `USE_SE=True` and `SE_REDUCTION=16`, defines an `SEBlock` with adaptive average pooling and two `1x1` convolution layers, and applies it after `bn2(conv2)` before shortcut addition in every `BasicBlock`. The existing optimizer, stage widths, schedule, data augmentation, label smoothing, compile path, channels-last path, and validation cadence were left unchanged.

### Surprises & Discoveries

No code-structure surprises. The existing block already had a clean insertion point after the second BatchNorm and before shortcut addition.

### Decisions

Used `nn.Conv2d` for the SE bottleneck instead of flattening plus `nn.Linear` so the gate stays naturally 4D and channels-last friendly. Kept a minimum bottleneck width of 4 to avoid overly narrow gates for the 28-channel first stage while still limiting parameter overhead.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local session 27986; shell PID 3483022; uv PID 3483023; python PID 3483026
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-09 18:01
- **Ended**: 2026-06-09 18:08

Description:
- Local foreground run of EXP-058 on one selected GPU. This run tests whether lightweight SE channel recalibration inside every residual block can improve the current CIFAR-10 ResNet-20 anchor without changing the training recipe. Expected behavior is a valid `uv run train.py` completion within 10 minutes, numeric `best_test_acc`, the step-21000 LR drop, and classification against the `94.07%` improvement threshold.

Observations:
- Startup succeeded on CUDA with `ResNet-20 | params: 830,143`, `SE blocks: enabled, reduction=16`, and unchanged `Batches per epoch: 390`. This confirms the SE parameter overhead is +7,353 versus the 822,790-param anchor and that batch geometry is unchanged. (source: run.log L1-L5)
- Preflight passed: tracked code diff is limited to `train.py`, `python3 -m py_compile train.py` exited 0, and `uv run ruff check train.py` reported `All checks passed!`.
- Baseline check passed before launch: `baseline=93.97`, `baseline_commit=755be2c`, so the improvement threshold is `94.07%`.
- GPU0 was selected because it was idle (`0MiB`, `0%` utilization) while another checkout had activity on GPU1. (source: `nvidia-smi` at 2026-06-09 18:01 UTC)
- Early training was stable through epoch 25 with best `87.04%`; step timing was mostly around 9-14ms/batch and the first LR drop remained reachable. (source: run.log L55-L57)
- First LR drop was confirmed at `step 21000 ep 54` with `lr: 0.0100`, and post-drop validation climbed to `93.57%` by epoch 58. (source: run.log L114-L123)
- Run completed cleanly with `best_test_acc=93.71%`, `final_test_acc=93.65%`, `num_epochs=66`, and `num_steps=25,716`. The result is a valid no-improvement because it is below the `93.97%` baseline and the required `94.07%` improvement threshold. (source: run.log L141-L150)

Key Metrics:
- `best_test_acc`: 93.71%
- `final_test_acc`: 93.65%
- `final_test_loss`: 0.2251
- `training_seconds`: 300.0
- `total_seconds`: 380.7
- `startup_seconds`: 2.5
- `peak_vram_mb`: 662.3
- `num_epochs`: 66
- `num_steps`: 25,716
- `num_params`: 830,143

## Verification Results

### Conditions Checked
- Baseline: pass. `exp-index.sh baseline` reported `baseline=93.97`, `baseline_commit=755be2c`; improvement threshold is `94.07%`.
- Scope: pass. `git diff --name-only` lists only `train.py`.
- Compile: pass. `python3 -m py_compile train.py` exited 0 during preflight.
- Style: pass. `uv run ruff check train.py` reported `All checks passed!` during preflight.
- Run completion: pass. Local foreground process exited 0 and `run.log` reports numeric `best_test_acc`.
- Startup SE configuration: pass. `run.log` reports `SE blocks: enabled, reduction=16`.
- Batch geometry: pass. `run.log` reports `Batches per epoch: 390`.
- LR drop: pass. `run.log` reports `step 21000 ep 54 ... lr: 0.0100`.
- Final metrics: pass. Summary metrics are present and `num_params` is `830,143`.
- Classification: no-improvement. `best_test_acc=93.71%` is below baseline `93.97%` and below the required improvement threshold `94.07%`.

### Informational Metrics
- SE added 7,353 parameters versus the 822,790-param anchor and completed 25,716 steps. Accuracy peaked at 93.71% on epoch 62, then ended at 93.65%.

## Errors & Dead Ends

## Human Notes

> Autopilot mode; no human approval or intervention requested during execution.
