# EXP-028: Zero-Initialize Residual Branch Last BatchNorm

## Execution

Overall Status & Info:
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-028.md
- **Plan**: plans/plan-028.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-028
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed - no-improvement

## Implementation Notes

### Summary
Implemented the planned identity-preserving residual initialization in `train.py` only. `ResNet.__init__` now runs a short post-initialization loop after `self.apply(self._weights_init)` and sets every `BasicBlock` second BatchNorm scale, `bn2.weight`, to zero. The architecture, parameter count, batch size, optimizer, LR schedule, augmentation, FP32 compile/channels-last path, seed, and once-per-epoch validation were left unchanged.

### Surprises & Discoveries
The local `_weights_init` function only initializes convolution and linear weights; BatchNorm affine parameters were relying on PyTorch defaults. That made the change clean: a post-`self.apply(...)` loop can target only residual branch terminal BatchNorm scales without disturbing the rest of initialization.

### Decisions
Used a direct `isinstance(m, BasicBlock)` check inside `ResNet.__init__`, mirroring the relevant Torchvision pattern for `BasicBlock` residual branches. No new configuration flag was added because the experiment is meant to isolate one initialization variant and only `train.py` is in scope.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local shell PID 1828170; uv PID 1828171; main Python PID 1828174
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-08 19:34 UTC
- **Ended**: 2026-06-08 19:40 UTC

Description:
- Runs the 28/56/112 ResNet-20 anchor with only the residual branch final BatchNorm scale zero-initialized. The run should preserve the fixed 300s training budget, batch size 128, once-per-epoch validation, and step-21000 first LR drop. The expected success condition is `best_test_acc >= 93.33%`, reflecting the current 93.23% baseline plus the goal's +0.10 percentage-point noise margin.

Observations:
- Startup is clean: CUDA is selected, the model is `ResNet-20` with `822,790` parameters, the fixed time budget is `300s`, and batch size is preserved with `Batches per epoch: 390`. (source: run.log L1-L4)
- Early training is healthy with no traceback/OOM/NaN/Inf patterns found; by epoch 14, best test accuracy reached `80.77%` while GPU 0 was active. (source: run.log L6-L32; `rg` error scan on run.log; `nvidia-smi` during Run 1)
- The first LR drop fired as planned at step 21000 with `lr: 0.0100`; the first post-drop epoch 54 evaluation reached `90.12%` best accuracy. (source: run.log L111-L112)
- Late post-drop refinement plateaued below the baseline range by epoch 72, with best accuracy at `91.74%`. (source: run.log L112-L148)
- The run completed normally with exit code 0 and printed summary metrics. Final `best_test_acc=91.74%`, below the 93.23% baseline and the 93.33% improvement threshold, so EXP-028 is no-improvement. (source: run.log L226-L235)

Key Metrics:
- `best_test_acc`: 91.74%
- `final_test_acc`: 91.48%
- `final_test_loss`: 0.3972
- `training_seconds`: 300.0
- `total_seconds`: 389.9
- `startup_seconds`: 1.9
- `peak_vram_mb`: 660.4
- `num_epochs`: 110
- `num_steps`: 42634
- `num_params`: 822,790

## Verification Results

### Conditions Checked
- Baseline and threshold: PASS. Baseline query reported `baseline=93.23`; EXP-028 threshold is `93.33%`. (source: exp-index.sh baseline output)
- Scope before launch/result: PASS. `git diff --name-only` reported only `train.py`; `git status --short --branch` showed branch `autoresearch/exp-028`, modified `train.py`, and untracked `data/`.
- Syntax and lint: PASS. `python3 -m py_compile train.py` exited 0 and `uv run ruff check train.py` reported all checks passed before launch.
- Validation cadence: PASS. `rg -n "evaluator\\.evaluate|Eval\\(" train.py` reported one `Eval()` construction and one `evaluator.evaluate(...)` call.
- Residual zero-init implementation: PASS. `rg -n "BasicBlock|bn2\\.weight|init\\.constant_" train.py` showed the planned `BasicBlock` loop and `init.constant_(m.bn2.weight, 0)`.
- Preserved batch size, schedule, and parameter count: PASS. `Batches per epoch: 390` confirmed `BATCH_SIZE=128`, `step 21000` showed `lr: 0.0100`, and `num_params=822,790`.
- Experiment completion: PASS. The process exited 0, printed numeric summary metrics, and `total_seconds=389.9`, under the 10-minute wall-clock cap.
- Metric improvement: FAIL. `best_test_acc=91.74%`, below the required `93.33%` threshold. Under the goal's +0.10 percentage-point rule, this is no-improvement.
- Hard constraints: PASS. Only the planned initialization diff was present during the run, `training_seconds=300.0`, `total_seconds=389.9`, and no protected files changed.

### Informational Metrics
- `final_test_acc`: 91.48%
- `final_test_loss`: 0.3972
- `training_seconds`: 300.0
- `total_seconds`: 389.9
- `startup_seconds`: 1.9
- `peak_vram_mb`: 660.4
- `num_epochs`: 110
- `num_steps`: 42634
- `num_params`: 822,790

## Errors & Dead Ends

## Human Notes

> No human intervention during autopilot execution.

<!-- NOTE: Human notes are high trust and privileged relative to other info in this document. -->
