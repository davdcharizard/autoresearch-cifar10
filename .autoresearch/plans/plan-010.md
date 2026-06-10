# Plan EXP-010: Isolated Nesterov Momentum
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-010.md

## Milestones

### Milestone 1: Implement optimizer-only Nesterov flag
- [x] Add an explicit `USE_NESTEROV = True` hyperparameter near the existing optimizer constants in `train.py`.
- [x] Pass `nesterov=USE_NESTEROV` into the existing `optim.SGD(...)` call.
- [x] Preserve ResNet-20, batch size, LR, momentum value, weight decay, LR milestones `[32000, 48000]`, seed, augmentation, FP32 precision, cuDNN benchmark, channels-last, `torch.compile`, and once-per-epoch evaluation.
- [x] Run syntax/lint checks and confirm the diff only touches the planned optimizer flag and SGD argument in `train.py`.

### Milestone 2: Launch local single-GPU run
- [x] Confirm a single NVIDIA H20 GPU is selected for execution.
- [x] Remove stale `run.log` if present.
- [x] Run `uv run train.py > run.log 2>&1` from the project root, with one GPU selected via `CUDA_VISIBLE_DEVICES`.
- [x] Monitor startup and early eval output for optimizer, CUDA, compile, or throughput errors.

### Milestone 3: Capture results and classify against the tightened threshold
- [x] Extract final summary metrics from `run.log`.
- [x] Compare `best_test_acc` to the current 91.95% baseline with the +0.10 point margin; EXP-010 needs `best_test_acc >= 92.05%`.
- [x] Check that throughput/step count remains in the EXP-002/009 FP32 baseline range rather than showing unexpected optimizer overhead.
- [x] Remove temporary run logs after the result is captured in the experiment log and report.

## Code Changes
- **train.py**: add `USE_NESTEROV = True` near the hyperparameter block.
- **train.py**: change the existing SGD construction from `optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)` to include `nesterov=USE_NESTEROV`.
- No model, transform, loss, LR schedule, precision, batch size, seed, dependency, or evaluation-harness changes.

## Configuration Changes
- `USE_NESTEROV`: absent -> `True`.
- All other optimizer settings remain unchanged: `LR=0.1`, `MOMENTUM=0.9`, `WEIGHT_DECAY=1e-4`, milestones `[32000, 48000]`.
- Rationale: Nesterov was only tested inside EXP-000's confounded bundle; this isolates the optimizer flag while preserving the successful EXP-002 FP32 throughput recipe.

## Execution Environment
- Method: local command from the project root: `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`. If GPU 0 is occupied by an unrelated run, choose another free physical GPU and record the adjustment in the exp-log before treating the run as a measurement.
- Resources: one visible NVIDIA H20 class GPU; no multi-GPU, no new packages, no dependency changes.
- Estimated runtime: roughly 6-7 minutes total, similar to EXP-002/008/009; kill if total wall-clock exceeds 10 minutes.
- Log output: all stdout/stderr redirected to `run.log`, with final metrics parsed from the summary block.
- Tool skill: none; this is a local run.

## Abort Criteria
- Abort and classify as crash if `run.log` shows a Python traceback, optimizer configuration error, CUDA OOM, or `torch.compile` failure that prevents training.
- Abort if no progress/eval output appears after 3 minutes from launch.
- Kill the run if total wall-clock time exceeds 10 minutes.
- Treat the run as no-improvement, not crash, if it completes but `best_test_acc < 92.05%`.

## Verification Protocol

### Verification Procedure
1. Confirm the baseline and threshold:
   `bash /root/.codex/plugins/cache/deoxys/autoresearch/2.9.6/skills/shared/scripts/exp-index.sh baseline .autoresearch/experiment-indices/maximize-cifar10-best-test-accuracy.tsv`
   Pass condition: output reports `baseline=91.95`; threshold is `92.05`.
2. Confirm single-GPU CUDA execution:
   `CUDA_VISIBLE_DEVICES=0 uv run python -c 'import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")'`
   Pass condition: CUDA is available, visible device count is 1, and device name is NVIDIA H20.
3. Run the experiment:
   `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`
   Timeout: kill if total wall-clock time exceeds 10 minutes.
4. Check completion:
   `grep "^best_test_acc:\|^peak_vram_mb:" run.log`
   Pass condition: output includes a numeric `best_test_acc`.
5. Check the primary metric condition:
   parse `best_test_acc` from `run.log` and require `best_test_acc >= 92.05`.
   If the value is below `92.05`, classify EXP-010 as `no-improvement` even if it is above 91.95.
6. Throughput sanity review:
   compare `num_steps` and `num_epochs` from the summary to EXP-002/008/009 ranges; large unexpected drops indicate optimizer overhead or compile behavior should be highlighted in analysis.
7. Scope review:
   `git diff -- train.py`
   Pass condition: only the planned `USE_NESTEROV` constant and SGD `nesterov` argument appear in `train.py`; `prepare.py`, dependency files, and evaluation harness are unchanged.
8. Validation cadence review:
   inspect `train.py` and `run.log` to confirm exactly one `evaluator.evaluate(...)` call per epoch and no additional validation loop.

### Informational Metrics (Optional)
- final_test_acc: final summary line `final_test_acc:` in `run.log`.
- final_test_loss: final summary line `final_test_loss:` in `run.log`.
- training_seconds: final summary line `training_seconds:` in `run.log`.
- total_seconds: final summary line `total_seconds:` in `run.log`.
- startup_seconds: final summary line `startup_seconds:` in `run.log`.
- peak_vram_mb: final summary line `peak_vram_mb:` in `run.log`.
- num_epochs: final summary line `num_epochs:` in `run.log`.
- num_steps: final summary line `num_steps:` in `run.log`.
- num_params: final summary line `num_params:` in `run.log`.
