# Plan EXP-007: Enable TF32 Throughput
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-007.md

## Milestones

### Milestone 1: Implement TF32 enablement
- [x] Add `USE_TF32 = True` near the existing throughput flags.
- [x] In CUDA setup, enable TF32 via `torch.set_float32_matmul_precision("high")`, `torch.backends.cuda.matmul.allow_tf32 = True`, and `torch.backends.cudnn.allow_tf32 = True`.
- [x] Preserve ResNet-20, batch size, optimizer, LR milestones `[32000, 48000]`, augmentation, seed, channels-last, compile, and evaluation cadence.
- [x] Run syntax and lint checks, and confirm the diff only touches `train.py`.

### Milestone 2: Launch local single-GPU run
- [x] Confirm a single NVIDIA H20 GPU is selected for execution.
- [x] Remove stale `run.log` if present.
- [x] Run `uv run train.py > run.log 2>&1` from the project root, with one GPU selected via `CUDA_VISIBLE_DEVICES`.
- [x] Monitor startup and early eval output for TF32, compile, CUDA, or throughput errors.

### Milestone 3: Capture results and classify against the tightened threshold
- [x] Extract final summary metrics from `run.log`.
- [x] Compare `best_test_acc` to the current 91.95% baseline with the +0.10 point margin; EXP-007 needs `best_test_acc >= 92.05%`.
- [x] Remove temporary run logs after the result is captured in the experiment log and report.

## Code Changes
- **train.py**: add `USE_TF32 = True` near `USE_CUDNN_BENCHMARK`, `USE_CHANNELS_LAST`, and `USE_COMPILE`.
- **train.py**: inside `if device.type == "cuda":`, keep `torch.backends.cudnn.benchmark = USE_CUDNN_BENCHMARK` and add a nested `if USE_TF32:` block that enables TF32 matmul and cuDNN behavior before model construction and `torch.compile`.
- No architecture, optimizer, LR schedule, precision autocast, batch size, data augmentation, seed, dependency, or evaluation-harness changes.

## Configuration Changes
- `USE_TF32`: `True`, testing the tensor-core acceleration path suggested by the repeated PyTorch warning.
- `torch.set_float32_matmul_precision("high")`: request TF32-capable internal matmul precision while keeping tensors and the training loop in the FP32 path.
- `torch.backends.cuda.matmul.allow_tf32 = True`: explicitly enable TF32 for CUDA matmul kernels; local check before planning showed this defaulted to `False`.
- `torch.backends.cudnn.allow_tf32 = True`: make cuDNN TF32 behavior explicit even though the local default was already `True`.

## Execution Environment
- Method: local command from the project root: `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`. If GPU 0 is occupied by an unrelated run, choose another free physical GPU and record the adjustment in the exp-log before treating the run as a measurement.
- Resources: one visible NVIDIA H20 class GPU; no multi-GPU, no new packages, no dependency changes.
- Estimated runtime: roughly 6-7 minutes total, similar to EXP-002 unless TF32 changes compile/runtime behavior; kill if it exceeds 10 minutes.
- Log output: all stdout/stderr redirected to `run.log`, with final metrics parsed from the summary block.
- Tool skill: none; this is a local run.

## Abort Criteria
- Abort and classify as crash if `run.log` shows a Python traceback, CUDA OOM, `torch.compile` failure that prevents training, or a TF32 API error.
- Abort if no progress/eval output appears after 3 minutes from launch.
- Kill the run if total wall-clock time exceeds 10 minutes.
- Treat the run as no-improvement, not crash, if it completes but `best_test_acc < 92.05%`.

## Verification Protocol

### Verification Procedure
1. Confirm the baseline and threshold:
   `bash /root/.codex/plugins/cache/deoxys/autoresearch/2.9.6/skills/shared/scripts/exp-index.sh baseline .autoresearch/experiment-indices/maximize-cifar10-best-test-accuracy.tsv`
   Pass condition: output reports `baseline=91.95`; threshold is `92.05`.
2. Confirm the TF32 control API and single-GPU selection:
   `CUDA_VISIBLE_DEVICES=0 uv run python -c 'import torch; print(hasattr(torch, "set_float32_matmul_precision")); print(torch.cuda.is_available()); print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")'`
   Pass condition: TF32 API exists, CUDA is available, visible device count is 1, and device name is NVIDIA H20.
3. Run the experiment:
   `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`
   Timeout: kill if total wall-clock time exceeds 10 minutes.
4. Check completion:
   `grep "^best_test_acc:\|^peak_vram_mb:" run.log`
   Pass condition: output includes a numeric `best_test_acc`.
5. Check the primary metric condition:
   parse `best_test_acc` from `run.log` and require `best_test_acc >= 92.05`.
   If the value is below `92.05`, classify EXP-007 as `no-improvement` even if it is above 91.95.
6. Scope review:
   `git diff -- train.py`
   Pass condition: only the planned TF32 flag and CUDA setup changes appear in `train.py`; `prepare.py`, dependency files, and evaluation harness are unchanged.
7. Validation cadence review:
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
