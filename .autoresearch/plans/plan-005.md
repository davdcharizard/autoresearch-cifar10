# Plan EXP-005: Isolated Cutout
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-005.md

## Milestones

### Milestone 1: Implement cutout-only augmentation
- [x] Add explicit cutout hyperparameters near the existing training hyperparameters.
- [x] Add `transforms.RandomErasing` to the training transform only, preserving model, optimizer, LR milestones, batch size, FP32 compile, channels-last, seed, and evaluation cadence.
- [x] Run syntax and lint checks, and confirm the diff only touches `train.py`.

### Milestone 2: Launch local single-GPU run
- [x] Confirm a single NVIDIA H20 GPU is selected for execution.
- [x] Remove stale `run.log` if present.
- [x] Run `uv run train.py > run.log 2>&1` from the project root, with `CUDA_VISIBLE_DEVICES=0` if multiple GPUs are visible.
- [x] Monitor startup and early eval output for transform or CUDA errors.

### Milestone 3: Capture results and classify against the tightened threshold
- [x] Extract final summary metrics from `run.log`.
- [x] Compare `best_test_acc` to the current 91.95% baseline with the +0.10 point margin; EXP-005 needs `best_test_acc >= 92.05%`.
- [x] Remove temporary run logs after the result is captured in the experiment log and report.

## Code Changes
- **train.py**: add `USE_CUTOUT = True`, `CUTOUT_PROB = 0.5`, `CUTOUT_SIZE = 16`, and derived area fraction notes near the hyperparameters.
- **train.py**: insert `transforms.RandomErasing(p=CUTOUT_PROB, scale=(0.25, 0.25), ratio=(1.0, 1.0), value=0.0)` into `train_tf` after `transforms.Normalize(mean, std)`. Applying it after normalization makes the erased patch exactly zero in model input space, matching the usual cutout convention.
- **train.py**: leave all other training settings unchanged from EXP-002: ResNet-20, SGD, LR `0.1`, milestones `[32000, 48000]`, batch size 128, FP32 arithmetic, channels-last, `torch.compile`, and once-per-epoch evaluation.

## Configuration Changes
- `USE_CUTOUT`: new `True` flag to make the intervention explicit and reversible.
- `CUTOUT_PROB`: `0.5`, matching the previously tested cutout component while isolating it from label smoothing, Nesterov, and cosine LR.
- `CUTOUT_SIZE`: `16`, corresponding to area fraction `16*16 / (32*32) = 0.25`.
- No architecture, optimizer, LR schedule, precision, batch size, seed, data path, dependency, or evaluation-harness changes.

## Execution Environment
- Method: local command from the project root: `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`.
- Resources: one visible NVIDIA H20 class GPU; no multi-GPU, no new packages, no dependency changes.
- Estimated runtime: roughly 6-7 minutes total, similar to EXP-002 unless CPU augmentation overhead is material; kill if it exceeds 10 minutes.
- Log output: all stdout/stderr redirected to `run.log`, with final metrics parsed from the summary block.
- Tool skill: none; this is a local run.

## Abort Criteria
- Abort and classify as crash if `run.log` shows a Python traceback, transform API error, CUDA OOM, or `torch.compile` failure that prevents training.
- Abort if no progress/eval output appears after 3 minutes from launch.
- Kill the run if total wall-clock time exceeds 10 minutes.
- Treat the run as no-improvement, not crash, if it completes but `best_test_acc < 92.05%`.

## Verification Protocol

### Verification Procedure
1. Confirm the baseline and threshold:
   `bash /root/.codex/plugins/cache/deoxys/autoresearch/2.9.6/skills/shared/scripts/exp-index.sh baseline .autoresearch/experiment-indices/maximize-cifar10-best-test-accuracy.tsv`
   Pass condition: output reports `baseline=91.95`; threshold is `92.05`.
2. Confirm the project command and transform API are available:
   `uv run python - <<'PY'
from torchvision import transforms
print(transforms.RandomErasing)
PY`
   Pass condition: command exits 0 and prints the `RandomErasing` class.
3. Confirm a single GPU is selected:
   `CUDA_VISIBLE_DEVICES=0 uv run python - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY`
   Pass condition: CUDA is available, visible device count is 1, and device name is NVIDIA H20.
4. Run the experiment:
   `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`
   Timeout: kill if total wall-clock time exceeds 10 minutes.
5. Check completion:
   `grep "^best_test_acc:\|^peak_vram_mb:" run.log`
   Pass condition: output includes a numeric `best_test_acc`.
6. Check the primary metric condition:
   parse `best_test_acc` from `run.log` and require `best_test_acc >= 92.05`.
   If the value is below `92.05`, classify EXP-005 as `no-improvement` even if it is above 91.95.
7. Scope review:
   `git diff -- train.py`
   Pass condition: only the planned cutout transform and constants changed in `train.py`; `prepare.py`, dependency files, and evaluation harness are unchanged.
8. Validation cadence review:
   inspect `train.py` and `run.log` to confirm exactly one `evaluator.evaluate(...)` call per epoch and no additional validation loop.

### Informational Metrics (Optional)
- final_test_acc: final summary line `final_test_acc:` in `run.log`.
- final_test_loss: final summary line `final_test_loss:` in `run.log`.
- training_seconds: final summary line `training_seconds:` in `run.log`.
- total_seconds: final summary line `total_seconds:` in `run.log`.
- peak_vram_mb: final summary line `peak_vram_mb:` in `run.log`.
- num_epochs: final summary line `num_epochs:` in `run.log`.
- num_steps: final summary line `num_steps:` in `run.log`.
- num_params: final summary line `num_params:` in `run.log`.
