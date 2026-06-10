# Plan EXP-006: Schedule-Calibrated ResNet-32
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-006.md

## Milestones

### Milestone 1: Implement ResNet-32 capacity change
- [x] Change `NUM_BLOCKS` from `3` to `5`, making the existing CIFAR ResNet implementation instantiate ResNet-32.
- [x] Add explicit LR milestone constants near the training hyperparameters.
- [x] Change `MultiStepLR` to use the new calibrated milestones `[26000, 39000]`.
- [x] Run syntax and lint checks, and confirm the diff only touches `train.py`.

### Milestone 2: Launch local single-GPU run
- [x] Confirm a single NVIDIA H20 GPU is selected for execution.
- [x] Remove stale `run.log` if present.
- [x] Run `uv run train.py > run.log 2>&1` from the project root, with one GPU selected via `CUDA_VISIBLE_DEVICES`.
- [x] Monitor startup and early eval output for architecture, compile, CUDA, or schedule errors.

### Milestone 3: Capture results and classify against the tightened threshold
- [x] Extract final summary metrics from `run.log`.
- [x] Compare `best_test_acc` to the current 91.95% baseline with the +0.10 point margin; EXP-006 needs `best_test_acc >= 92.05%`.
- [x] Remove temporary run logs after the result is captured in the experiment log and report.

## Code Changes
- **train.py**: change `NUM_BLOCKS = 3` to `NUM_BLOCKS = 5`, which reuses the existing CIFAR ResNet implementation and changes the printed model from ResNet-20 to ResNet-32.
- **train.py**: add `LR_MILESTONES = [26000, 39000]` near the other top-level hyperparameters.
- **train.py**: change `optim.lr_scheduler.MultiStepLR(..., milestones=[32000, 48000], gamma=0.1)` to use `milestones=LR_MILESTONES`.
- No changes to `prepare.py`, `Eval.evaluate`, dependencies, data loading semantics, seed, optimizer type, batch size, augmentation, loss, FP32 arithmetic, channels-last, `torch.compile`, or evaluation cadence.

## Configuration Changes
- `NUM_BLOCKS`: `5`, testing a modest capacity increase from ResNet-20 to ResNet-32 while preserving the local architecture pattern.
- `LR_MILESTONES`: `[26000, 39000]`.
  - Rationale: ResNet-32 is expected to run fewer optimizer steps than ResNet-20, so the first drop should occur earlier than 32k to preserve low-LR refinement.
  - The second milestone is set late enough to avoid repeating EXP-003's too-early 40k drop on the faster ResNet-20, while still allowing a short LR 0.001 phase if the larger model reaches roughly 39k steps.
- All other settings remain inherited from the EXP-002 FP32 throughput baseline.

## Execution Environment
- Method: local command from the project root: `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`. If GPU 0 is occupied by an unrelated run, choose another free physical GPU and record the adjustment in the exp-log before treating the run as a measurement.
- Resources: one visible NVIDIA H20 class GPU; no multi-GPU, no new packages, no dependency changes.
- Estimated runtime: roughly 7-9 minutes total because compilation and validation may be slower for ResNet-32; kill if it exceeds 10 minutes.
- Log output: all stdout/stderr redirected to `run.log`, with final metrics parsed from the summary block.
- Tool skill: none; this is a local run.

## Abort Criteria
- Abort and classify as crash if `run.log` shows a Python traceback, CUDA OOM, `torch.compile` failure that prevents training, or a model shape error.
- Abort if no progress/eval output appears after 3 minutes from launch.
- Kill the run if total wall-clock time exceeds 10 minutes.
- Treat the run as no-improvement, not crash, if it completes but `best_test_acc < 92.05%`.

## Verification Protocol

### Verification Procedure
1. Confirm the baseline and threshold:
   `bash /root/.codex/plugins/cache/deoxys/autoresearch/2.9.6/skills/shared/scripts/exp-index.sh baseline .autoresearch/experiment-indices/maximize-cifar10-best-test-accuracy.tsv`
   Pass condition: output reports `baseline=91.95`; threshold is `92.05`.
2. Confirm a single GPU is selected:
   `CUDA_VISIBLE_DEVICES=0 uv run python - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY`
   Pass condition: CUDA is available, visible device count is 1, and device name is NVIDIA H20.
3. Run the experiment:
   `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`
   Timeout: kill if total wall-clock time exceeds 10 minutes.
4. Check completion:
   `grep "^best_test_acc:\|^peak_vram_mb:" run.log`
   Pass condition: output includes a numeric `best_test_acc`.
5. Check the primary metric condition:
   parse `best_test_acc` from `run.log` and require `best_test_acc >= 92.05`.
   If the value is below `92.05`, classify EXP-006 as `no-improvement` even if it is above 91.95.
6. Scope review:
   `git diff -- train.py`
   Pass condition: only the planned `NUM_BLOCKS`, `LR_MILESTONES`, and scheduler milestone changes appear in `train.py`; `prepare.py`, dependency files, and evaluation harness are unchanged.
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
