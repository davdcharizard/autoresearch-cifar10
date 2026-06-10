# Plan EXP-009: Weak 8x8 Cutout
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-009.md

## Milestones

### Milestone 1: Implement weak cutout
- [x] Add explicit cutout constants for an 8x8 fixed mask: `USE_CUTOUT = True`, `CUTOUT_PROB = 0.25`, `CUTOUT_AREA = 0.0625`, and `CUTOUT_RATIO = 1.0`.
- [x] Insert `transforms.RandomErasing` after normalization in the training transform, matching EXP-005 placement but with weaker area/probability.
- [x] Preserve ResNet-20, batch size, optimizer, LR milestones `[32000, 48000]`, seed, FP32 precision, cuDNN benchmark, channels-last, `torch.compile`, and once-per-epoch evaluation cadence.
- [x] Run syntax and lint checks, and confirm the diff only touches the planned cutout constants and training transform in `train.py`.

### Milestone 2: Launch local single-GPU run
- [x] Confirm a single NVIDIA H20 GPU is selected for execution.
- [x] Remove stale `run.log` if present.
- [x] Run `uv run train.py > run.log 2>&1` from the project root, with one GPU selected via `CUDA_VISIBLE_DEVICES`.
- [x] Monitor startup and early eval output for transform, CUDA, compile, or throughput errors.

### Milestone 3: Capture results and classify against the tightened threshold
- [x] Extract final summary metrics from `run.log`.
- [x] Compare `best_test_acc` to the current 91.95% baseline with the +0.10 point margin; EXP-009 needs `best_test_acc >= 92.05%`.
- [x] Check that throughput/step count remains in the FP32 baseline range rather than showing major transform overhead.
- [x] Remove temporary run logs after the result is captured in the experiment log and report.

## Code Changes
- **train.py**: add constants near the existing hyperparameters:
  - `USE_CUTOUT = True`
  - `CUTOUT_PROB = 0.25`
  - `CUTOUT_AREA = 0.0625`
  - `CUTOUT_RATIO = 1.0`
- **train.py**: add `transforms.RandomErasing(p=CUTOUT_PROB, scale=(CUTOUT_AREA, CUTOUT_AREA), ratio=(CUTOUT_RATIO, CUTOUT_RATIO), value=0)` immediately after `transforms.Normalize(mean, std)` in `train_tf` when `USE_CUTOUT` is true.
- The fixed area `0.0625` corresponds to an 8x8 mask on a 32x32 CIFAR image; fixed ratio `1.0` makes it square. `value=0` after normalization produces a mean-valued erased region in normalized input space.
- No architecture, optimizer, LR schedule, precision, batch size, seed, dependency, or evaluation-harness changes.

## Configuration Changes
- `USE_CUTOUT`: absent -> `True`.
- `CUTOUT_PROB`: absent -> `0.25`, lower than the prior full-strength cutout attempt.
- `CUTOUT_AREA`: absent -> `0.0625`, one quarter of EXP-005's 16x16 mask area.
- `CUTOUT_RATIO`: absent -> `1.0`, fixed square mask.
- Rationale: EXP-005 showed 16x16 cutout preserved throughput but over-regularized; this tests whether much weaker masking can recover useful generalization without the same convergence delay.

## Execution Environment
- Method: local command from the project root: `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`. If GPU 0 is occupied by an unrelated run, choose another free physical GPU and record the adjustment in the exp-log before treating the run as a measurement.
- Resources: one visible NVIDIA H20 class GPU; no multi-GPU, no new packages, no dependency changes.
- Estimated runtime: roughly 6-7 minutes total, similar to EXP-002/005/008; kill if it exceeds 10 minutes.
- Log output: all stdout/stderr redirected to `run.log`, with final metrics parsed from the summary block.
- Tool skill: none; this is a local run.

## Abort Criteria
- Abort and classify as crash if `run.log` shows a Python traceback, transform/value error, CUDA OOM, or `torch.compile` failure that prevents training.
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
   If the value is below `92.05`, classify EXP-009 as `no-improvement` even if it is above 91.95.
6. Throughput sanity review:
   compare `num_steps` and `num_epochs` from the summary to EXP-002/005 ranges; large unexpected drops indicate transform overhead should be highlighted in analysis.
7. Scope review:
   `git diff -- train.py`
   Pass condition: only the planned cutout constants and transform change appear in `train.py`; `prepare.py`, dependency files, and evaluation harness are unchanged.
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
