# Plan EXP-014: ResNet-20 Width 1.75x with 22k First Drop
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-014.md

## Milestones

### Milestone 1: Implement width and schedule calibration
- [x] Change `STAGE_WIDTHS` in `train.py` from `(24, 48, 96)` to `(28, 56, 112)`.
- [x] Change `LR_MILESTONES` in `train.py` from `[24000, 64000]` to `[22000, 64000]`.
- [x] Preserve `NUM_BLOCKS = 3`, batch size, optimizer, augmentation, seed, FP32 precision, cuDNN benchmark, channels-last, `torch.compile`, and once-per-epoch evaluation.
- [x] Run syntax/lint checks and confirm the diff only touches the planned width and scheduler constants in `train.py`.

### Milestone 2: Launch local single-GPU run
- [x] Confirm a single NVIDIA H20 GPU is selected for execution.
- [x] Remove stale `run.log` if present.
- [x] Run `uv run train.py > run.log 2>&1` from the project root with one GPU selected via `CUDA_VISIBLE_DEVICES`.
- [x] Monitor startup and early eval output for CUDA, compile, OOM, or severe throughput errors.

### Milestone 3: Capture results and classify against tightened threshold
- [x] Extract final summary metrics from `run.log`.
- [x] Compare `best_test_acc` to the current 92.49% baseline with the +0.10 point margin; EXP-014 needs `best_test_acc >= 92.59%`.
- [x] Check whether the run reaches the planned first LR drop at 22k steps and retains enough LR 0.01 refinement time.
- [x] Remove temporary run logs after the result is captured in the experiment log and report.

## Code Changes
- **train.py**: change `STAGE_WIDTHS = (24, 48, 96)` to `STAGE_WIDTHS = (28, 56, 112)`.
- **train.py**: change `LR_MILESTONES = [24000, 64000]` to `LR_MILESTONES = [22000, 64000]`.
- No depth, optimizer, loss, augmentation, precision, batch size, seed, dependency, or evaluation-harness changes.

## Configuration Changes
- `STAGE_WIDTHS`: `(24, 48, 96)` -> `(28, 56, 112)`. Rationale: EXP-011 and EXP-013 both validated width scaling; the new threshold likely needs another capacity step.
- `LR_MILESTONES`: `[24000, 64000]` -> `[22000, 64000]`. Rationale: the wider model is expected to complete fewer steps than EXP-013, so the first drop is moved earlier to preserve LR 0.01 refinement time. The second milestone remains unreachable to avoid the harmful LR 0.001 phase.

## Execution Environment
- Method: local command from the project root: `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`. If physical GPU 0 is occupied by an unrelated run, choose another free physical GPU and record the adjustment in the exp-log before treating the run as a measurement.
- Resources: one visible NVIDIA H20 class GPU; no multi-GPU, no new packages, no dependency changes.
- Estimated runtime: roughly 6-8 minutes total; kill if total wall-clock exceeds 10 minutes.
- Log output: all stdout/stderr redirected to `run.log`, with final metrics parsed from the summary block.
- Tool skill: none; this is a local run.

## Abort Criteria
- Abort and classify as crash if `run.log` shows a Python traceback, CUDA OOM, or `torch.compile` failure that prevents training.
- Abort if no progress/eval output appears after 3 minutes from launch.
- Kill the run if total wall-clock time exceeds 10 minutes.
- Treat the run as no-improvement, not crash, if it completes but `best_test_acc < 92.59%`.

## Verification Protocol

### Verification Procedure
1. Confirm the baseline and threshold:
   `bash /root/.codex/plugins/cache/deoxys/autoresearch/2.9.6/skills/shared/scripts/exp-index.sh baseline .autoresearch/experiment-indices/maximize-cifar10-best-test-accuracy.tsv`
   Pass condition: output reports `baseline=92.49`; threshold is `92.59`.
2. Confirm single-GPU CUDA execution:
   `CUDA_VISIBLE_DEVICES=0 uv run python -c 'import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")'`
   Pass condition: CUDA is available, visible device count is 1, and device name is NVIDIA H20. If GPU 0 is occupied, rerun this check with the chosen free physical GPU.
3. Run the experiment:
   `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`
   Timeout: kill if total wall-clock time exceeds 10 minutes.
4. Check completion:
   `grep "^best_test_acc:\|^peak_vram_mb:" run.log`
   Pass condition: output includes a numeric `best_test_acc`.
5. Check the primary metric condition:
   parse `best_test_acc` from `run.log` and require `best_test_acc >= 92.59`.
   If the value is below `92.59`, classify EXP-014 as `no-improvement` even if it is above 92.49.
6. Schedule/throughput sanity review:
   inspect `run.log` for `lr: 0.0100`, `num_steps`, `num_epochs`, and `num_params`. The run should reach the first LR drop at step 22000; if it never reaches 22k, analysis should classify the result as undertrained or schedule-missed.
7. Scope review:
   `git diff -- train.py`
   Pass condition: only the planned `STAGE_WIDTHS` and `LR_MILESTONES` changes appear in `train.py`; `prepare.py`, dependency files, and evaluation harness are unchanged.
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
