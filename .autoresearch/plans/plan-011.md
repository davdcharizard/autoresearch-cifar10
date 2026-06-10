# Plan EXP-011: ResNet-20 Width 1.25x
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-011.md

## Milestones

### Milestone 1: Implement modest width increase and calibrated milestone
- [x] Add `STAGE_WIDTHS = (20, 40, 80)` near the hyperparameter block in `train.py`.
- [x] Add `LR_MILESTONES = [24000, 64000]` near the hyperparameter block and wire the scheduler to it.
- [x] Modify `ResNet.__init__` to use `STAGE_WIDTHS` for `conv1`, `bn1`, `layer1`, `layer2`, `layer3`, and `fc`.
- [x] Preserve `NUM_BLOCKS = 3`, batch size, optimizer, augmentation, seed, FP32 precision, cuDNN benchmark, channels-last, `torch.compile`, and once-per-epoch evaluation.
- [x] Run syntax/lint checks and confirm the diff only touches planned width constants, `ResNet` channel wiring, and scheduler milestones in `train.py`.

### Milestone 2: Launch local single-GPU run
- [x] Confirm a single NVIDIA H20 GPU is selected for execution.
- [x] Remove stale `run.log` if present.
- [x] Run `uv run train.py > run.log 2>&1` from the project root, with one GPU selected via `CUDA_VISIBLE_DEVICES`.
- [x] Monitor startup and early eval output for model-shape, compile, CUDA, or throughput errors.

### Milestone 3: Capture results and classify against the tightened threshold
- [x] Extract final summary metrics from `run.log`.
- [x] Compare `best_test_acc` to the current 91.95% baseline with the +0.10 point margin; EXP-011 needs `best_test_acc >= 92.05%`.
- [x] Check whether the run reaches the planned first LR drop at 24k steps and whether step count is high enough for meaningful LR 0.01 refinement.
- [x] Remove temporary run logs after the result is captured in the experiment log and report.

## Code Changes
- **train.py**: add `STAGE_WIDTHS = (20, 40, 80)` as the explicit 1.25x channel schedule, replacing hardcoded 16/32/64 values inside `ResNet.__init__`.
- **train.py**: add `LR_MILESTONES = [24000, 64000]` and replace the scheduler's inline `[32000, 48000]` with `LR_MILESTONES`.
- **train.py**: update `conv1`, `bn1`, `layer1`, `layer2`, `layer3`, and `fc` to use `(w1, w2, w3) = STAGE_WIDTHS`.
- No depth, optimizer, loss, augmentation, precision, batch size, seed, dependency, or evaluation-harness changes.

## Configuration Changes
- `STAGE_WIDTHS`: absent -> `(20, 40, 80)`, a 1.25x increase over the baseline `(16, 32, 64)`.
- Scheduler milestones: `[32000, 48000]` -> `[24000, 64000]`.
- Rationale: EXP-006 showed deeper ResNet-32 missed a 26k first LR drop at 23,642 steps; a modest width increase should be cheaper than depth, and a 24k first drop gives the wider model a chance to enter LR 0.01 refinement even if it completes fewer steps than ResNet-20. The second milestone is intentionally unreachable to avoid the harmful LR 0.001 phase seen in EXP-003.

## Execution Environment
- Method: local command from the project root: `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`. If GPU 0 is occupied by an unrelated run, choose another free physical GPU and record the adjustment in the exp-log before treating the run as a measurement.
- Resources: one visible NVIDIA H20 class GPU; no multi-GPU, no new packages, no dependency changes.
- Estimated runtime: roughly 6-8 minutes total; kill if total wall-clock exceeds 10 minutes.
- Log output: all stdout/stderr redirected to `run.log`, with final metrics parsed from the summary block.
- Tool skill: none; this is a local run.

## Abort Criteria
- Abort and classify as crash if `run.log` shows a Python traceback, tensor/channel shape mismatch, CUDA OOM, or `torch.compile` failure that prevents training.
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
   If the value is below `92.05`, classify EXP-011 as `no-improvement` even if it is above 91.95.
6. Schedule/throughput sanity review:
   inspect `run.log` for `lr: 0.0100`, `num_steps`, `num_epochs`, and `num_params`. The first LR drop should occur at step 24000; if the run never reaches 24k, analysis should classify the result as undertrained capacity rather than a clean accuracy ceiling.
7. Scope review:
   `git diff -- train.py`
   Pass condition: only planned width constants, `ResNet` channel wiring, and scheduler milestone changes appear in `train.py`; `prepare.py`, dependency files, and evaluation harness are unchanged.
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
