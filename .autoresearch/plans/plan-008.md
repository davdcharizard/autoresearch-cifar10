# Plan EXP-008: Earlier First LR Drop Without Second Drop
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-008.md

## Milestones

### Milestone 1: Implement schedule-only change
- [x] Change the ResNet-20 scheduler milestones from `[32000, 48000]` to `[30000, 64000]`.
- [x] Preserve ResNet-20, batch size, optimizer, augmentation, seed, FP32 precision, cuDNN benchmark, channels-last, `torch.compile`, and once-per-epoch evaluation cadence.
- [x] Run syntax and lint checks, and confirm the diff only touches the planned scheduler line in `train.py`.

### Milestone 2: Launch local single-GPU run
- [x] Confirm a single NVIDIA H20 GPU is selected for execution.
- [x] Remove stale `run.log` if present.
- [x] Run `uv run train.py > run.log 2>&1` from the project root, with one GPU selected via `CUDA_VISIBLE_DEVICES`.
- [x] Monitor startup and early eval output for CUDA, compile, scheduler, or throughput errors.

### Milestone 3: Capture results and classify against the tightened threshold
- [x] Extract final summary metrics from `run.log`.
- [x] Compare `best_test_acc` to the current 91.95% baseline with the +0.10 point margin; EXP-008 needs `best_test_acc >= 92.05%`.
- [x] Confirm the first LR drop occurs at step 30,000 and the second LR drop is not reached.
- [x] Remove temporary run logs after the result is captured in the experiment log and report.

## Code Changes
- **train.py**: change the scheduler construction from `milestones=[32000, 48000]` to `milestones=[30000, 64000]`.
- This tests whether an earlier first LR drop gives more LR 0.01 refinement while avoiding the LR 0.001 phase that hurt EXP-003.
- No architecture, optimizer, augmentation, precision, batch size, seed, dependency, or evaluation-harness changes.

## Configuration Changes
- `MultiStepLR.milestones`: `[32000, 48000]` -> `[30000, 64000]`.
- Rationale: EXP-002 peaked after the first LR drop and never reached 48k; EXP-003 showed a reachable 40k second drop to LR 0.001 reduced accuracy. `[30000, 64000]` increases LR 0.01 exposure without entering LR 0.001 under the expected 43k-step budget.

## Execution Environment
- Method: local command from the project root: `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`. If GPU 0 is occupied by an unrelated run, choose another free physical GPU and record the adjustment in the exp-log before treating the run as a measurement.
- Resources: one visible NVIDIA H20 class GPU; no multi-GPU, no new packages, no dependency changes.
- Estimated runtime: roughly 6-7 minutes total, similar to EXP-002 and EXP-003; kill if it exceeds 10 minutes.
- Log output: all stdout/stderr redirected to `run.log`, with final metrics parsed from the summary block.
- Tool skill: none; this is a local run.

## Abort Criteria
- Abort and classify as crash if `run.log` shows a Python traceback, CUDA OOM, `torch.compile` failure that prevents training, or scheduler construction/runtime error.
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
   If the value is below `92.05`, classify EXP-008 as `no-improvement` even if it is above 91.95.
6. Schedule behavior review:
   inspect `run.log` for `step 30000` with `lr: 0.0100` and verify no `lr: 0.0010` lines appear.
7. Scope review:
   `git diff -- train.py`
   Pass condition: only the planned scheduler milestone change appears in `train.py`; `prepare.py`, dependency files, and evaluation harness are unchanged.
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
