# Plan EXP-046: Time-Budget-Matched Cosine Schedule
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-046.md

## Milestones

### Milestone 1: Code Change and Preflight
- [x] Modify `train.py` only: replace step-milestone scheduling with elapsed-time no-restart cosine.
- [x] Verify the diff preserves model, optimizer, augmentation, label smoothing, batch size, and weight decay anchors.
- [x] Run `python3 -m py_compile train.py`.
- [x] Run `uv run ruff check train.py` if the project environment exposes ruff.

### Milestone 2: Local Experiment Run
- [x] Confirm the experiment branch is based on `autoresearch/dev` at baseline commit `755be2c`.
- [x] Remove any stale `run.log`.
- [x] Select one GPU with `CUDA_VISIBLE_DEVICES=<id>`.
- [x] Run `env CUDA_VISIBLE_DEVICES=<id> uv run train.py > run.log 2>&1` in a foreground exec session.
- [x] Monitor the log for startup, LR decay, epoch evaluations, and final metric output.

### Milestone 3: Result Capture
- [x] Extract `best_test_acc`, `final_test_acc`, `final_test_loss`, `training_seconds`, `total_seconds`, `peak_vram_mb`, `num_epochs`, `num_steps`, and `num_params` from `run.log`.
- [x] Classify the result against baseline `93.97%` and improvement threshold `94.07%`.
- [x] Record final metrics in `logs/exp-log-046.md` for analysis.

## Code Changes
- **`train.py`**: Add `math` import and a small `lr_at_fraction(frac)` helper that returns `LR * 0.5 * (1 + cos(pi * frac))`, clamped to `frac in [0, 1]`.
- **`train.py`**: Remove the `LR_MILESTONES`/`MultiStepLR` path and set each optimizer param group's LR once per training step from `total_training_time / TIME_BUDGET_S` before the forward/backward pass.
- **`train.py`**: Keep all anchor settings unchanged: `STAGE_WIDTHS=(28, 56, 112)`, `BATCH_SIZE=128`, `LR=0.1`, `MOMENTUM=0.9`, `WEIGHT_DECAY=2e-4`, reflection crop padding, `label_smoothing=0.05`, channels-last, and compile.

## Configuration Changes
- `LR_MILESTONES=[21000, 64000]` -> removed from active scheduling path. Rationale: test smooth no-restart cosine against the current abrupt step schedule.
- Scheduler: `MultiStepLR(..., gamma=0.1)` -> elapsed-time cosine decay from `LR=0.1` to approximately zero across `TIME_BUDGET_S`. Rationale: the benchmark is wall-clock limited, so the schedule should complete regardless of variable realized step count.
- `MAX_STEPS=64000` remains unchanged. Rationale: time budget remains the effective limiter; this avoids changing the benchmark budget.

## Execution Environment
- Method: local foreground run from the project root.
- Resources: one NVIDIA H20-class GPU, selected via `CUDA_VISIBLE_DEVICES`.
- Estimated runtime: about 6-8 minutes wall clock including validation and startup; kill if total runtime exceeds 10 minutes.
- Log output: capture all stdout/stderr to `run.log`; final analysis reads metrics from that file.
- Tool skill: none; local execution only.

## Abort Criteria
- Stop and classify as crash if `run.log` shows a Python traceback, CUDA OOM, dataset/download failure, or compile failure.
- Stop if no training progress line or eval appears within 3 minutes of process start.
- Stop if wall-clock runtime exceeds 10 minutes total.
- Stop if LR logging shows a constant `0.1000` throughout the run, because that would mean the schedule change did not execute.
- Do not abort merely because early pre-drop accuracy is lower than the step baseline; cosine intentionally trades early plateau behavior for smoother late refinement.

## Verification Protocol

### Verification Procedure
1. Confirm the active baseline with:
   `bash /root/.codex/plugins/cache/deoxys/autoresearch/2.9.6/skills/shared/scripts/exp-index.sh baseline .autoresearch/experiment-indices/maximize-cifar10-best-test-accuracy.tsv`
   Pass condition: output reports `baseline=93.97` and `baseline_commit=755be2c`; improvement threshold is `94.07`.
2. Confirm only `train.py` is modified after implementation with:
   `git diff --name-only`
   Pass condition: the only tracked code path listed is `train.py`; `.autoresearch/` artifacts are local loop metadata.
3. Compile-check with:
   `python3 -m py_compile train.py`
   Pass condition: command exits 0.
4. Run style check when available:
   `uv run ruff check train.py`
   Pass condition: command exits 0; if ruff is unavailable in the existing environment, record that and proceed with compile-only preflight.
5. Remove stale logs and run:
   `rm -f run.log`
   `env CUDA_VISIBLE_DEVICES=<id> uv run train.py > run.log 2>&1`
   Pass condition: process exits 0 within 10 minutes and `run.log` reports numeric `best_test_acc`.
6. Extract metrics:
   `grep "^best_test_acc:\\|^final_test_acc:\\|^final_test_loss:\\|^training_seconds:\\|^total_seconds:\\|^startup_seconds:\\|^peak_vram_mb:\\|^num_epochs:\\|^num_steps:\\|^num_params:" run.log`
   Pass condition: all expected final summary metrics are present.
7. Classify:
   - `improvement` if `best_test_acc >= 94.07%`.
   - `no-improvement` if the run is valid but `best_test_acc < 94.07%`.
   - `crash` if no numeric `best_test_acc` is produced.

### Informational Metrics (Optional)
- `final_test_acc`: final summary line in `run.log`.
- `final_test_loss`: final summary line in `run.log`.
- `training_seconds`: final summary line in `run.log`; should be near `300.0`.
- `total_seconds`: final summary line in `run.log`; must remain under 600 seconds.
- `startup_seconds`: final summary line in `run.log`.
- `peak_vram_mb`: final summary line in `run.log`.
- `num_epochs`: final summary line in `run.log`.
- `num_steps`: final summary line in `run.log`.
- `num_params`: final summary line in `run.log`; should remain `822,790`.
