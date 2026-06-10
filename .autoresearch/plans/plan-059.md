# Plan EXP-059: Average-Pool Option-A Downsample Shortcut
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-059.md

## Milestones

### Milestone 1: Code Change and Preflight
- [x] Create or check out `autoresearch/exp-059` from `autoresearch/dev`.
- [x] Modify `train.py` only to replace stride-2 shortcut slicing with average pooling in `BasicBlock.forward`.
- [x] Apply `F.avg_pool2d(shortcut, kernel_size=self.stride, stride=self.stride)` only inside the existing `if self.need_pad:` branch.
- [x] Preserve zero-channel padding after shortcut spatial downsampling.
- [x] Preserve `STAGE_WIDTHS=(28, 56, 112)`, `BATCH_SIZE=128`, `LR=0.1`, `MOMENTUM=0.9`, `WEIGHT_DECAY=2e-4`, `LR_MILESTONES=[21000, 64000]`, reflection crop padding, full-run `label_smoothing=0.05`, compile, channels-last, and once-per-epoch validation.
- [x] Add a startup print summarizing `Shortcut downsample: avg_pool_option_a`.
- [x] Run `python3 -m py_compile train.py`.
- [x] Run `uv run ruff check train.py`.
- [x] Confirm `git diff --name-only` lists only `train.py`.

### Milestone 2: Local Foreground Experiment Run
- [x] Remove any stale `run.log`.
- [x] Check GPU state with `nvidia-smi` and select one GPU with enough free memory and low utilization.
- [x] Run `env CUDA_VISIBLE_DEVICES=<id> uv run train.py > run.log 2>&1` in an attached foreground session.
- [x] Record shell/worker PIDs if available and verify their cwd before any kill action.
- [x] Monitor startup, early evals, shape/runtime errors, step timing, first LR drop at step 21000, and final summary metrics.

### Milestone 3: Result Capture
- [x] Extract final summary metrics from `run.log`.
- [x] Confirm startup reports `Shortcut downsample: avg_pool_option_a`.
- [x] Confirm startup reports unchanged batch geometry (`Batches per epoch: 390`).
- [x] Confirm the first LR drop appears at step 21000 as `lr: 0.0100`; if absent, flag the comparison as confounded.
- [x] Confirm `num_params` remains `822,790`.
- [x] Classify against baseline `93.97%` and improvement threshold `94.07%`.
- [x] Record final metrics and observations in `logs/exp-log-059.md`.

## Code Changes
- **`train.py`**:
  - Add a constant `SHORTCUT_DOWNSAMPLE = "avg_pool_option_a"` for startup logging.
  - In `BasicBlock.forward`, replace `shortcut = shortcut[:, :, :: self.stride, :: self.stride]` with `shortcut = F.avg_pool2d(shortcut, kernel_size=self.stride, stride=self.stride)` inside the existing downsample branch.
  - Keep the existing `F.pad(shortcut, (0, 0, 0, 0, 0, self.pad_channels))` channel padding exactly as-is.
  - Print the shortcut setting at startup after model creation.

This tests a parameter-free stage-transition change: preserve more local shortcut information during stride-2 transitions while keeping the option-A zero-padding family and all training anchors unchanged.

## Configuration Changes
- Shortcut spatial downsample: strided slicing (`::stride`) -> average pooling (`avg_pool2d`) before zero-channel padding.
- No changes to trainable parameter count, model widths/depth, optimizer, batch size, LR milestones, data augmentation, loss smoothing, compile mode, memory format, validation cadence, or training time budget.

This differs from EXP-018 because it does not add learned projection shortcuts, and differs from EXP-058 because it does not add per-block attention parameters or broad compute overhead.

## Execution Environment
- Method: local foreground run from the project root.
- Resources: one NVIDIA H20-class GPU selected via `CUDA_VISIBLE_DEVICES`.
- Estimated runtime: about 7-9 minutes wall clock including validation and startup; kill if total runtime exceeds 10 minutes.
- Log output: capture stdout/stderr to `run.log`.
- Tool skill: none; local execution only.
- Infrastructure note: detached/nohup background launches are unreliable in this shell; use a foreground attached exec session and monitor `run.log`.

## Abort Criteria
- Stop and classify as crash if `run.log` shows a Python traceback, CUDA OOM, dataset/download failure, compile failure, or tensor shape mismatch in shortcut addition.
- Stop if no training progress line or eval appears within 3 minutes of process start.
- Stop if total wall-clock runtime exceeds 10 minutes.
- Stop if the tracked diff includes anything other than `train.py`.
- Stop and classify as no-improvement/weak attribution if severe GPU contention prevents reaching the step-21000 first LR drop; otherwise allow normal early accuracy variance.

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
4. Style-check with:
   `uv run ruff check train.py`
   Pass condition: command exits 0.
5. Remove stale logs and run:
   `rm -f run.log`
   `env CUDA_VISIBLE_DEVICES=<id> uv run train.py > run.log 2>&1`
   Pass condition: process exits 0 within 10 minutes and `run.log` reports numeric `best_test_acc`.
6. Confirm startup shortcut configuration:
   `grep "Shortcut downsample" run.log`
   Pass condition: line reports `avg_pool_option_a`.
7. Confirm batch geometry:
   `grep "Batches per epoch" run.log`
   Pass condition: line reports `Batches per epoch: 390`.
8. Confirm first LR drop behavior:
   `grep "step 21000" run.log`
   Pass condition: the matching progress line reports `lr: 0.0100`; if absent, mark the comparison as confounded by missed milestone in the analysis.
9. Extract metrics:
   `grep "^best_test_acc:\\|^final_test_acc:\\|^final_test_loss:\\|^training_seconds:\\|^total_seconds:\\|^startup_seconds:\\|^peak_vram_mb:\\|^num_epochs:\\|^num_steps:\\|^num_params:" run.log`
   Pass condition: expected final summary metrics are present and `num_params` is `822,790`.
10. Classify:
   - `improvement` if `best_test_acc >= 94.07%`.
   - `no-improvement` if the run is valid but `best_test_acc < 94.07%`.
   - `crash` if no numeric `best_test_acc` is produced.

### Informational Metrics (Optional)
- `final_test_acc`: final summary line in `run.log`.
- `final_test_loss`: final summary line in `run.log`.
- `training_seconds`: final summary line in `run.log`.
- `total_seconds`: final summary line in `run.log`; must remain under 600 seconds.
- `startup_seconds`: final summary line in `run.log`.
- `peak_vram_mb`: final summary line in `run.log`.
- `num_epochs`: final summary line in `run.log`.
- `num_steps`: final summary line in `run.log`.
- `num_params`: final summary line in `run.log`; should remain `822,790`.
