# Plan EXP-058: Squeeze-and-Excitation BasicBlocks
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-058.md

## Milestones

### Milestone 1: Code Change and Preflight
- [x] Create or check out `autoresearch/exp-058` from `autoresearch/dev`.
- [x] Modify `train.py` only to add a lightweight `SEBlock`.
- [x] Insert the SE gate in every `BasicBlock` after `bn2(conv2)` and before shortcut addition.
- [x] Use a conservative reduction ratio of 16 with a minimum bottleneck width of 4.
- [x] Preserve `STAGE_WIDTHS=(28, 56, 112)`, `BATCH_SIZE=128`, `LR=0.1`, `MOMENTUM=0.9`, `WEIGHT_DECAY=2e-4`, `LR_MILESTONES=[21000, 64000]`, reflection crop padding, full-run `label_smoothing=0.05`, compile, channels-last, and once-per-epoch validation.
- [x] Add a startup print summarizing the SE setting and reduction ratio.
- [x] Run `python3 -m py_compile train.py`.
- [x] Run `uv run ruff check train.py`.
- [x] Confirm `git diff --name-only` lists only `train.py`.

### Milestone 2: Local Foreground Experiment Run
- [x] Remove any stale `run.log`.
- [x] Check GPU state with `nvidia-smi` and select one GPU with enough free memory and low utilization.
- [x] Run `env CUDA_VISIBLE_DEVICES=<id> uv run train.py > run.log 2>&1` in an attached foreground session.
- [x] Record shell/worker PIDs if available and verify their cwd before any kill action.
- [x] Monitor startup, early evals, step timing, first LR drop at step 21000, and final summary metrics.

### Milestone 3: Result Capture
- [x] Extract final summary metrics from `run.log`.
- [x] Confirm startup reports `SE blocks: enabled, reduction=16`.
- [x] Confirm startup reports unchanged batch geometry (`Batches per epoch: 390`).
- [x] Confirm the first LR drop appears at step 21000 as `lr: 0.0100`; if absent, flag the comparison as confounded.
- [x] Record `num_params` so the architectural overhead is explicit.
- [x] Classify against baseline `93.97%` and improvement threshold `94.07%`.
- [x] Record final metrics and observations in `logs/exp-log-058.md`.

## Code Changes
- **`train.py`**:
  - Add constants `USE_SE = True` and `SE_REDUCTION = 16`.
  - Add an `SEBlock` module using `nn.AdaptiveAvgPool2d(1)`, a bottleneck `1x1` convolution, ReLU, a second `1x1` convolution, and sigmoid gating.
  - Instantiate `self.se = SEBlock(out_channels, SE_REDUCTION) if USE_SE else nn.Identity()` in `BasicBlock`.
  - In `BasicBlock.forward`, apply `out = self.se(out)` immediately after `out = self.bn2(self.conv2(out))` and before adding the shortcut.
  - Print `SE blocks: enabled, reduction=16` at startup after model creation.

The implementation tests channel-wise feature recalibration while intentionally preserving the current training recipe and benchmark protocol.

## Configuration Changes
- Architecture: plain `BasicBlock` -> `BasicBlock` with SE gate on the residual branch.
- SE reduction ratio: `16`, with bottleneck width `max(channels // 16, 4)`.
- No changes to model stage widths, depth, optimizer, batch size, LR milestones, data augmentation, loss smoothing, compile mode, memory format, validation cadence, or training time budget.

This differs from failed width/projection experiments because it keeps the validated 28/56/112 backbone and option-A shortcut unchanged; only residual-branch channel recalibration is added.

## Execution Environment
- Method: local foreground run from the project root.
- Resources: one NVIDIA H20-class GPU selected via `CUDA_VISIBLE_DEVICES`.
- Estimated runtime: about 7-9 minutes wall clock including validation and startup; kill if total runtime exceeds 10 minutes.
- Log output: capture stdout/stderr to `run.log`.
- Tool skill: none; local execution only.
- Infrastructure note: detached/nohup background launches are unreliable in this shell; use a foreground attached exec session and monitor `run.log`.

## Abort Criteria
- Stop and classify as crash if `run.log` shows a Python traceback, CUDA OOM, dataset/download failure, compile failure, or invalid tensor shape from the SE gate.
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
6. Confirm startup SE configuration:
   `grep "SE blocks" run.log`
   Pass condition: line reports `enabled, reduction=16`.
7. Confirm batch geometry:
   `grep "Batches per epoch" run.log`
   Pass condition: line reports `Batches per epoch: 390`.
8. Confirm first LR drop behavior:
   `grep "step 21000" run.log`
   Pass condition: the matching progress line reports `lr: 0.0100`; if absent, mark the comparison as confounded by missed milestone in the analysis.
9. Extract metrics:
   `grep "^best_test_acc:\\|^final_test_acc:\\|^final_test_loss:\\|^training_seconds:\\|^total_seconds:\\|^startup_seconds:\\|^peak_vram_mb:\\|^num_epochs:\\|^num_steps:\\|^num_params:" run.log`
   Pass condition: expected final summary metrics are present.
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
- `num_params`: final summary line in `run.log`; compare to anchor `822,790` to quantify SE overhead.
