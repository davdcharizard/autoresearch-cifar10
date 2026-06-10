# Plan EXP-055: Reliable Mild Mixup Retry
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-055.md

## Milestones

### Milestone 1: Code Change and Preflight
- [x] Create or check out `autoresearch/exp-055` from `autoresearch/dev`.
- [x] Modify `train.py` only to add mild batch-level mixup with `MIXUP_ALPHA = 0.1`.
- [x] Preserve the full current anchor: `STAGE_WIDTHS=(28, 56, 112)`, `BATCH_SIZE=128`, `LR=0.1`, `MOMENTUM=0.9`, `WEIGHT_DECAY=2e-4`, `LR_MILESTONES=[21000, 64000]`, reflection crop padding, label smoothing 0.05, compile, channels-last, and once-per-epoch validation.
- [x] Add a startup print for `Mixup alpha: 0.1`.
- [x] Run `python3 -m py_compile train.py`.
- [x] Run `uv run ruff check train.py`.
- [x] Confirm `git diff --name-only` lists only `train.py`.

### Milestone 2: Local Foreground Experiment Run
- [x] Remove any stale `run.log`.
- [x] Check GPU state with `nvidia-smi` and select one GPU with enough free memory and low utilization.
- [x] Launch with attached foreground execution: `env CUDA_VISIBLE_DEVICES=<id> uv run train.py > run.log 2>&1`.
- [x] Record shell/worker PIDs if available and verify their cwd before any kill action.
- [x] Monitor startup, `Mixup alpha` print, early evals, step timing, first LR drop at step 21000, and final summary metrics.

### Milestone 3: Result Capture
- [x] Extract final summary metrics from `run.log`.
- [x] Confirm startup reports `Mixup alpha: 0.1`.
- [x] Confirm startup reports unchanged batch geometry (`Batches per epoch: 390`).
- [x] Confirm first LR drop appears at step 21000 with `lr: 0.0100`; if absent, mark attribution as weak in analysis.
- [x] Confirm `num_params` remains `822,790`.
- [x] Classify against baseline `93.97%` and improvement threshold `94.07%`.
- [x] Record final metrics, LR-drop status, and observations in `logs/exp-log-055.md`.

## Code Changes
- **`train.py`**:
  - Add `MIXUP_ALPHA = 0.1` near the other top-level hyperparameters.
  - Print `Mixup alpha: 0.1` after model startup for log verification.
  - Add a helper or inline training-loop block that samples `lam ~ Beta(MIXUP_ALPHA, MIXUP_ALPHA)` for each batch, creates an on-device random permutation, computes `mixed_inputs = lam * inputs + (1 - lam) * inputs[index]`, and keeps `targets_a`, `targets_b`.
  - Replace the standard single-target loss in training with:
    `lam * F.cross_entropy(outputs, targets_a, label_smoothing=0.05) + (1 - lam) * F.cross_entropy(outputs, targets_b, label_smoothing=0.05)`.
  - Keep evaluation unchanged; mixup is active only in the training loop.

The implementation directly retries EXP-042 under a reliable execution method. It tests whether input/label interpolation improves generalization without changing architecture, optimizer, LR schedule, augmentation, or the fixed evaluation harness.

## Configuration Changes
- `MIXUP_ALPHA`: absent -> `0.1` (same mild value as EXP-042, chosen to directly resolve the prior crash-only result).
- No changes to model widths, depth, batch size, optimizer, LR, LR milestones, weight decay, crop/flip augmentation, label smoothing value, compile mode, memory format, or validation cadence.

## Execution Environment
- Method: local foreground run from the project root.
- Resources: one NVIDIA H20-class GPU selected via `CUDA_VISIBLE_DEVICES`.
- Estimated runtime: about 7-9 minutes wall clock including validation and startup; kill if total runtime exceeds 10 minutes.
- Log output: capture stdout/stderr to `run.log`.
- Tool skill: none; local execution only.
- Infrastructure note: detached/nohup background launches are unreliable in this shell. Use attached foreground execution only. If a session appears to fail, inspect `run.log` and process cwd before killing or classifying.

## Abort Criteria
- Stop and classify as crash if `run.log` shows a Python traceback, CUDA OOM, dataset/download failure, compile failure, mixup tensor shape error, or label/loss runtime error.
- Stop if no training progress line or eval appears within 3 minutes of process start.
- Stop if total wall-clock runtime exceeds 10 minutes.
- Stop if the tracked diff includes anything other than `train.py`.
- Stop and classify as no-improvement/weak attribution if severe GPU contention prevents reaching the step-21000 first LR drop; otherwise allow normal early accuracy variance.
- If a foreground exec returns without final metrics and no obvious traceback, inspect the log tail and process state once. Retry at most once only if there is clear infrastructure interruption rather than a code error.

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
6. Confirm mixup config:
   `grep "Mixup alpha" run.log`
   Pass condition: line reports `Mixup alpha: 0.1`.
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
