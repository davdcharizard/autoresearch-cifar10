# Plan EXP-062: Compact ResNet-14 with Moderate Width Increase
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-062.md

## Milestones

### Milestone 1: Code Change and Preflight
- [x] Create or check out `autoresearch/exp-062` from `autoresearch/dev`.
- [x] Modify `train.py` only to set `NUM_BLOCKS = 2`.
- [x] Modify `train.py` only to set `STAGE_WIDTHS = (32, 64, 128)`.
- [x] Preserve all non-architecture anchor settings: `BATCH_SIZE=128`, `LR=0.1`, `MOMENTUM=0.9`, `WEIGHT_DECAY=2e-4`, `LR_MILESTONES=[21000, 64000]`, reflection crop padding, `label_smoothing=0.05`, compile, channels-last, and once-per-epoch validation.
- [x] Confirm startup should report `ResNet-14` and record the new parameter count from `run.log`.
- [x] Run `python3 -m py_compile train.py`.
- [x] Run `uv run ruff check train.py`.
- [x] Confirm `git diff --name-only` lists only `train.py`.

### Milestone 2: Local Foreground Experiment Run
- [x] Remove any stale `run.log`.
- [x] Check GPU state with `nvidia-smi` and select one GPU with enough free memory and low utilization.
- [x] Launch with attached foreground execution: `env CUDA_VISIBLE_DEVICES=<id> uv run train.py > run.log 2>&1`.
- [x] Record shell/worker PIDs if available and verify their cwd before any kill action.
- [x] Monitor startup, early evals, step timing, first LR drop at step 21000, and final summary metrics.

### Milestone 3: Result Capture
- [x] Extract final summary metrics from `run.log`.
- [x] Confirm startup reports `ResNet-14`.
- [x] Confirm startup reports unchanged batch geometry (`Batches per epoch: 390`).
- [x] Confirm first LR drop appears at step 21000 with `lr: 0.0100`; if absent, mark attribution as weak in analysis.
- [x] Record new `num_params`, `num_steps`, and throughput/epoch count relative to the anchor.
- [x] Classify against baseline `93.97%` and improvement threshold `94.07%`.
- [x] Record final metrics, LR-drop status, and observations in `logs/exp-log-062.md`.

## Code Changes
- **`train.py`**:
  - Change `NUM_BLOCKS = 3` to `NUM_BLOCKS = 2`, making the model report `ResNet-14`.
  - Change `STAGE_WIDTHS = (28, 56, 112)` to `STAGE_WIDTHS = (32, 64, 128)`.
  - Keep the existing `BasicBlock`, option-A shortcut, optimizer, scheduler, transforms, loss, evaluator, compile path, channels-last path, and timing logic unchanged.

This tests whether a shallower two-block-per-stage residual network can afford moderately wider channels under the fixed time budget and improve the capacity/throughput tradeoff beyond the current three-block 28/56/112 anchor.

## Configuration Changes
- `NUM_BLOCKS`: `3` -> `2`.
- `STAGE_WIDTHS`: `(28, 56, 112)` -> `(32, 64, 128)`.
- Expected startup model name: `ResNet-14`.
- Parameter count: expected to change; record the exact value from `run.log`.
- All optimizer, schedule, data, loss, evaluation, and runtime settings remain unchanged.

## Execution Environment
- Method: local foreground run from the project root.
- Resources: one NVIDIA H20-class GPU selected via `CUDA_VISIBLE_DEVICES`.
- Estimated runtime: about 7-9 minutes wall clock including validation and startup; kill if total runtime exceeds 10 minutes.
- Log output: capture stdout/stderr to `run.log`.
- Tool skill: none; local execution only.
- Infrastructure note: detached/nohup background launches are unreliable in this shell. Use attached foreground execution only. If a foreground session appears to fail, inspect `run.log` and process cwd before killing or classifying.

## Abort Criteria
- Stop and classify as crash if `run.log` shows a Python traceback, CUDA OOM, dataset/download failure, compile failure, shape/runtime error, or non-finite loss.
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
6. Confirm model depth:
   `grep "ResNet-" run.log`
   Pass condition: line reports `ResNet-14`.
7. Confirm batch geometry:
   `grep "Batches per epoch" run.log`
   Pass condition: line reports `Batches per epoch: 390`.
8. Confirm first LR drop behavior:
   `grep "step 21000" run.log`
   Pass condition: the matching progress line reports `lr: 0.0100`; if absent, mark the comparison as confounded by missed milestone in the analysis.
9. Extract metrics:
   `grep "^best_test_acc:\\|^final_test_acc:\\|^final_test_loss:\\|^training_seconds:\\|^total_seconds:\\|^startup_seconds:\\|^peak_vram_mb:\\|^num_epochs:\\|^num_steps:\\|^num_params:" run.log`
   Pass condition: expected final summary metrics are present and `num_params` is numeric.
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
- `num_params`: final summary line in `run.log`; compare to the 822,790-param anchor.
