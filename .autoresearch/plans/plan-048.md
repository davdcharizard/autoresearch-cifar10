# Plan EXP-048: Lower BatchNorm Momentum to 0.05
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-048.md

## Milestones

### Milestone 1: Code Change and Preflight
- [x] Create or check out `autoresearch/exp-048` from `autoresearch/dev`.
- [x] Modify `train.py` only: introduce `BN_MOMENTUM = 0.05` and pass it to every `nn.BatchNorm2d` layer.
- [x] Verify the diff preserves model widths, batch size, optimizer, LR schedule, augmentation, label smoothing, weight decay, compile, and channels-last anchors.
- [x] Run `python3 -m py_compile train.py`.
- [x] Run `uv run ruff check train.py`.

### Milestone 2: Local Experiment Run
- [x] Remove any stale `run.log`.
- [x] Select one GPU with `CUDA_VISIBLE_DEVICES=<id>`, preferring a GPU with enough memory and lower visible utilization.
- [x] Run `env CUDA_VISIBLE_DEVICES=<id> uv run train.py > run.log 2>&1` in a foreground exec session.
- [x] Monitor startup, first epoch eval, first LR drop at step 21000, and final metric output.

### Milestone 3: Result Capture
- [x] Extract final summary metrics from `run.log`.
- [x] Confirm whether the first LR drop occurred (`lr: 0.0100` after step 21000).
- [x] Classify against baseline `93.97%` and improvement threshold `94.07%`.
- [x] Record final metrics and observations in `logs/exp-log-048.md`.

## Code Changes
- **`train.py`**: Add `BN_MOMENTUM = 0.05` near the existing hyperparameter block.
- **`train.py`**: Change the top-level `ResNet.bn1` and all `BasicBlock.bn1`/`BasicBlock.bn2` construction sites from default `nn.BatchNorm2d(channels)` to `nn.BatchNorm2d(channels, momentum=BN_MOMENTUM)`.
- **No other changes**: Preserve `STAGE_WIDTHS=(28, 56, 112)`, `BATCH_SIZE=128`, `LR=0.1`, `MOMENTUM=0.9`, `WEIGHT_DECAY=2e-4`, `LR_MILESTONES=[21000, 64000]`, reflection crop padding, `label_smoothing=0.05`, channels-last, and compile.

## Configuration Changes
- BatchNorm running-stat momentum: default `0.1` -> explicit `0.05`.
- Rationale: lower momentum smooths running mean/variance updates used by `model.eval()` in `Eval.evaluate()` without adding parameter count or meaningful runtime overhead.

## Execution Environment
- Method: local foreground run from the project root.
- Resources: one NVIDIA H20-class GPU, selected via `CUDA_VISIBLE_DEVICES`.
- Estimated runtime: about 7-9 minutes wall clock including validation and startup; kill if total runtime exceeds 10 minutes.
- Log output: capture stdout/stderr to `run.log`.
- Tool skill: none; local execution only.
- Infrastructure note: detached/nohup background launches are unreliable in this environment; use a foreground attached exec session.

## Abort Criteria
- Stop and classify as crash if `run.log` shows a Python traceback, CUDA OOM, dataset/download failure, or compile failure.
- Stop if no training progress line or eval appears within 3 minutes of process start.
- Stop if total wall-clock runtime exceeds 10 minutes.
- Stop if severe GPU contention makes reaching the first LR drop at step 21000 impossible and the run still has no realistic path to a clean comparison; otherwise allow normal early accuracy variance.

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
6. Confirm first LR drop behavior:
   `grep "lr: 0.0100" run.log`
   Pass condition: at least one post-step-21000 progress line reports `lr: 0.0100`; if absent, mark the comparison as confounded by missed milestone in the analysis.
7. Extract metrics:
   `grep "^best_test_acc:\\|^final_test_acc:\\|^final_test_loss:\\|^training_seconds:\\|^total_seconds:\\|^startup_seconds:\\|^peak_vram_mb:\\|^num_epochs:\\|^num_steps:\\|^num_params:" run.log`
   Pass condition: expected final summary metrics are present.
8. Classify:
   - `improvement` if `best_test_acc >= 94.07%`.
   - `no-improvement` if the run is valid but `best_test_acc < 94.07%`.
   - `crash` if no numeric `best_test_acc` is produced.

### Informational Metrics
- `final_test_acc`: final summary line in `run.log`.
- `final_test_loss`: final summary line in `run.log`.
- `training_seconds`: final summary line in `run.log`.
- `total_seconds`: final summary line in `run.log`; must remain under 600 seconds.
- `startup_seconds`: final summary line in `run.log`.
- `peak_vram_mb`: final summary line in `run.log`.
- `num_epochs`: final summary line in `run.log`.
- `num_steps`: final summary line in `run.log`.
- `num_params`: final summary line in `run.log`; should remain `822,790`.
