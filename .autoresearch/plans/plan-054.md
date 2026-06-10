# Plan EXP-054: Very Mild Residual Stochastic Depth
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-054.md

## Milestones

### Milestone 1: Code Change and Preflight
- [x] Create or check out `autoresearch/exp-054` from `autoresearch/dev`.
- [x] Modify `train.py` only to add training-only stochastic depth inside residual branches.
- [x] Add `STOCHASTIC_DEPTH_MAX_P = 0.03` as a top-level hyperparameter.
- [x] Pass linearly increasing drop probabilities across the 9 residual blocks, ending at 0.03 in the final block.
- [x] Preserve `STAGE_WIDTHS=(28, 56, 112)`, `BATCH_SIZE=128`, `LR=0.1`, `MOMENTUM=0.9`, `WEIGHT_DECAY=2e-4`, `LR_MILESTONES=[21000, 64000]`, reflection crop padding, label smoothing 0.05, compile, channels-last, and once-per-epoch validation.
- [x] Run `python3 -m py_compile train.py`.
- [x] Run `uv run ruff check train.py`.

### Milestone 2: Local Experiment Run
- [x] Remove any stale `run.log`.
- [x] Check GPU state with `nvidia-smi` and select one GPU with enough free memory and low utilization.
- [x] Run `env CUDA_VISIBLE_DEVICES=<id> uv run train.py > run.log 2>&1` in a foreground attached exec session.
- [x] Monitor startup, stochastic-depth config print, early evals, step timing, the step-21000 LR drop, and final metric output.

### Milestone 3: Result Capture
- [x] Extract final summary metrics from `run.log`.
- [x] Confirm startup reports `Stochastic depth max p: 0.03`.
- [x] Confirm startup reports unchanged batch geometry (`Batches per epoch: 390`).
- [x] Confirm the first LR drop appears at step 21000 as `lr: 0.0100`.
- [x] Confirm `num_params` remains `822,790`.
- [x] Classify against baseline `93.97%` and improvement threshold `94.07%`.
- [x] Record final metrics and observations in `logs/exp-log-054.md`.

## Code Changes
- **`train.py`**:
  - Add `STOCHASTIC_DEPTH_MAX_P = 0.03` next to the other top-level hyperparameters.
  - Extend `BasicBlock.__init__` with `drop_prob=0.0` and store it as `self.drop_prob`.
  - In `BasicBlock.forward`, after `bn2(conv2(out))` and before adding the shortcut, apply stochastic depth only when `self.training` and `self.drop_prob > 0`.
  - Use per-sample binary masks shaped `(batch, 1, 1, 1)` on the residual branch and divide by keep probability so the expected residual magnitude is preserved.
  - Extend `ResNet._make_layer` to pass per-block drop probabilities.
  - Assign a linear schedule over the 9 residual blocks: first block near 0, final block `0.03`.
  - Print `Stochastic depth max p: 0.03` at startup for verification.

This tests a train-time residual regularizer. Evaluation still uses the full deterministic model because stochastic depth is gated by `self.training`.

## Configuration Changes
- New hyperparameter: `STOCHASTIC_DEPTH_MAX_P = 0.03`.
- No changes to model widths, depth, batch size, optimizer, LR, LR milestones, weight decay, augmentation, label smoothing, compile, memory format, or evaluation cadence.

The 0.03 maximum is intentionally conservative because EXP-028 and EXP-051 showed residual identity bias can undertrain this fixed-budget model. This experiment differs from those failures by keeping BatchNorm initialization unchanged and applying only train-time stochastic branch omission.

## Execution Environment
- Method: local foreground run from the project root.
- Resources: one NVIDIA H20-class GPU selected via `CUDA_VISIBLE_DEVICES`.
- Estimated runtime: about 7-9 minutes wall clock including validation and startup; kill if total runtime exceeds 10 minutes.
- Log output: capture stdout/stderr to `run.log`.
- Tool skill: none; local execution only.
- Infrastructure note: detached/nohup background launches are unreliable in this environment; use a foreground attached exec session and monitor `run.log`.

## Abort Criteria
- Stop and classify as crash if `run.log` shows a Python traceback, CUDA OOM, dataset/download failure, compile failure, or stochastic-depth shape/broadcast error.
- Stop if no training progress line or eval appears within 3 minutes of process start.
- Stop if total wall-clock runtime exceeds 10 minutes.
- Stop if the tracked diff includes anything other than `train.py`.
- Stop and classify as no-improvement/weak attribution if severe GPU contention prevents reaching the step-21000 first LR drop; otherwise allow normal early accuracy variance.
- Stop and fix before launch if stochastic depth is active in evaluation mode or changes `num_params`.

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
6. Confirm stochastic-depth config:
   `grep "Stochastic depth max p" run.log`
   Pass condition: line reports `Stochastic depth max p: 0.03`.
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
