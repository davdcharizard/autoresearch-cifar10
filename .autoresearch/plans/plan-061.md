# Plan EXP-061: Final Classifier Dropout
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-061.md

## Milestones

### Milestone 1: Code Change and Preflight
- [x] Create or check out `autoresearch/exp-061` from `autoresearch/dev`.
- [x] Modify `train.py` only to add `CLASSIFIER_DROPOUT_P = 0.1`.
- [x] Add a `nn.Dropout(p=CLASSIFIER_DROPOUT_P)` module in `ResNet.__init__` immediately before the final `fc` classifier.
- [x] Apply the dropout to the pooled flattened feature vector in `ResNet.forward()` before `self.fc(out)`.
- [x] Add a startup print for `Classifier dropout p: 0.1`.
- [x] Preserve the current anchor outside this targeted head regularizer: `STAGE_WIDTHS=(28, 56, 112)`, `BATCH_SIZE=128`, `LR=0.1`, `MOMENTUM=0.9`, `WEIGHT_DECAY=2e-4`, `LR_MILESTONES=[21000, 64000]`, reflection crop padding, `label_smoothing=0.05`, compile, channels-last, and once-per-epoch validation.
- [x] Run `python3 -m py_compile train.py`.
- [x] Run `uv run ruff check train.py`.
- [x] Confirm `git diff --name-only` lists only `train.py`.

### Milestone 2: Local Foreground Experiment Run
- [x] Remove any stale `run.log`.
- [x] Check GPU state with `nvidia-smi` and select one GPU with enough free memory and low utilization.
- [x] Launch with attached foreground execution: `env CUDA_VISIBLE_DEVICES=<id> uv run train.py > run.log 2>&1`.
- [x] Record shell/worker PIDs if available and verify their cwd before any kill action.
- [x] Monitor startup, dropout print, early evals, step timing, first LR drop at step 21000, and final summary metrics.

### Milestone 3: Result Capture
- [x] Extract final summary metrics from `run.log`.
- [x] Confirm startup reports `Classifier dropout p: 0.1`.
- [x] Confirm startup reports unchanged batch geometry (`Batches per epoch: 390`).
- [x] Confirm first LR drop appears at step 21000 with `lr: 0.0100`; if absent, mark attribution as weak in analysis.
- [x] Confirm `num_params` remains `822,790` because dropout adds no parameters.
- [x] Classify against baseline `93.97%` and improvement threshold `94.07%`.
- [x] Record final metrics, LR-drop status, and observations in `logs/exp-log-061.md`.

## Code Changes
- **`train.py`**:
  - Add a top-level constant `CLASSIFIER_DROPOUT_P = 0.1`.
  - In `ResNet.__init__`, add `self.classifier_dropout = nn.Dropout(p=CLASSIFIER_DROPOUT_P)` before `self.fc`.
  - In `ResNet.forward()`, after `out = out.view(out.size(0), -1)` and before the final classifier, add `out = self.classifier_dropout(out)`.
  - Print `Classifier dropout p: 0.1` after the parameter-count/model startup line for log verification.
  - Keep all optimizer, scheduler, data augmentation, loss, evaluation, and timing behavior unchanged.

This tests a narrow training-only classifier-head regularizer. `nn.Dropout` automatically disables itself in `model.eval()`, so the fixed `Eval.evaluate()` path receives deterministic logits from the learned weights.

## Configuration Changes
- `CLASSIFIER_DROPOUT_P`: absent -> `0.1` (small head-only feature dropout before the final classifier).
- Parameter count: expected unchanged at `822,790`.
- All anchor recipe values remain unchanged.

## Execution Environment
- Method: local foreground run from the project root.
- Resources: one NVIDIA H20-class GPU selected via `CUDA_VISIBLE_DEVICES`.
- Estimated runtime: about 7-9 minutes wall clock including validation and startup; kill if total runtime exceeds 10 minutes.
- Log output: capture stdout/stderr to `run.log`.
- Tool skill: none; local execution only.
- Infrastructure note: detached/nohup background launches are unreliable in this shell. Use attached foreground execution only. If a foreground session appears to fail, inspect `run.log` and process cwd before killing or classifying.

## Abort Criteria
- Stop and classify as crash if `run.log` shows a Python traceback, CUDA OOM, dataset/download failure, compile failure, dropout/module runtime error, or non-finite loss.
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
6. Confirm dropout config:
   `grep "Classifier dropout p" run.log`
   Pass condition: line reports `Classifier dropout p: 0.1`.
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
