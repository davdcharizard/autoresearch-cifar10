# Plan EXP-064: Probabilistic CutMix Regional Mixing
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-064.md

## Milestones

### Milestone 1: Code Change and Preflight
- [x] Create or check out `autoresearch/exp-064` from `autoresearch/dev`.
- [x] Modify `train.py` only to add probabilistic CutMix with `CUTMIX_ALPHA = 1.0`, `CUTMIX_PROB = 0.5`, and `CUTMIX_LABEL_SMOOTHING = 0.05`.
- [x] Add a small helper that samples a clipped rectangular box from lambda and recomputes lambda from the actual pasted area.
- [x] Apply CutMix on-device after inputs and targets are moved to the selected GPU; leave non-CutMix batches on the anchor loss.
- [x] Preserve anchor settings: `NUM_BLOCKS=3`, `STAGE_WIDTHS=(28,56,112)`, `BATCH_SIZE=128`, `LR=0.1`, `MOMENTUM=0.9`, `WEIGHT_DECAY=2e-4`, `LR_MILESTONES=[21000,64000]`, reflection crop padding, FP32 compile, channels-last, and once-per-epoch validation.
- [x] Add startup print markers for `CutMix alpha`, `CutMix prob`, and `CutMix label smoothing`.
- [x] Run `python3 -m py_compile train.py`.
- [x] Run `uv run ruff check train.py`.
- [x] Confirm `git diff --name-only` lists only `train.py`.

### Milestone 2: Local Foreground Experiment Run
- [x] Remove any stale `run.log`.
- [x] Check GPU state with `nvidia-smi` and select one GPU with enough free memory and low utilization.
- [x] Launch with attached foreground execution: `env CUDA_VISIBLE_DEVICES=<id> uv run train.py > run.log 2>&1`.
- [x] Record shell/worker PIDs if available and verify their cwd before any kill action.
- [x] Monitor startup, early evals, step timing, the first LR drop at step 21000, and final summary metrics.

### Milestone 3: Result Capture
- [x] Extract final summary metrics from `run.log`.
- [x] Confirm startup reports `ResNet-20`.
- [x] Confirm startup reports `CutMix alpha: 1.0, prob: 0.5, label smoothing: 0.05`.
- [x] Confirm startup reports unchanged batch geometry (`Batches per epoch: 390`).
- [x] Confirm first LR drop appears at step 21000 with `lr: 0.0100`; if absent, mark attribution as weak in analysis.
- [x] Record `num_params`, `num_steps`, `num_epochs`, and throughput relative to the anchor and EXP-055/EXP-060 mixup runs.
- [x] Classify against baseline `93.97%` and improvement threshold `94.07%`.
- [x] Record final metrics, LR-drop status, and observations in `logs/exp-log-064.md`.

## Code Changes
- **`train.py`**:
  - Add `CUTMIX_ALPHA = 1.0`, `CUTMIX_PROB = 0.5`, and `CUTMIX_LABEL_SMOOTHING = 0.05`.
  - Add a `rand_bbox(size, lam, device)` helper that computes the CutMix patch size from `sqrt(1 - lam)`, samples the center on-device, clips the coordinates to the image bounds, and returns `bbx1, bby1, bbx2, bby2`.
  - In the training loop, after moving `inputs` and `targets` to the GPU, sample whether to apply CutMix. If active, clone `inputs`, permute the batch, paste the source patch into the cloned tensor, recompute lambda as `1 - patch_area / (H * W)`, and train with `lam * CE(outputs, targets_a) + (1 - lam) * CE(outputs, targets_b)`.
  - Keep `label_smoothing=0.05` in both CutMix endpoint losses. This preserves the validated label-smoothing anchor while making the intervention regional mixing, not a label-smoothing deviation.
  - For non-CutMix batches, keep the existing single-target `F.cross_entropy(outputs, targets, label_smoothing=0.05)` path.
  - Print the CutMix settings at startup for verification.

This tests a regional image/label mixing mechanism that is distinct from both erased-patch Cutout and global whole-image mixup.

## Configuration Changes
- `CUTMIX_ALPHA`: new `1.0`, the standard symmetric beta setting for broad patch-area variation.
- `CUTMIX_PROB`: new `0.5`, bounding regularization strength so half of batches remain the current anchor.
- `CUTMIX_LABEL_SMOOTHING`: new `0.05`, preserving the current validated smoothing level for endpoint losses.
- No architecture, optimizer, schedule, batch-size, dataset, dependency, or evaluation-harness changes.

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
   Pass condition: line reports `ResNet-20`.
7. Confirm CutMix startup settings:
   `grep "CutMix alpha" run.log`
   Pass condition: line reports `CutMix alpha: 1.0, prob: 0.5, label smoothing: 0.05`.
8. Confirm batch geometry:
   `grep "Batches per epoch" run.log`
   Pass condition: line reports `Batches per epoch: 390`.
9. Confirm first LR drop behavior:
   `grep "step 21000" run.log`
   Pass condition: the matching progress line reports `lr: 0.0100`; if absent, mark the comparison as confounded by missed milestone in analysis.
10. Extract metrics:
   `grep "^best_test_acc:\\|^final_test_acc:\\|^final_test_loss:\\|^training_seconds:\\|^total_seconds:\\|^startup_seconds:\\|^peak_vram_mb:\\|^num_epochs:\\|^num_steps:\\|^num_params:" run.log`
   Pass condition: expected final summary metrics are present and `num_params` is numeric.
11. Classify:
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
- `num_params`: final summary line in `run.log`; expected to match the 822,790-param anchor.
