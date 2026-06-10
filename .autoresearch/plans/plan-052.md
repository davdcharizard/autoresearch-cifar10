# Plan EXP-052: Hybrid Post-Drop Cosine LR Tail
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-052.md

## Milestones

### Milestone 1: Code Change and Preflight
- [x] Create or check out `autoresearch/exp-052` from `autoresearch/dev`.
- [x] Modify `train.py` only: replace the flat post-21000 LR tail with a manual cosine tail from `0.0100` to `0.0020`.
- [x] Preserve model widths, batch size, optimizer family, momentum, weight decay, augmentation, label smoothing, compile, channels-last, and validation cadence.
- [x] Run `python3 -m py_compile train.py`.
- [x] Run `uv run ruff check train.py`.

### Milestone 2: Local Experiment Run
- [x] Remove any stale `run.log`.
- [x] Check GPU state with `nvidia-smi` and select one GPU with enough free memory and low utilization.
- [x] Run `env CUDA_VISIBLE_DEVICES=<id> uv run train.py > run.log 2>&1` in a foreground exec session.
- [x] Monitor startup, early evals, the step-21000 first LR drop, the cosine-tail LR values, and final metric output.

### Milestone 3: Result Capture
- [x] Extract final summary metrics from `run.log`.
- [x] Confirm the first LR drop still appears at step 21000 as `lr: 0.0100`.
- [x] Confirm later tail lines show LR below 0.0100 and above or equal to the 0.0020 floor.
- [x] Confirm `num_params` remains `822,790`.
- [x] Classify against baseline `93.97%` and improvement threshold `94.07%`.
- [x] Record final metrics and observations in `logs/exp-log-052.md`.

## Code Changes
- **`train.py`**:
  - Add `import math`.
  - Replace `LR_MILESTONES = [21000, 64000]` with explicit tail constants:
    - `TAIL_START_STEP = 21000`
    - `TAIL_END_STEP = 42000`
    - `TAIL_MIN_LR = 0.002`
  - Remove `optim.lr_scheduler.MultiStepLR`.
  - Add a small local helper that computes the LR to use after each completed step:
    - steps `< 21000`: `0.1`
    - step `21000`: `0.01`, matching the anchor's first-drop log behavior
    - steps `21000..42000`: cosine decay from `0.01` to `0.002`
    - steps `>= 42000`: hold `0.002`
  - After each `optimizer.step()`, increment `step`, set all optimizer param-group LRs using the helper, and then log the current LR.

This tests a schedule-tail shape only. It intentionally preserves the validated high-LR phase and differs from the known failed second-drop family by using a smooth decay and a nonzero LR floor rather than an abrupt `0.001` refinement phase.

## Configuration Changes
- `LR_MILESTONES=[21000, 64000]` flat tail -> manual hybrid schedule with first drop at 21000 and cosine tail to `TAIL_MIN_LR=0.002`.
- `TAIL_END_STEP=42000`: chosen because recent clean anchor-like runs usually finish around 40k-42k steps; the floor should be reached near the end of the 300s training budget without requiring a second milestone.
- No changes to `STAGE_WIDTHS=(28, 56, 112)`, `BATCH_SIZE=128`, `LR=0.1`, `MOMENTUM=0.9`, `WEIGHT_DECAY=2e-4`, crop/flip/reflection transforms, `label_smoothing=0.05`, compile, channels-last, or once-per-epoch evaluation.

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
- Stop if the run clearly misses the first LR drop at step 21000 due to severe GPU contention; otherwise allow normal early accuracy variance.
- Stop if the implementation changes optimizer family, model size, data transforms, label smoothing, batch size, weight decay, or evaluation frequency.
- Stop and fix before launch if preflight shows the tracked diff includes anything other than `train.py`.

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
   `grep "step 21000" run.log`
   Pass condition: the matching progress line reports `lr: 0.0100`; if absent, mark the comparison as confounded by missed milestone in the analysis.
7. Confirm cosine-tail behavior:
   inspect progress lines after step 21000, especially around steps 30000-42000.
   Pass condition: reported LR decreases smoothly below `0.0100` and never below `0.0020`.
8. Extract metrics:
   `grep "^best_test_acc:\\|^final_test_acc:\\|^final_test_loss:\\|^training_seconds:\\|^total_seconds:\\|^startup_seconds:\\|^peak_vram_mb:\\|^num_epochs:\\|^num_steps:\\|^num_params:" run.log`
   Pass condition: expected final summary metrics are present and `num_params` is `822,790`.
9. Classify:
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
