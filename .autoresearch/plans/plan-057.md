# Plan EXP-057: Post-Drop Label Smoothing Anneal
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-057.md

## Milestones

### Milestone 1: Code Change and Preflight
- [x] Create or check out `autoresearch/exp-057` from `autoresearch/dev`.
- [x] Modify `train.py` only to add explicit label-smoothing schedule constants.
- [x] Keep `label_smoothing=0.05` while `step < 21000`.
- [x] Switch to `label_smoothing=0.0` once the training step counter reaches 21000 and the scheduler has dropped LR to 0.01.
- [x] Preserve `STAGE_WIDTHS=(28, 56, 112)`, `BATCH_SIZE=128`, `LR=0.1`, `MOMENTUM=0.9`, `WEIGHT_DECAY=2e-4`, `LR_MILESTONES=[21000, 64000]`, reflection crop padding, compile, channels-last, and once-per-epoch validation.
- [x] Add a startup print summarizing the pre-drop and post-drop label smoothing values and switch step.
- [x] Ensure progress logs expose the active smoothing value so the switch can be verified from `run.log`.
- [x] Run `python3 -m py_compile train.py`.
- [x] Run `uv run ruff check train.py`.
- [x] Confirm `git diff --name-only` lists only `train.py`.

### Milestone 2: Local Foreground Experiment Run
- [x] Remove any stale `run.log`.
- [x] Check GPU state with `nvidia-smi` and select one GPU with enough free memory and low utilization.
- [x] Run `env CUDA_VISIBLE_DEVICES=<id> uv run train.py > run.log 2>&1` in an attached foreground session.
- [x] Record shell/worker PIDs if available and verify their cwd before any kill action.
- [x] Monitor startup, early evals, active smoothing values, step timing, the step-21000 LR drop, and final summary metrics.

### Milestone 3: Result Capture
- [x] Extract final summary metrics from `run.log`.
- [x] Confirm startup reports `Label smoothing schedule: pre_drop=0.05, post_drop=0.0, switch_step=21000`.
- [x] Confirm progress logs show `ls: 0.050` before step 21000 and `ls: 0.000` after the switch.
- [x] Confirm startup reports unchanged batch geometry (`Batches per epoch: 390`).
- [x] Confirm the first LR drop appears at step 21000 as `lr: 0.0100`.
- [x] Confirm `num_params` remains `822,790`.
- [x] Classify against baseline `93.97%` and improvement threshold `94.07%`.
- [x] Record final metrics and observations in `logs/exp-log-057.md`.

## Code Changes
- **`train.py`**:
  - Add constants such as `LABEL_SMOOTHING_PRE_DROP = 0.05`, `LABEL_SMOOTHING_POST_DROP = 0.0`, and `LABEL_SMOOTHING_SWITCH_STEP = LR_MILESTONES[0]`.
  - Print the label-smoothing schedule at startup.
  - Before computing the loss each batch, choose the active smoothing value from the current `step` counter before incrementing it: use 0.05 for steps before 21000 and 0.0 for step 21000 onward.
  - Replace the hard-coded `F.cross_entropy(..., label_smoothing=0.05)` call with `F.cross_entropy(..., label_smoothing=label_smoothing)`.
  - Include `label_smoothing` in the existing every-50-step progress print as `ls: {label_smoothing:.3f}`.

The implementation tests whether the validated high-LR smoothing should be removed during low-LR refinement. It intentionally preserves the rest of the current anchor.

## Configuration Changes
- Loss schedule: static `label_smoothing=0.05` -> `0.05` before step 21000 and `0.0` from step 21000 onward.
- No changes to model width/depth, optimizer, batch size, LR milestones, data augmentation, compile mode, memory format, validation cadence, or training time budget.

This differs from the recurring smoothing-deviation failures because EXP-033 and EXP-037 used static smoothing values for the whole run; EXP-057 preserves the validated smoothing value during the high-LR phase and changes only the low-LR phase.

## Execution Environment
- Method: local foreground run from the project root.
- Resources: one NVIDIA H20-class GPU selected via `CUDA_VISIBLE_DEVICES`.
- Estimated runtime: about 7-9 minutes wall clock including validation and startup; kill if total runtime exceeds 10 minutes.
- Log output: capture stdout/stderr to `run.log`.
- Tool skill: none; local execution only.
- Infrastructure note: detached/nohup background launches are unreliable in this shell; use a foreground attached exec session and monitor `run.log`.

## Abort Criteria
- Stop and classify as crash if `run.log` shows a Python traceback, CUDA OOM, dataset/download failure, compile failure, invalid `label_smoothing` value, or scheduler/step logic error.
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
6. Confirm startup label-smoothing schedule:
   `grep "Label smoothing schedule" run.log`
   Pass condition: line reports `pre_drop=0.05`, `post_drop=0.0`, and `switch_step=21000`.
7. Confirm active smoothing switch:
   `grep "ls: 0.050" run.log | head` and `grep "ls: 0.000" run.log | head`
   Pass condition: pre-drop progress lines include `ls: 0.050`, and post-drop progress lines include `ls: 0.000`.
8. Confirm batch geometry:
   `grep "Batches per epoch" run.log`
   Pass condition: line reports `Batches per epoch: 390`.
9. Confirm first LR drop behavior:
   `grep "step 21000" run.log`
   Pass condition: the matching progress line reports `lr: 0.0100`; if absent, mark the comparison as confounded by missed milestone in the analysis.
10. Extract metrics:
   `grep "^best_test_acc:\\|^final_test_acc:\\|^final_test_loss:\\|^training_seconds:\\|^total_seconds:\\|^startup_seconds:\\|^peak_vram_mb:\\|^num_epochs:\\|^num_steps:\\|^num_params:" run.log`
   Pass condition: expected final summary metrics are present and `num_params` is `822,790`.
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
- `num_params`: final summary line in `run.log`; should remain `822,790`.
