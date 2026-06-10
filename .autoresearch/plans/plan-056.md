# Plan EXP-056: Strong Weight Decay on Weights Only
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-056.md

## Milestones

### Milestone 1: Code Change and Preflight
- [x] Create or check out `autoresearch/exp-056` from `autoresearch/dev`.
- [x] Modify `train.py` only to split optimizer parameters into decay and no-decay groups.
- [x] Keep `WEIGHT_DECAY = 2e-4` for convolution and linear weights.
- [x] Set weight decay to `0.0` for all bias parameters and BatchNorm affine parameters.
- [x] Preserve `STAGE_WIDTHS=(28, 56, 112)`, `BATCH_SIZE=128`, `LR=0.1`, `MOMENTUM=0.9`, `LR_MILESTONES=[21000, 64000]`, reflection crop padding, label smoothing 0.05, compile, channels-last, and once-per-epoch validation.
- [x] Add a startup print summarizing decay/no-decay parameter counts.
- [x] Run `python3 -m py_compile train.py`.
- [x] Run `uv run ruff check train.py`.
- [x] Confirm `git diff --name-only` lists only `train.py`.

### Milestone 2: Local Foreground Experiment Run
- [x] Remove any stale `run.log`.
- [x] Check GPU state with `nvidia-smi` and select one GPU with enough free memory and low utilization.
- [x] Run `env CUDA_VISIBLE_DEVICES=<id> uv run train.py > run.log 2>&1` in an attached foreground session.
- [x] Record shell/worker PIDs if available and verify their cwd before any kill action.
- [x] Monitor startup, parameter-group print, early evals, step timing, the step-21000 LR drop, and final summary metrics.

### Milestone 3: Result Capture
- [x] Extract final summary metrics from `run.log`.
- [x] Confirm startup reports the expected weight-decay parameter groups.
- [x] Confirm startup reports unchanged batch geometry (`Batches per epoch: 390`).
- [x] Confirm the first LR drop appears at step 21000 as `lr: 0.0100`.
- [x] Confirm `num_params` remains `822,790`.
- [x] Classify against baseline `93.97%` and improvement threshold `94.07%`.
- [x] Record final metrics and observations in `logs/exp-log-056.md`.

## Code Changes
- **`train.py`**:
  - Add a small helper such as `make_weight_decay_param_groups(model)` that iterates over `model.named_modules()` and each module's direct parameters.
  - Put parameters named `bias` and all direct parameters of `nn.BatchNorm2d` modules into a no-decay group with `weight_decay=0.0`.
  - Put all remaining trainable parameters into a decay group with `weight_decay=WEIGHT_DECAY`.
  - Construct the parameter groups before `torch.compile(model)` so module names and types are straightforward, then pass those same parameter objects into `optim.SGD`.
  - Change optimizer construction from `optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)` to `optim.SGD(param_groups, lr=LR, momentum=MOMENTUM)`.
  - Print `Weight decay groups: decay_params=<N>, no_decay_params=<M>` at startup for verification.

The implementation tests whether EXP-038's useful stronger weight decay should apply only to true weights while BatchNorm affine and bias terms remain free to calibrate late refinement.

## Configuration Changes
- Optimizer parameter grouping: global all-parameter decay -> two groups:
  - Conv/linear weights: `weight_decay=2e-4`.
  - BatchNorm affine and all biases: `weight_decay=0.0`.
- No changes to nominal `WEIGHT_DECAY`, model widths, depth, batch size, LR, LR milestones, momentum, augmentation, label smoothing, compile mode, memory format, or validation cadence.

This revisits EXP-027 only because the anchor has materially changed: the active baseline now depends on stronger `2e-4` decay plus reflection padding and label smoothing. The plan is testing a coupled strong-decay refinement, not the older global-`1e-4` setting.

## Execution Environment
- Method: local foreground run from the project root.
- Resources: one NVIDIA H20-class GPU selected via `CUDA_VISIBLE_DEVICES`.
- Estimated runtime: about 7-9 minutes wall clock including validation and startup; kill if total runtime exceeds 10 minutes.
- Log output: capture stdout/stderr to `run.log`.
- Tool skill: none; local execution only.
- Infrastructure note: detached/nohup background launches are unreliable in this shell; use a foreground attached exec session and monitor `run.log`.

## Abort Criteria
- Stop and classify as crash if `run.log` shows a Python traceback, CUDA OOM, dataset/download failure, compile failure, empty optimizer parameter group error, duplicate-parameter optimizer warning/error, or no-decay grouping bug.
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
6. Confirm weight-decay groups:
   `grep "Weight decay groups" run.log`
   Pass condition: line reports nonzero decay and no-decay parameter counts.
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
