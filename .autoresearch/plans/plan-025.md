# Plan EXP-025: Batch Size 96 with Step-Budget-Aware Milestones
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-025.md

## Milestones

### Milestone 1: Implement batch-size and schedule changes
- [x] Create experiment branch `autoresearch/exp-025` from `autoresearch/dev`.
- [x] Modify only `train.py`.
- [x] Change `BATCH_SIZE` from `128` to `96`.
- [x] Change `LR_MILESTONES` from `[21000, 64000]` to `[26000, 44000]`.
- [x] Preserve `STAGE_WIDTHS = (28, 56, 112)`, `WEIGHT_DECAY = 1e-4`, FP32 training, channels-last, cuDNN benchmark, `torch.compile`, seed, augmentation, optimizer class, and once-per-epoch validation.
- [x] Run `python3 -m py_compile train.py`.
- [x] Run `uv run ruff check train.py`.
- [x] Confirm `git diff -- train.py` contains only the planned `BATCH_SIZE` and `LR_MILESTONES` changes.
- [x] Confirm validation cadence remains once per epoch with `rg -n "evaluator\\.evaluate|Eval\\(" train.py`.

### Milestone 2: Launch and monitor single-GPU experiment
- [x] Query the current baseline with `bash /root/.codex/plugins/cache/deoxys/autoresearch/2.9.6/skills/shared/scripts/exp-index.sh baseline .autoresearch/experiment-indices/maximize-cifar10-best-test-accuracy.tsv`; expected baseline is `93.23`.
- [x] Check GPU availability with `nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits`.
- [x] Select one idle GPU and confirm CUDA sees only that GPU with `CUDA_VISIBLE_DEVICES=${GPU_ID} uv run python -c 'import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")'`.
- [x] Remove stale `run.log` before launch.
- [x] Launch `CUDA_VISIBLE_DEVICES=${GPU_ID} uv run train.py > run.log 2>&1`.
- [x] Monitor startup until `run.log` reports parameter count, 300s time budget, batches per epoch, and epoch evaluations.
- [x] Confirm `Batches per epoch: 520` appears in `run.log`, reflecting `BATCH_SIZE = 96` on 50,000 CIFAR-10 training images with `drop_last=True`.
- [x] Confirm the first LR drop occurs at step 26000 by checking `run.log` for `step 26000` and `lr: 0.0100`.
- [x] Confirm the second LR drop occurs at step 44000 if reached by checking `run.log` for `step 44000` and `lr: 0.0010`; if total steps are below 44000, record that schedule reachability failed and classify as no-improvement.

### Milestone 3: Capture result, verify, and clean up
- [x] Ensure the run exits before 10 minutes total wall-clock time.
- [x] Extract summary metrics with `grep "^best_test_acc:\\|^final_test_acc:\\|^final_test_loss:\\|^training_seconds:\\|^total_seconds:\\|^startup_seconds:\\|^peak_vram_mb:\\|^num_epochs:\\|^num_steps:\\|^num_params:" run.log`.
- [x] Compare `best_test_acc` to the concrete EXP-025 threshold `93.33%` (`93.23 + 0.10`).
- [x] Record verdict and metrics in `.autoresearch/logs/exp-log-025.md`; the experiment index is updated during analysis.
- [ ] On improvement, commit `train.py`, merge into `autoresearch/dev`, and skip PR creation if no remote exists.
- [x] On no-improvement, crash, or invalid result, revert `train.py` batch-size/schedule changes with `apply_patch` and return to `autoresearch/dev`.
- [x] Remove `run.log` and root `__pycache__` after metrics are captured.

## Code Changes
- **train.py**:
  - Change `BATCH_SIZE` from `128` to `96`.
  - Change `LR_MILESTONES` from `[21000, 64000]` to `[26000, 44000]`.

These changes test whether smaller-batch stochasticity and a higher optimizer-step budget improve the 28/56/112 anchor. The milestone changes keep the first drop near the middle of the expected reachable step budget and make a late second drop possible if the smaller batch increases steps enough.

## Configuration Changes
- `BATCH_SIZE`: `128` -> `96`.
- `LR_MILESTONES`: `[21000, 64000]` -> `[26000, 44000]`.
- `STAGE_WIDTHS`: unchanged at `(28, 56, 112)`.
- `WEIGHT_DECAY`: unchanged at `1e-4`.
- `LR`, `MOMENTUM`, `MAX_STEPS`, `USE_CUDNN_BENCHMARK`, `USE_CHANNELS_LAST`, `USE_COMPILE`, seed, augmentation, optimizer class, and once-per-epoch validation remain unchanged.

## Execution Environment
- Method: local command from the project root using one idle selected GPU; actual launch command is `CUDA_VISIBLE_DEVICES=${GPU_ID} uv run train.py > run.log 2>&1`.
- Resources: one NVIDIA H20 class GPU; no dependency, data, harness, or package changes.
- Estimated runtime: about 6 to 7 minutes total based on recent local runs; kill if total wall-clock exceeds 10 minutes.
- Log output: stdout and stderr captured in project-root `run.log`, then summarized into `.autoresearch/logs/exp-log-025.md` before cleanup.
- Tool skill: none; this is a local single-GPU run.

## Abort Criteria
- Stop and classify as failure if total wall-clock runtime exceeds 10 minutes.
- Stop and classify as crash if `run.log` shows an exception, CUDA OOM, missing data failure, NaN/Inf loss, or no progress output after launch.
- Stop and classify as invalid if any tracked source file other than `train.py` changes, if the fixed harness is touched, or if validation runs more than once per epoch.
- Treat the result as no-improvement if `Batches per epoch` is not 520, because the run did not test the intended batch size.
- Treat the result as no-improvement if the first LR drop does not occur at step 26000.
- Treat the result as no-improvement if the run does not reach the second LR drop at step 44000, because the schedule calibration assumption failed.

## Verification Protocol

### Verification Procedure
1. Verify baseline and threshold:
   - Command: `bash /root/.codex/plugins/cache/deoxys/autoresearch/2.9.6/skills/shared/scripts/exp-index.sh baseline .autoresearch/experiment-indices/maximize-cifar10-best-test-accuracy.tsv`
   - Pass condition: output reports `baseline=93.23`; the EXP-025 improvement threshold is `best_test_acc >= 93.33`.
   - Timeout: 30 seconds.
2. Verify scope before launch:
   - Command: `git diff --name-only`
   - Pass condition: the only tracked source diff is `train.py`; `.autoresearch/` artifacts are local-only and `data/` remains untracked.
   - Timeout: 30 seconds.
3. Verify syntax and lint:
   - Commands: `python3 -m py_compile train.py` and `uv run ruff check train.py`
   - Pass condition: both commands exit 0.
   - Timeout: 2 minutes total.
4. Verify validation cadence:
   - Command: `rg -n "evaluator\\.evaluate|Eval\\(" train.py`
   - Pass condition: one `Eval()` construction and one `evaluator.evaluate(...)` call remain, with the evaluate call in the epoch-level loop.
   - Timeout: 30 seconds.
5. Verify batch-size effect and schedule:
   - Command: `grep "Batches per epoch" run.log`; `grep "step 26000" run.log`; `grep "step 44000" run.log`.
   - Pass condition: batches per epoch is 520, step 26000 line includes `lr: 0.0100`, and step 44000 line includes `lr: 0.0010`.
   - Timeout: 30 seconds.
6. Verify experiment completion:
   - Command: `CUDA_VISIBLE_DEVICES=${GPU_ID} uv run train.py > run.log 2>&1`
   - Pass condition: process exits 0 before 10 minutes total wall-clock and prints a numeric `best_test_acc`.
   - Timeout: 10 minutes.
7. Verify metric improvement:
   - Command: `grep "^best_test_acc:" run.log`
   - Pass condition: parsed `best_test_acc` is at least `93.33%`; smaller increases over 93.23 are `no-improvement` under the goal's +0.10 percentage-point rule.
   - Timeout: 30 seconds.
8. Verify hard constraints:
   - Commands: `git diff -- train.py`, `git status --short --branch`, and `grep "^training_seconds:\\|^total_seconds:" run.log`
   - Pass condition: only the planned `BATCH_SIZE` and `LR_MILESTONES` changes were present during the run, fixed training budget was used, total wall-clock stayed under 10 minutes, and no protected files changed.
   - Timeout: 1 minute.

### Informational Metrics (Optional)
- final_test_acc: `grep "^final_test_acc:" run.log` — final evaluation accuracy.
- final_test_loss: `grep "^final_test_loss:" run.log` — final evaluation loss.
- training_seconds: `grep "^training_seconds:" run.log` — fixed-budget training time.
- total_seconds: `grep "^total_seconds:" run.log` — total runtime including startup and validation.
- peak_vram_mb: `grep "^peak_vram_mb:" run.log` — peak CUDA allocation.
- num_epochs: `grep "^num_epochs:" run.log` — epoch count completed.
- num_steps: `grep "^num_steps:" run.log` — optimization steps completed.
- num_params: `grep "^num_params:" run.log` — confirms the anchor architecture remained unchanged.
