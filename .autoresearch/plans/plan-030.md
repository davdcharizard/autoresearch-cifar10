# Plan EXP-030: Reflection Anchor With 32k Second LR Drop
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-030.md

## Milestones

### Milestone 1: Implement reachable second LR drop
- [x] Create experiment branch `autoresearch/exp-030` from `autoresearch/dev`.
- [x] Modify only `train.py`.
- [x] Change `LR_MILESTONES = [21000, 64000]` to `LR_MILESTONES = [21000, 32000]`.
- [x] Preserve reflected `RandomCrop`, `STAGE_WIDTHS = (28, 56, 112)`, `BATCH_SIZE = 128`, `LR = 0.1`, `MOMENTUM = 0.9`, `WEIGHT_DECAY = 1e-4`, FP32 training, channels-last, cuDNN benchmark, `torch.compile`, seed, optimizer class, and once-per-epoch validation.
- [x] Run `python3 -m py_compile train.py`.
- [x] Run `uv run ruff check train.py`.
- [x] Confirm `git diff -- train.py` contains only the planned `LR_MILESTONES` change.
- [x] Confirm validation cadence remains once per epoch with `rg -n "evaluator\\.evaluate|Eval\\(" train.py`.
- [x] Confirm reflected crop padding is preserved with `rg -n "RandomCrop\\(32, padding=4, padding_mode=\"reflect\"\\)" train.py`.
- [x] Confirm the new schedule is present with `rg -n "LR_MILESTONES = \\[21000, 32000\\]" train.py`.

### Milestone 2: Launch and monitor single-GPU experiment
- [x] Query the current baseline with `bash /root/.codex/plugins/cache/deoxys/autoresearch/2.9.6/skills/shared/scripts/exp-index.sh baseline .autoresearch/experiment-indices/maximize-cifar10-best-test-accuracy.tsv`; expected baseline is `93.58`, so the improvement threshold is `93.68`.
- [x] Check GPU availability with `nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits`.
- [x] Select one idle GPU and confirm CUDA sees only that GPU with `CUDA_VISIBLE_DEVICES=${GPU_ID} uv run python -c 'import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")'`.
- [x] Remove stale `run.log` before launch.
- [x] Launch `CUDA_VISIBLE_DEVICES=${GPU_ID} uv run train.py > run.log 2>&1`.
- [x] Monitor startup until `run.log` reports parameter count, 300s time budget, batches per epoch, and epoch evaluations.
- [x] Confirm `Batches per epoch: 390` appears in `run.log`, reflecting preserved `BATCH_SIZE = 128`.
- [x] Confirm the first LR drop occurs at step 21000 with `lr: 0.0100`.
- [x] Confirm the second LR drop occurs at step 32000 with `lr: 0.0010`.

### Milestone 3: Capture result, verify, and clean up
- [x] Ensure the run exits before 10 minutes total wall-clock time.
- [x] Extract summary metrics with `grep "^best_test_acc:\\|^final_test_acc:\\|^final_test_loss:\\|^training_seconds:\\|^total_seconds:\\|^startup_seconds:\\|^peak_vram_mb:\\|^num_epochs:\\|^num_steps:\\|^num_params:" run.log`.
- [x] Compare `best_test_acc` to the concrete EXP-030 threshold `93.68%` (`93.58 + 0.10`).
- [x] Record verdict and metrics in `.autoresearch/logs/exp-log-030.md`; the experiment index is updated during analysis.
- [ ] On improvement, commit `train.py`, merge into `autoresearch/dev`, and skip PR creation if no remote exists.
- [x] On no-improvement, crash, or invalid result, revert the `LR_MILESTONES` change with `apply_patch` and return to `autoresearch/dev`.
- [x] Remove `run.log` and root `__pycache__` after metrics are captured.

## Code Changes
- **train.py**:
  - Change `LR_MILESTONES` from `[21000, 64000]` to `[21000, 32000]`.
  - Leave reflected crop padding and all architecture, optimizer, batch-size, compile, seed, and validation settings unchanged.

This tests whether the EXP-029 reflection-padding anchor benefits from a reachable LR 0.001 refinement phase after the observed LR 0.01 accuracy oscillation.

## Configuration Changes
- `LR_MILESTONES`: `[21000, 64000]` -> `[21000, 32000]`.
- No augmentation, model, optimizer, weight-decay, batch-size, seed, dependency, or harness configuration changes.

## Execution Environment
- Method: local command from the project root using one idle selected GPU; actual launch command is `CUDA_VISIBLE_DEVICES=${GPU_ID} uv run train.py > run.log 2>&1`.
- Resources: one NVIDIA H20 class GPU; no dependency, data, harness, or package changes.
- Estimated runtime: about 6 to 7 minutes total based on recent local runs; kill if total wall-clock exceeds 10 minutes.
- Log output: stdout and stderr captured in project-root `run.log`, then summarized into `.autoresearch/logs/exp-log-030.md` before cleanup.
- Tool skill: none; this is a local single-GPU run.

## Abort Criteria
- Stop and classify as failure if total wall-clock runtime exceeds 10 minutes.
- Stop and classify as crash if `run.log` shows an exception, CUDA OOM, missing data failure, NaN/Inf loss, or no progress output after launch.
- Stop and classify as invalid if any tracked source file other than `train.py` changes, if the fixed harness is touched, or if validation runs more than once per epoch.
- Treat the result as invalid if `num_params` differs from `822,790`, because the experiment is intended to change schedule only.
- Treat the result as no-improvement if `Batches per epoch` is not 390, the first LR drop does not occur at step 21000, or the second LR drop does not occur at step 32000.

## Verification Protocol

### Verification Procedure
1. Verify baseline and threshold:
   - Command: `bash /root/.codex/plugins/cache/deoxys/autoresearch/2.9.6/skills/shared/scripts/exp-index.sh baseline .autoresearch/experiment-indices/maximize-cifar10-best-test-accuracy.tsv`
   - Pass condition: output reports `baseline=93.58`; the EXP-030 improvement threshold is `best_test_acc >= 93.68`.
   - Timeout: 30 seconds.
2. Verify scope before launch:
   - Command: `git diff --name-only`
   - Pass condition: the only tracked source diff is `train.py`; `.autoresearch/` artifacts are local-only and `data/` remains untracked.
   - Timeout: 30 seconds.
3. Verify syntax and lint:
   - Commands: `python3 -m py_compile train.py` and `uv run ruff check train.py`
   - Pass condition: both commands exit 0.
   - Timeout: 2 minutes total.
4. Verify validation cadence and preserved augmentation:
   - Commands: `rg -n "evaluator\\.evaluate|Eval\\(" train.py` and `rg -n "RandomCrop\\(32, padding=4, padding_mode=\"reflect\"\\)" train.py`
   - Pass condition: one `Eval()` construction, one epoch-level `evaluator.evaluate(...)` call, and reflected crop padding remains present.
   - Timeout: 30 seconds.
5. Verify schedule implementation:
   - Command: `rg -n "LR_MILESTONES = \\[21000, 32000\\]" train.py`
   - Pass condition: output shows the planned milestones.
   - Timeout: 30 seconds.
6. Verify preserved batch size, schedule behavior, and parameter count:
   - Commands: `grep "Batches per epoch" run.log`; `grep "step 21000" run.log`; `grep "step 32000" run.log`; `grep "^num_params:" run.log`
   - Pass condition: batches per epoch is 390, step 21000 includes `lr: 0.0100`, step 32000 includes `lr: 0.0010`, and `num_params` is `822,790`.
   - Timeout: 30 seconds.
7. Verify experiment completion:
   - Command: `CUDA_VISIBLE_DEVICES=${GPU_ID} uv run train.py > run.log 2>&1`
   - Pass condition: process exits 0 before 10 minutes total wall-clock and prints a numeric `best_test_acc`.
   - Timeout: 10 minutes.
8. Verify metric improvement:
   - Command: `grep "^best_test_acc:" run.log`
   - Pass condition: parsed `best_test_acc` is at least `93.68%`; smaller increases over 93.58 are `no-improvement` under the goal's +0.10 percentage-point rule.
   - Timeout: 30 seconds.
9. Verify hard constraints:
   - Commands: `git diff -- train.py`, `git status --short --branch`, and `grep "^training_seconds:\\|^total_seconds:" run.log`
   - Pass condition: only the planned schedule diff was present during the run, fixed training budget was used, total wall-clock stayed under 10 minutes, and no protected files changed.
   - Timeout: 1 minute.

### Informational Metrics (Optional)
- final_test_acc: `grep "^final_test_acc:" run.log` — final evaluation accuracy.
- final_test_loss: `grep "^final_test_loss:" run.log` — final evaluation loss.
- training_seconds: `grep "^training_seconds:" run.log` — fixed-budget training time.
- total_seconds: `grep "^total_seconds:" run.log` — total runtime including startup and validation.
- startup_seconds: `grep "^startup_seconds:" run.log` — startup and compile overhead.
- peak_vram_mb: `grep "^peak_vram_mb:" run.log` — peak CUDA allocation.
- num_epochs: `grep "^num_epochs:" run.log` — epoch count completed.
- num_steps: `grep "^num_steps:" run.log` — optimization-step budget.
- num_params: `grep "^num_params:" run.log` — confirms the anchor architecture remained unchanged.
