# Plan EXP-067: CutMix Alpha 0.5
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-067.md

## Milestones

### Milestone 1: Code change implemented and locally checked
- [x] Create experiment branch `autoresearch/exp-067` from `autoresearch/dev`.
- [x] Change only `train.py`, setting `CUTMIX_ALPHA = 0.5`.
- [x] Confirm `git diff --name-only` lists only `train.py`.
- [x] Run `python3 -m py_compile train.py`.
- [x] Run `uv run ruff check train.py`.

### Milestone 2: Experiment launched and confirmed running
- [x] Check GPU availability with `nvidia-smi`.
- [x] Remove stale `run.log` before launch.
- [x] Launch foreground local run with `CUDA_VISIBLE_DEVICES=<gpu> uv run train.py > run.log 2>&1`.
- [x] Confirm startup markers in `run.log`: CUDA device, ResNet-20 parameter count, `CutMix alpha: 0.5, prob: 0.5, label smoothing: 0.05`, 300s budget, and 390 batches per epoch.

### Milestone 3: Run completed and metrics captured
- [x] Monitor `run.log` for tracebacks, CUDA OOMs, non-finite losses, and missing progress.
- [x] Confirm the first LR drop occurs at step 21000.
- [x] Capture final summary metrics from `run.log`, especially `best_test_acc`.
- [x] Update `logs/exp-log-067.md` with final metrics and verification results.

## Code Changes
- **train.py**: Change `CUTMIX_ALPHA` from `1.0` to `0.5`. This tests the lower-alpha patch-area distribution while preserving the validated EXP-064 CutMix probability, endpoint smoothing, architecture, optimizer, LR schedule, augmentation, compile path, and validation cadence.

## Configuration Changes
- `CUTMIX_ALPHA`: `1.0` -> `0.5` (tests more variable CutMix lambda and patch-area samples after probability brackets around `p=0.5` failed).

## Execution Environment
- Method: local single-GPU foreground command from the project root.
- Resources: one available NVIDIA H20 class GPU; expected VRAM similar to prior CutMix runs, around 660 MB peak.
- Estimated runtime: about 6-7 minutes wall-clock including startup and validation; kill if total runtime exceeds 10 minutes.
- Log output: capture stdout/stderr to `run.log`; this file is the source for startup markers, progress, final metrics, and failure diagnosis.
- Tool skill: none; local execution only. Prior infrastructure notes say detached background launches are unreliable, so use a foreground exec session.

## Abort Criteria
- Kill the run and classify as failure if total wall-clock time exceeds 10 minutes.
- Abort if `run.log` shows a Python traceback, CUDA OOM, non-finite loss, or repeated missing progress for more than 2 minutes after startup.
- Abort if startup markers do not show `CUTMIX_ALPHA=0.5` and `CUTMIX_PROB=0.5`, or if the run is not using CUDA.
- Treat the run as no-improvement or crash, as appropriate, if it fails to report a numeric `best_test_acc`.

## Verification Protocol

### Verification Procedure
1. Confirm branch and diff:
   - Command: `git status --short --branch && git diff -- train.py`
   - Pass condition: branch is `autoresearch/exp-067`; tracked diff is limited to `train.py`; the only intended code change is `CUTMIX_ALPHA = 0.5`.
   - Timeout: 30 seconds.
2. Confirm syntax and lint:
   - Command: `python3 -m py_compile train.py`
   - Pass condition: exits with status 0.
   - Timeout: 30 seconds.
   - Command: `uv run ruff check train.py`
   - Pass condition: exits with status 0.
   - Timeout: 60 seconds.
3. Run the experiment:
   - Command: `CUDA_VISIBLE_DEVICES=<gpu> uv run train.py > run.log 2>&1`
   - Pass condition: process exits cleanly within 10 minutes and `run.log` contains final summary metrics.
   - Timeout: 10 minutes total wall-clock.
4. Verify startup and scope markers:
   - Command: `grep -E "Device:|ResNet-|CutMix alpha:|Time budget:|Batches per epoch:" run.log`
   - Pass condition: CUDA device is used, ResNet-20 has 822,790 params, CutMix line reports `alpha: 0.5, prob: 0.5, label smoothing: 0.05`, time budget is 300s, and batches per epoch is 390.
   - Timeout: 30 seconds.
5. Verify LR milestone:
   - Command: `grep "step 21000" run.log`
   - Pass condition: log shows the first LR drop at step 21000 with LR 0.0100.
   - Timeout: 30 seconds.
6. Extract primary metric:
   - Command: `grep "^best_test_acc:" run.log`
   - Pass condition: a numeric `best_test_acc` is present.
   - Timeout: 30 seconds.
7. Compare against current baseline:
   - Baseline from experiment index before EXP-067: `94.11%`; required improvement threshold: `94.21%`.
   - Pass condition for `improvement`: `best_test_acc >= 94.21`.
   - If `best_test_acc < 94.21` but the run is otherwise valid, classify as `no-improvement`.
8. Confirm hard constraints:
   - Command: `git diff --name-only`
   - Pass condition: only `train.py` is modified; `prepare.py`, dependency files, and evaluation harness remain unchanged.
   - Timeout: 30 seconds.

### Informational Metrics (Optional)
- `final_test_acc`: `grep "^final_test_acc:" run.log` — final checkpoint accuracy context.
- `final_test_loss`: `grep "^final_test_loss:" run.log` — loss/calibration context.
- `training_seconds`: `grep "^training_seconds:" run.log` — confirms fixed training budget.
- `total_seconds`: `grep "^total_seconds:" run.log` — monitors startup and validation overhead.
- `peak_vram_mb`: `grep "^peak_vram_mb:" run.log` — tracks VRAM soft constraint.
- `num_epochs`: `grep "^num_epochs:" run.log` — tracks epoch throughput.
- `num_steps`: `grep "^num_steps:" run.log` — tracks optimization-step budget.
- `num_params`: `grep "^num_params:" run.log` — confirms architecture unchanged.
