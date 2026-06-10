# Plan EXP-069: Post-Drop CutMix Probability Taper to 0.25
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-069.md

## Milestones

### Milestone 1: Code change implemented and locally checked
- [x] Create experiment branch `autoresearch/exp-069` from `autoresearch/dev`.
- [x] Change only `train.py`, adding a post-drop CutMix probability taper from 0.5 to 0.25 after step 21000.
- [x] Add startup and one-time taper log markers so the schedule is auditable in `run.log`.
- [x] Confirm `git diff --name-only` lists only `train.py`.
- [x] Run `python3 -m py_compile train.py`.
- [x] Run `uv run ruff check train.py`.

### Milestone 2: Experiment launched and confirmed running
- [x] Check GPU availability with `nvidia-smi`.
- [x] Remove stale `run.log` before launch.
- [x] Launch foreground local run with `CUDA_VISIBLE_DEVICES=<gpu> uv run train.py > run.log 2>&1`.
- [x] Confirm startup markers in `run.log`: CUDA device, ResNet-20 parameter count, `CutMix alpha: 1.0, prob: 0.5, label smoothing: 0.05`, post-drop probability `0.25`, 300s budget, and 390 batches per epoch.

### Milestone 3: Run completed and metrics captured
- [x] Monitor `run.log` for tracebacks, CUDA OOMs, non-finite losses, and missing progress.
- [x] Confirm the first LR drop occurs at step 21000 and the CutMix taper marker appears.
- [x] Capture final summary metrics from `run.log`, especially `best_test_acc`.
- [x] Update `logs/exp-log-069.md` with final metrics and verification results.

## Code Changes
- **train.py**:
  - Add `CUTMIX_POST_DROP_PROB = 0.25`.
  - Add `CUTMIX_TAPER_STEP = LR_MILESTONES[0]`.
  - Print the taper configuration during startup.
  - In the training loop, compute `current_cutmix_prob = CUTMIX_POST_DROP_PROB if step >= CUTMIX_TAPER_STEP else CUTMIX_PROB` before the CutMix sampling decision.
  - Print a one-time marker when `step == CUTMIX_TAPER_STEP`, confirming the next updates use the reduced CutMix probability.

This tests whether persistent CutMix noise during low-LR refinement is limiting the validated `alpha=1.0, p=0.5` anchor, while preserving all architecture, optimizer, LR schedule, reflection padding, label smoothing, weight decay, compile/channels-last, and validation-cadence settings.

## Configuration Changes
- `CUTMIX_ALPHA`: unchanged at `1.0`.
- `CUTMIX_PROB`: unchanged at `0.5` before the first LR drop.
- `CUTMIX_POST_DROP_PROB`: new value `0.25` after step 21000.
- `CUTMIX_TAPER_STEP`: new value `21000`, tied to `LR_MILESTONES[0]`.

Rationale: EXP-064 validated `p=0.5`, EXP-065 showed full-run `p=0.25` remains close, and EXP-067/068 bracketed static alpha as worse. A post-drop taper is the smallest remaining test of temporal CutMix strength.

## Execution Environment
- Method: local single-GPU foreground command from the project root.
- Resources: one available NVIDIA H20 class GPU; expected VRAM near prior CutMix runs, roughly 660 MB peak.
- Estimated runtime: about 6-7 minutes wall-clock including startup and validation; kill if total runtime exceeds 10 minutes.
- Log output: capture stdout/stderr to `run.log`; this file is the source for startup markers, progress, taper marker, final metrics, and failure diagnosis.
- Tool skill: none; local execution only. Prior infrastructure notes say detached background launches are unreliable, so use a foreground exec session.

## Abort Criteria
- Kill the run and classify as failure if total wall-clock time exceeds 10 minutes.
- Abort if `run.log` shows a Python traceback, CUDA OOM, non-finite loss, or repeated missing progress for more than 2 minutes after startup.
- Abort if startup markers do not show `CUTMIX_ALPHA=1.0`, pre-drop `CUTMIX_PROB=0.5`, post-drop probability `0.25`, or CUDA execution.
- Treat the run as no-improvement or crash, as appropriate, if it fails to report a numeric `best_test_acc`.

## Verification Protocol

### Verification Procedure
1. Confirm branch and diff:
   - Command: `git status --short --branch && git diff -- train.py`
   - Pass condition: branch is `autoresearch/exp-069`; tracked diff is limited to `train.py`; the intended code change is the CutMix post-drop probability taper.
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
   - Command: `grep -E "Device:|ResNet-|CutMix alpha:|post-drop|Time budget:|Batches per epoch:" run.log`
   - Pass condition: CUDA device is used, ResNet-20 has 822,790 params, CutMix line reports `alpha: 1.0, prob: 0.5, label smoothing: 0.05`, taper line reports post-drop probability 0.25 after step 21000, time budget is 300s, and batches per epoch is 390.
   - Timeout: 30 seconds.
5. Verify LR and taper milestone:
   - Command: `grep "step 21000" run.log && grep "CutMix probability tapered" run.log`
   - Pass condition: log shows the first LR drop at step 21000 with LR 0.0100 and a taper marker indicating post-drop probability 0.25.
   - Timeout: 30 seconds.
6. Extract primary metric:
   - Command: `grep "^best_test_acc:" run.log`
   - Pass condition: a numeric `best_test_acc` is present.
   - Timeout: 30 seconds.
7. Compare against current baseline:
   - Baseline from experiment index before EXP-069: `94.11%`; required improvement threshold: `94.21%`.
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
