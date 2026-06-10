# Plan EXP-071: CIFAR AutoAugment on CutMix Anchor
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-071.md

## Milestones

### Milestone 1: Code change implemented and locally checked
- [x] Create experiment branch `autoresearch/exp-071` from `autoresearch/dev`.
- [x] Modify only `train.py`, adding `transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10)` after `RandomHorizontalFlip()` and before `ToTensor()`.
- [x] Add a startup log marker for the training augmentation policy so `run.log` verifies CIFAR AutoAugment is active.
- [x] Preserve unit-std normalization `(1, 1, 1)` and all EXP-064 CutMix anchor settings.
- [x] Confirm `git diff --name-only` lists only `train.py`.
- [x] Run `python3 -m py_compile train.py`.
- [x] Run `uv run ruff check train.py`.

### Milestone 2: Experiment launched and confirmed running
- [x] Check GPU availability with `nvidia-smi`.
- [x] Remove stale `run.log` before launch.
- [x] Launch foreground local run with `CUDA_VISIBLE_DEVICES=<gpu> uv run train.py > run.log 2>&1`.
- [x] Confirm startup markers in `run.log`: CUDA device, ResNet-20 parameter count, CutMix anchor settings, CIFAR AutoAugment marker, 300s budget, and 390 batches per epoch.

### Milestone 3: Run completed and metrics captured
- [x] Monitor `run.log` for tracebacks, CUDA OOMs, non-finite losses, and missing progress.
- [x] Confirm the first LR drop occurs at step 21000 unless AutoAugment overhead prevents it; if the drop is missed, record that as a key result.
- [x] Capture final summary metrics from `run.log`, especially `best_test_acc`.
- [x] Update `logs/exp-log-071.md` with final metrics and verification results.

## Code Changes
- **train.py**:
  - Insert `transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10)` in the training transform after `transforms.RandomHorizontalFlip()` and before `transforms.ToTensor()`.
  - Add a startup print such as `Train augmentation: CIFAR AutoAugment` after the CutMix settings print.
  - Preserve `mean = (0.4914, 0.4822, 0.4465)` and `std = (1, 1, 1)`.
  - Preserve `CUTMIX_ALPHA=1.0`, `CUTMIX_PROB=0.5`, `CUTMIX_LABEL_SMOOTHING=0.05`, architecture, optimizer, LR milestones, batch size, reflection crop padding, compile/channels-last path, fixed seed, and once-per-epoch validation.

This tests whether a CIFAR-specific policy augmentation supplies complementary invariance on top of the validated CutMix anchor.

## Configuration Changes
- Training transform: add CIFAR AutoAugment policy after crop/flip and before tensor conversion.
- `CUTMIX_ALPHA`: unchanged at `1.0`.
- `CUTMIX_PROB`: unchanged at `0.5`.
- `CUTMIX_LABEL_SMOOTHING`: unchanged at `0.05`.
- Input normalization: unchanged at unit std `(1, 1, 1)` after EXP-070 showed standard CIFAR std is harmful.

## Execution Environment
- Method: local single-GPU foreground command from the project root.
- Resources: one available NVIDIA H20 class GPU; expected VRAM near prior CutMix runs, roughly 660 MB peak.
- Estimated runtime: 7-8 minutes wall-clock if AutoAugment adds CPU transform overhead; kill if total runtime exceeds 10 minutes.
- Log output: capture stdout/stderr to `run.log`; this file is the source for startup markers, LR milestone checks, final metrics, and failure diagnosis.
- Tool skill: none; local execution only. Detached background launches are unreliable in this environment, so use a foreground exec session.

## Abort Criteria
- Kill the run and classify as failure if total wall-clock time exceeds 10 minutes.
- Abort if `run.log` shows a Python traceback, CUDA OOM, non-finite loss, or repeated missing progress for more than 2 minutes after startup.
- Abort if startup markers do not show CUDA execution, CIFAR AutoAugment active, or the unchanged CutMix anchor settings.
- Treat the run as no-improvement or crash, as appropriate, if it fails to report a numeric `best_test_acc`.

## Verification Protocol

### Verification Procedure
1. Confirm branch and diff:
   - Command: `git status --short --branch && git diff -- train.py`
   - Pass condition: branch is `autoresearch/exp-071`; tracked diff is limited to `train.py`; the intended behavior change is the CIFAR AutoAugment transform insertion plus startup logging.
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
   - Command: `grep -E "Device:|ResNet-|CutMix alpha:|Train augmentation:|Time budget:|Batches per epoch:" run.log`
   - Pass condition: CUDA device is used, ResNet-20 has 822,790 params, CutMix reports `alpha: 1.0, prob: 0.5, label smoothing: 0.05`, train augmentation reports CIFAR AutoAugment, time budget is 300s, and batches per epoch is 390.
   - Timeout: 30 seconds.
5. Verify LR milestone:
   - Command: `grep "step 21000" run.log`
   - Pass condition: preferred pass is the first LR drop at step 21000 with LR 0.0100. If the run completes with final metrics but misses step 21000 because AutoAugment overhead reduces step coverage, the experiment remains valid but this condition is recorded as a throughput-related failure mode.
   - Timeout: 30 seconds.
6. Extract primary metric:
   - Command: `grep "^best_test_acc:" run.log`
   - Pass condition: a numeric `best_test_acc` is present.
   - Timeout: 30 seconds.
7. Compare against current baseline:
   - Baseline from experiment index before EXP-071: `94.11%`; required improvement threshold: `94.21%`.
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
- `num_steps`: `grep "^num_steps:" run.log` — tracks optimization-step budget under the fixed wall-clock training window.
- `num_params`: `grep "^num_params:" run.log` — confirms architecture unchanged.
