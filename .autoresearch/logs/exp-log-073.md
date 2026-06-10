# EXP-073: Short Clean Warmup Before CutMix

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-073.md
- **Plan**: plans/plan-073.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-073
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary

EXP-073 implements the approved short clean warmup before CutMix. The branch `autoresearch/exp-073` was created from `autoresearch/dev`, and `train.py` was the only tracked file changed. The implementation adds `CUTMIX_WARMUP_STEPS = 2000`, gates CutMix sampling with `step >= CUTMIX_WARMUP_STEPS`, and keeps the existing clean label-smoothed loss for the first 2000 updates. After warmup, the existing `CUTMIX_ALPHA=1.0`, `CUTMIX_PROB=0.5`, and `CUTMIX_LABEL_SMOOTHING=0.05` behavior is restored.

### Surprises & Discoveries

No implementation surprises. The training loop already had the global `step` available before CutMix sampling, so the warmup gate could be added without touching optimizer, scheduler, dataloader, evaluator, architecture, or transforms.

### Decisions

Used `step >= CUTMIX_WARMUP_STEPS` before the per-batch update, which makes updates 1-2000 clean and enables CutMix starting with update 2001. Added both a startup marker and a one-time runtime marker so the log verifies the intended schedule.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local foreground session 48310 on GPU0
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log
- **WandB**: N/A
- **Status**: running
- **Started**: 2026-06-09 21:39
- **Ended**: 2026-06-09 21:46

Description:
- This run tests whether disabling CutMix for the first 2000 updates improves early feature formation while preserving the validated static CutMix recipe afterward. It keeps architecture, optimizer, schedule, reflection crop padding, unit-std normalization, batch size, full-run label smoothing, and evaluation unchanged. The active baseline is 94.11%, so the required improvement threshold is 94.21%.

Observations:
- Preflight scope check passed: `git diff --name-only` listed only `train.py`.
- Preflight syntax check passed: `python3 -m py_compile train.py` exited 0.
- Preflight style check passed: `uv run ruff check train.py` reported `All checks passed!`.
- GPU availability check found both H20 GPUs idle; Run 1 launched on GPU0 with `CUDA_VISIBLE_DEVICES=0`.
- Startup confirmed the expected anchor and warmup: `Device: cuda`, `CutMix alpha: 1.0, prob: 0.5, label smoothing: 0.05`, and `CutMix warmup steps: 2000` (source: run.log L1-L4).
- The runtime marker printed immediately after `step 02000`, confirming CutMix was disabled for the first 2000 clean updates and enabled from update 2001 onward (source: run.log L17-L18).
- Early clean warmup produced lower pre-CutMix training loss and reached `test_acc: 78.06%` by epoch 5; after CutMix enabled, epoch 6 reached `81.41%` (source: run.log L16-L20).
- The first LR drop was reached at `step 21000` in epoch 54 with `lr: 0.0100`; post-drop convergence climbed from `test_acc: 91.81%` at epoch 54 to `93.64%` by epoch 60 (source: run.log L115-L128).
- The run peaked at `94.14%` on epoch 69, then stayed below the 94.21% improvement threshold through the fixed-budget finish (source: run.log L146-L223).

Key Metrics:
- `best_test_acc`: 94.14%
- `final_test_acc`: 93.05%
- `final_test_loss`: 0.2956
- `training_seconds`: 300.0
- `total_seconds`: 393.1
- `startup_seconds`: 1.9
- `peak_vram_mb`: 660.4
- `num_epochs`: 102
- `num_steps`: 39399
- `num_params`: 822,790

## Verification Results

### Conditions Checked
- Scope check: passed. `git diff --name-only` listed only `train.py`.
- Syntax check: passed. `python3 -m py_compile train.py` exited 0.
- Style check: passed. `uv run ruff check train.py` reported `All checks passed!`.
- Warmup marker check: passed. `run.log` included `CutMix warmup steps: 2000` and `CutMix enabled after 2000 clean steps`.
- First LR drop check: passed. `run.log` showed `step 21000` with `lr: 0.0100`.
- Metric check: passed. The run completed and reported numeric `best_test_acc: 94.14%`.
- Improvement check: failed. Current baseline is 94.11%, and the goal requires `best_test_acc >= 94.21%`; 94.14% is only +0.03pp over baseline and does not clear the noise guard.
- Verdict: `no-improvement`.

### Informational Metrics
- Peak accuracy occurred early in the post-drop phase at epoch 69; later epochs did not recover above 94.14%.
- Step coverage was 39,399 steps, comparable to other CutMix-anchor runs and sufficient for the first LR drop.
- VRAM stayed at 660.4 MB, matching the existing CutMix anchor footprint.

## Errors & Dead Ends
- No runtime errors, CUDA errors, import errors, NaNs, or infrastructure interruptions were observed.
- The mechanism produced a small positive signal over baseline but remained inside the explicit +0.10pp noise guard.

## Human Notes

> Autopilot execution; no human intervention during implementation.
