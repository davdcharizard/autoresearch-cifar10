# EXP-062: Compact ResNet-14 with Moderate Width Increase

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-062.md
- **Plan**: plans/plan-062.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-062
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary

Implemented the planned compact architecture change in `train.py`. The patch changes `NUM_BLOCKS` from 3 to 2, making the startup model report `ResNet-14`, and changes `STAGE_WIDTHS` from `(28, 56, 112)` to `(32, 64, 128)`. All optimizer, scheduler, augmentation, loss, compile/channels-last, evaluation, and timing behavior remains unchanged.

### Surprises & Discoveries

No code-structure surprises. The existing implementation already derives model depth from `NUM_BLOCKS` and channel sizes from `STAGE_WIDTHS`, so the depth/width tradeoff is implemented as two constant edits without touching `BasicBlock`, shortcuts, loaders, or the training loop.

### Decisions

- Kept the first LR milestone at step 21000 rather than retuning it so the experiment isolates the compact architecture tradeoff.
- Kept batch size, weight decay, label smoothing, and crop padding fixed to preserve the current anchor recipe and make attribution clean.
- Used local foreground execution because prior detached launches in this shell can be reaped before producing final metrics.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local session 34476; shell PID 3572012; uv PID 3572013; main python PID 3572016
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log`
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-09 18:58:28 UTC
- **Ended**: 2026-06-09 19:07:22 UTC

Description:
- Local foreground run of EXP-062 on one selected GPU with output captured to `run.log`. This tests whether reducing to two residual blocks per stage while widening to `(32, 64, 128)` improves the fixed-budget capacity/throughput tradeoff. Expected behavior is startup reporting `ResNet-14`, unchanged batch geometry, a numeric new parameter count, the first LR drop at step 21000, and final `best_test_acc` classified against the 94.07% improvement threshold.

Observations:
- Preflight passed: tracked diff is limited to `train.py`, `python3 -m py_compile train.py` exited 0, and `uv run ruff check train.py` reported `All checks passed!`.
- Baseline for classification: `93.97%`; improvement threshold: `94.07%`.
- 2026-06-09 18:58 UTC: GPU0 selected after `nvidia-smi` showed GPU0 at `0MiB` and `0%` utilization; GPU1 was already running an unrelated process in another checkout.
- 2026-06-09 18:58 UTC: Foreground run launched on GPU0. Process table showed shell PID 3572012, uv PID 3572013, and main Python PID 3572016; `/proc/3572016/cwd` verified this project root.
- Startup confirmed CUDA, `ResNet-14 | params: 685,994`, 300s budget, and `Batches per epoch: 390`.
- 2026-06-09 18:59 UTC: Early training healthy through epoch 13 with best test accuracy 83.57%, mostly 5-6ms batch timings, no traceback/OOM/runtime-error patterns, and GPU0 active.
- 2026-06-09 19:00 UTC: Mid-run pre-drop progress healthy through epoch 30. Best reached 88.54% at epoch 28; batch timing remains mostly 5-6ms, confirming the compact model is faster than the anchor pre-drop.
- 2026-06-09 19:02 UTC: First LR drop confirmed in `run.log` at `step 21000 ep 54` with `lr: 0.0100`. Pre-drop best was 88.54%, and post-drop refinement reached 93.51% by epoch 60.
- 2026-06-09 19:07 UTC: Run exited cleanly with final summary metrics. The compact ResNet-14 completed 132 epochs / 51,471 steps within the 300s training budget, but peak accuracy remained 93.51% and late training decayed to final accuracy 92.55%.

Key Metrics:
- `best_test_acc`: 93.51%
- `final_test_acc`: 92.55%
- `final_test_loss`: 0.2683
- `training_seconds`: 300.0
- `total_seconds`: 429.8
- `startup_seconds`: 2.3
- `peak_vram_mb`: 669.9
- `num_epochs`: 132
- `num_steps`: 51,471
- `num_params`: 685,994
- Verdict for execution: valid no-improvement because 93.51% is below both the 93.97% baseline and the 94.07% improvement threshold.

## Verification Results

### Conditions Checked
- Baseline check: `exp-index.sh baseline` reported `baseline=93.97`, `baseline_commit=755be2c`, `total_experiments=63`, `improvements=9`; pass.
- Scoped diff check: `git diff --name-only` listed only `train.py`; pass.
- Compile check: `python3 -m py_compile train.py` exited 0; pass.
- Style check: `uv run ruff check train.py` reported `All checks passed!`; pass.
- Execution summary check: `run.log` contains numeric final metrics including `best_test_acc: 93.51%`; pass.
- Model-depth check: `run.log` reports `ResNet-14 | params: 685,994`; pass.
- Batch-geometry check: `run.log` reports `Batches per epoch: 390`; pass.
- LR-drop check: `run.log` contains `step 21000 ep 54 ... lr: 0.0100`; pass.
- Error scan: `rg -n "Traceback|CUDA out of memory|RuntimeError|\bnan\b|\binf\b" run.log` returned no matches; pass.
- Classification check: valid run, but `93.51% < 94.07%`; classified as no-improvement.

### Informational Metrics
- Parameter count decreased from the 822,790-param anchor to 685,994 while throughput increased enough to reach 132 epochs / 51,471 steps.
- The extra optimization steps did not compensate for the depth reduction; the best accuracy plateaued at 93.51% and the final checkpoint decayed to 92.55%.

## Errors & Dead Ends
- None.

## Human Notes

> Autopilot mode; no human approval or intervention requested during execution.
