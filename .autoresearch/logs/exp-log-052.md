# EXP-052: Hybrid Post-Drop Cosine LR Tail

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-052.md
- **Plan**: plans/plan-052.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-052
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary

Implemented the planned hybrid post-drop cosine LR tail in `train.py`. The patch removes `MultiStepLR`, adds explicit tail constants and a small `lr_after_step` helper, and updates optimizer param-group LRs after each completed optimizer step. The first 21000-step high-LR phase is preserved, step 21000 logs `lr: 0.0100`, and later steps decay smoothly toward a `0.0020` floor.

### Surprises & Discoveries

The manual LR helper required moving `step += 1` before LR logging so the logged LR reflects the schedule state after the completed step, matching the previous `MultiStepLR` log semantics at the first drop. No other code structure surprises were encountered.

### Decisions

- Used a nonzero `TAIL_MIN_LR = 0.002` to avoid recreating the failed abrupt second-drop-to-0.001 behavior.
- Set `TAIL_END_STEP = 42000` because recent clean runs finish near 40k-42k steps, making the tail reach the floor close to the end of the fixed training budget.
- Kept the optimizer family, batch size, model, data transforms, weight decay, label smoothing, compile, channels-last, and once-per-epoch validation unchanged.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local foreground process on GPU0, session 10552, shell PID 3316555, uv PID 3316556, main python PID 3316559
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log`
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-09 16:33:19 UTC
- **Ended**: 2026-06-09 16:40 UTC

Description:
- Run the EXP-052 hybrid post-drop cosine LR tail on a single local GPU with output captured to `run.log`. The run tests whether preserving the validated 21k-step first drop while smoothing the LR 0.01 tail toward a 0.002 floor can improve late plateau behavior. Expected behavior is unchanged parameter count, clean startup, `lr: 0.0100` logged at step 21000, later LR values smoothly below 0.0100, and final `best_test_acc` compared against the 94.07% threshold.

Observations:
- Preflight passed: tracked diff is limited to `train.py`, `python3 -m py_compile train.py` exited 0, and `uv run ruff check train.py` reported `All checks passed!`.
- Baseline check passed before launch: `baseline=93.97`, `baseline_commit=755be2c`, so the improvement threshold is `94.07%`.
- 2026-06-09 16:33 UTC: Both GPUs were idle at launch; GPU0 selected with 0 MiB allocated and 0% utilization. Stale `run.log` was removed immediately before launch.
- 2026-06-09 16:33 UTC: Foreground session launched on GPU0. Startup log confirms CUDA, `num_params=822,790`, 300s budget, and 390 batches per epoch.
- 2026-06-09 16:35 UTC: Early training healthy through epoch 22 with best test accuracy 85.43%, no traceback/OOM patterns, and normal 6-8ms step timing. LR remains at 0.1000 before the planned step-21000 transition.
- 2026-06-09 16:37 UTC: First LR transition confirmed. The progress line at `step 21000` reports `lr: 0.0100`, matching the anchor first-drop behavior. Post-drop convergence reached 93.61% by epoch 61, with no traceback/OOM/nan/inf patterns observed.
- 2026-06-09 16:39 UTC: Cosine tail behavior confirmed: `step 25000` logged `lr: 0.0093`, `step 30000` logged `lr: 0.0069`, `step 35000` logged `lr: 0.0040`, `step 40000` logged `lr: 0.0022`, and steps 40950-41300 logged the planned `0.0020` floor.
- 2026-06-09 16:40 UTC: Run exited cleanly with final `best_test_acc=93.87%`, below the 94.07% improvement threshold. This is a valid no-improvement result, not a crash.

Key Metrics:
- `best_test_acc`: 93.87%
- `final_test_acc`: 93.48%
- `final_test_loss`: 0.2327
- `training_seconds`: 300.0
- `total_seconds`: 404.1
- `startup_seconds`: 2.2
- `peak_vram_mb`: 660.4
- `num_epochs`: 107
- `num_steps`: 41,571
- `num_params`: 822,790
- Verdict versus baseline 93.97% / threshold 94.07%: no-improvement (`-0.10pp` vs baseline, `-0.20pp` vs threshold).

## Verification Results

### Conditions Checked
- Baseline check: `exp-index.sh baseline ...` reported `baseline=93.97`, `baseline_commit=755be2c`, `total_experiments=53`, `improvements=9` — pass.
- Scope check: `git diff --name-only` reported only `train.py` among tracked files — pass.
- Compile check: `python3 -m py_compile train.py` exited 0 — pass.
- Style check: `uv run ruff check train.py` reported `All checks passed!` — pass.
- Run completion: foreground process exited 0 and final summary metrics were present in `run.log` — pass.
- First LR drop: `grep "step 21000" run.log` found `lr: 0.0100` — pass.
- Cosine tail: progress lines showed `lr: 0.0093` near step 25000, `0.0069` at step 30000, `0.0040` at step 35000, `0.0022` at step 40000, and `0.0020` near the end — pass.
- Parameter count: final summary reported `num_params: 822,790` — pass.
- Error scan: `grep -n "Traceback\|CUDA out of memory\|RuntimeError\| nan\| inf" run.log` returned no matches — pass.

### Informational Metrics
- Best epoch-level result occurred at epoch 85 with `test_acc=93.87%`.
- The smooth tail did not improve on the 93.97% baseline despite preserving the validated first drop and reaching the nonzero LR floor.

## Errors & Dead Ends

## Human Notes

> No human interventions during autopilot execution.
