# EXP-010: SiLU (Swish) activation in place of ReLU

## Execution

Overall Status & Info:
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-010.md
- **Plan**: plans/plan-010.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-010
- **Commit**: (none — no-improvement, changes discarded)
- **PR**: (none — no-improvement)
- **Outcome**: completed (clean run; verification cond 2 failed → no-improvement verdict in analyze)

## Implementation Notes

### Summary
Two edits to `train.py` only (Milestone 1): (1) replaced all 3 `F.relu` sites with `F.silu` (BasicBlock post-bn1 +
post-residual, ResNet stem); (2) added `compiled_model = torch.compile(model, mode="reduce-overhead")` after the
model is on device + num_params printed, routing the training forward through it; eval unchanged on eager `model`.
Parse-clean, ruff clean, diff train.py-only, 0 relu / 3 silu, seed 42. Param count to be confirmed at runtime
(expected 4,299,866 — UNCHANGED, since SiLU is parameter-free; a key sanity check that only the nonlinearity changed).

### Surprises & Discoveries
- (none at implementation time — trivial elementwise swap on the validated EXP-007/008 compile pattern.)

### Decisions
- **Compile enabler included from the start** (not gated on a dt check): SiLU adds an extra elementwise sigmoid+mul
  per activation, which on this launch-bound net could mildly cut epochs and confound a fair test (the EXP-008/SE
  trap). Compile absorbs it and keeps the converged ~80–89 epoch budget. EXP-007 showed compiled-k4 = 95.92 ≈
  baseline (null standalone effect), so any gain over ~96.0 is attributable to SiLU.

## Experimental Adjustments

(none yet)

## Run Log

### Run 1

Metadata:
- **Job ID**: (PID; local background run)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-opus-4-8/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-08
- **Ended**: 2026-06-08 (exit 0)

Description:
- Running the converged EXP-003 k=4 recipe with ReLU→SiLU on all 3 activation sites, compiled (reduce-overhead).
  Tests whether a smoother nonlinearity adds generalization at fixed capacity/budget. Expect params 4,299,866
  (unchanged), clean compile, ~80–89 epochs, clean run < 600s. Success = best_test_acc ≥ 96.10.

Observations:
- Clean startup; **params 4,299,866 (UNCHANGED vs k=4 baseline)** — confirms the parameter-free swap; only the
  nonlinearity changed (run.log head). Clean compile, no traceback.
- **Steady-state dt = 9ms/step (~14,785 img/s)** from step 50 — essentially identical to compiled-k4's 8ms
  (EXP-007). SiLU's extra elementwise sigmoid is negligible and the compile enabler absorbed it. Tracking toward
  ~85–89 epochs ⇒ a FAIR, fully-converged test (SiLU isolated, no epoch-starvation confound). (run.log ep1.)
- Loss decreasing normally (2.36 @ step 50), no NaN.

Key Metrics:
- best_test_acc: **95.73%** @ ep ~82 — BELOW baseline 96.00 (−0.27pp) and bar 96.10 (run.log summary).
- **num_epochs: 85** / num_steps 32,778 — fair, fully-converged test (eval count 85 = num_epochs ⇒ eval once/epoch);
  far past the ~77 convergence point, so SiLU got a clean, non-starved shot.
- final_test_acc 95.67; **final_test_loss 0.2136** (≈ compiled-k4's 0.208, ≈ EXP-003's 0.204) — SiLU did not lower
  the loss; late evals plateaued 95.63–95.73 (ep 83–85).
- num_params 4,299,866 (UNCHANGED — parameter-free swap confirmed). peak_vram 548.7 MB. dt 9ms (~14,785 img/s).
- SiLU-k4 (95.73) ≈ compiled-k4 (95.92, EXP-007) within the ~0.2pp noise band → **SiLU added no accuracy** (if
  anything marginally below). The nonlinearity axis is non-binding for this model.

## Verification Results

### Conditions Checked

- **Cond 1 — clean completion within budget**: PASS. best_test_acc 95.73% present, total_seconds 400.3 < 600, no
  traceback (run.log; tracebacks=0).
- **Cond 2 — metric ≥ 96.10**: **FAIL**. 95.73 < 96.10 (also < 96.00 baseline, −0.27pp). → no-improvement.
- **Cond 3 — no constraint violations**: skipped — aborted after Cond 2 per protocol. (Informational: clean — diff
  = only train.py, num_params unchanged 4,299,866, seed 42.)

### Informational Metrics

- num_epochs 85 / num_steps 32,778 (fair converged test — NOT epoch-starved). final_test_loss 0.2136 (no
  improvement over ReLU). peak_vram_mb 548.7. img/s ~14,785 (dt 9ms). num_params 4,299,866 (parameter-free).

## Errors & Dead Ends

## Human Notes

> (none — autopilot)
