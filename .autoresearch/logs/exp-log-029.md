# EXP-029: SGDR — cosine annealing with warm restarts (2 cycles)

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-029.md
- **Plan**: plans/plan-029.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-029
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary
Milestone 1 per plan-029: added `N_CYCLES = 2` after WARMUP_FRAC and rewrote `lr_at_fraction(frac)` into a 2-cycle SGDR schedule — split [0,1] into N_CYCLES equal cosine cycles, each annealing PEAK_LR→~0, with linear warmup over WARMUP_FRAC of the FIRST cycle only and restarts jumping straight to PEAK_LR. Smoke test passed all shape assertions: lr(0)=0, lr(0.025)=0.2 (warmup end), lr(0.499)≈0 (cycle-1 anneal complete), lr(0.5)=0.2 (restart), lr(0.75)=0.1 (cycle-2 cosine mid), lr(1.0)=0; all values in [0, PEAK]. params 4,299,866 unchanged (schedule-only); git diff = train.py only.

### Surprises & Discoveries
None. The `int(frac/cycle_len)` edge at frac==1.0 is correctly clamped into the last cycle by `min(..., N_CYCLES-1)` (verified: lr(1.0)=0, not a spurious restart).

### Decisions
- Warmup is applied over WARMUP_FRAC of the FIRST CYCLE (~2.3 ep) rather than of the full budget (~4.5 ep before). Kept simple/clean for standard SGDR semantics; warmup is second-order here since the 50% restart jumps to PEAK with no warmup anyway. Documented in the plan.
- N_CYCLES=1 exactly reproduces the prior single cosine-to-0 (clean, auditable fallback) — confirms the change is a pure generalization of the baseline schedule.

## Experimental Adjustments

<!-- none yet -->

## Run Log

### Run 1

Metadata:
- **Job ID**: (PID — background task)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-opus-4-8/run.log
- **WandB**: N/A
- **Status**: completed (exit 0)
- **Started**: 2026-06-09
- **Ended**: 2026-06-09

Description:
- Full 300s-compute-budget training of the k=4 WideResNet with a 2-cycle SGDR LR schedule (warm restart at the 50%-budget mark), on a single H20. Hypothesis: the warm restart re-explores past the single-cosine basin into a flatter/better-generalizing minimum, lifting best_test_acc above the 96.32 bar. Schedule-only change → compute/param-neutral, so epochs should hold ~91 / dt ~8ms (throughput-neutral check).

Observations:
- Run exited 0, clean compile, no NaN/Traceback (Traceback count 0). (source: run.log)
- **SGDR confirmed active**: lr annealed to ~0.0000 by 49.8% budget (cycle-1 cosine complete), then JUMPED to 0.2000 at 50.1% (restart), with the expected transient loss bump 0.87→1.16. Cycle 2 then re-annealed. (source: run.log lr/loss trace ~48-51%)
- **REGRESSION (fair test)**: best_test_acc 95.55% vs baseline 96.22 (−0.67pp), well below the 96.32 bar. (source: run.log summary)
- **Throughput-NEUTRAL / fully fair**: num_epochs 91, num_steps 35292 (= baseline ~91/~35500), dt 8ms (664/705 sampled lines). Schedule-only change added zero compute → the −0.67pp is entirely attributable to the schedule, NOT a confound.
- **Loss also worse**: final_test_loss 0.2076 vs baseline 0.195 — the restart destroyed cycle-1's converged minimum (loss 0.84) and cycle 2 (~45 ep) re-converged to a worse point than the single full-budget cosine.

Key Metrics:
- best_test_acc: 95.55% (source: run.log summary)
- final_test_loss: 0.2076 (vs baseline 0.195)
- num_epochs: 91 | num_steps: 35292 | num_params: 4,299,866 | peak_vram_mb: 453.8 | total_seconds: 403.4 (source: run.log)
- mean dt ≈ 8ms (664/705 sampled lines; throughput-neutral with baseline)

## Verification Results

### Conditions Checked

- **Cond 1 — primary metric clears bar**: FAIL. best_test_acc = 95.55% < 96.32 (baseline 96.22 + 0.1). Per plan, stop at first failure. (source: run.log summary)
- **Cond 2 — clean completion within budget**: PASS (informational). Summary printed, Traceback 0, total_seconds 403.4 < 600. (source: run.log)
- **Cond 3 — no constraint violations**: PASS (informational). git diff = train.py only; num_params 4,299,866 unchanged; 91 evals for 91 epochs (≤1/epoch); no new deps (pure arithmetic); seed 42 unchanged. (source: git diff, run.log)

**MANDATORY attribution note (epoch-wall + FLOPs-neutral-≠-wall-clock-neutral, EXP-015/024):** num_epochs 91, dt 8ms — IDENTICAL to baseline. Schedule-only change is perfectly throughput-neutral, so this is a FULLY FAIR test (the cleanest of the recent runs — no epoch confound at all). The restart (verified firing at 50.1%) genuinely HURT: −0.67pp top-1 AND loss 0.195→0.208. Verdict: **no-improvement**.

### Informational Metrics

- peak_vram_mb: 453.8 (≈ baseline, schedule-only)
- num_epochs / num_steps: 91 / 35292 (= baseline — perfectly throughput-neutral)
- final_test_loss: 0.2076 (WORSE than baseline 0.195)

## Errors & Dead Ends

<!-- none yet -->

## Human Notes

> (none — autopilot)
