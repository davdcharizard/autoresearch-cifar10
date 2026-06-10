# EXP-013: Reduce Cutout hole size 16→8px under the TA+compile recipe

## Execution

Overall Status & Info:
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-013.md
- **Plan**: plans/plan-013.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-013
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed (clean run; verification cond 2 failed → no-improvement verdict in analyze)

## Implementation Notes

### Summary
Single-constant `train.py` edit (Milestone 1): `CUTOUT_SIZE = 16` → `8`, on top of the EXP-012 baseline (TA +
Cutout + compile, commit 6c417a4). Ruff clean; `git diff` = the one CUTOUT_SIZE line only (TA and compile untouched).
Tests whether the occlusion sweet spot shifted down once TrivialAugment raised total augmentation strength.

### Surprises & Discoveries
- (none at implementation time — trivial constant change.)

### Decisions
- **8px (half of 16), not an intermediate like 12**: a clear single-step probe of "less occlusion under TA"; 8px is
  a common smaller-Cutout value. If 8px wins, a follow-up could bracket 8–12; if it loses, the 16px hole was already
  near-optimal under TA.

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
- Running the EXP-012 recipe (k=4 WRN + TrivialAugmentWide + compile) with Cutout reduced 16→8px. Tests whether
  reducing total augmentation strength (now that TA supplies diverse augmentation) improves convergence within the
  300s budget and lifts best_test_acc above the 96.32 bar. Expect params 4,299,866 (unchanged), ~8ms/step (≈ EXP-012;
  Cutout size doesn't change cost), ~89–91 epochs. KEY corroborator: final_test_loss should drop below 0.195 if the
  smaller hole helped. Success = best_test_acc ≥ 96.32.

Observations:
- Clean startup: **params 4,299,866 (UNCHANGED)**, clean compile, no traceback, no NaN (run.log head).
- Throughput settling to ~9ms/step (~15,000 img/s) by step 400–500 after epoch-1 warmup jitter (compile + epoch-1
  gc.collect) — consistent with EXP-012's 8ms steady state; Cutout size doesn't affect cost as expected. Tracking
  toward ~89–91 epochs ⇒ fair converged test. ep1 eval 46.55% (source: run.log step 50–500 + ep1 eval line).

Key Metrics:
- **best_test_acc: 95.92%** — BELOW the 96.32 bar AND below the 96.22 baseline (−0.30pp) (run.log summary).
- **num_epochs: 92** / num_steps 35,662 — fair, fully-converged test (eval count 92 == num_epochs ⇒ eval once/epoch);
  NOT epoch-starved, so the smaller Cutout got a fair shot.
- **final_test_loss: 0.2023** — HIGHER than EXP-012's 0.195 (the 16px+TA baseline). Loss↑ AND acc↓ together ⇒ the
  smaller 8px hole UNDER-regularized: reducing occlusion hurt both fit-quality and accuracy. Late evals cluster
  95.82–95.92 (ep 88–92) — stable, clearly below the 96.22 baseline's 96.12–96.22 cluster.
- num_params 4,299,866 (UNCHANGED). peak_vram 453.8 MB. dt ~9ms (~15,000 img/s ≈ EXP-012; Cutout size no throughput effect).

## Verification Results

### Conditions Checked

- **Cond 1 — clean completion within budget**: PASS. best_test_acc 95.92% present, total_seconds 410.3 < 600, no
  traceback (run.log; tracebacks=0).
- **Cond 2 — metric ≥ 96.32**: **FAIL**. 95.92 < 96.32 (also < 96.22 baseline, −0.30pp). → no-improvement.
- **Cond 3 — no constraint violations**: skipped — aborted after Cond 2. (Informational: clean — diff = train.py
  one line, num_params unchanged 4,299,866, seed 42, eval count 92 == num_epochs.)

### Informational Metrics

- num_epochs 92 / num_steps 35,662 (fair converged test, not epoch-starved). final_test_loss 0.2023 (HIGHER than
  EXP-012's 0.195 — the smaller hole under-regularized). peak_vram_mb 453.8. img/s ~15,000.

## Errors & Dead Ends

## Human Notes

> (none — autopilot)
