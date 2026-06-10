# EXP-017: LR-schedule micro-tuning — lower peak LR 0.2 → 0.15 (sign-corrected probe)

## Execution

Overall Status & Info:
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-017.md
- **Plan**: plans/plan-017.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-017
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary
Single-constant change per plan-017 Milestone 1: `PEAK_LR` 0.2 → 0.15 in train.py line 23 (inline comment
notes the EXP-017 sign-corrected retune). `PEAK_LR` feeds both `lr_at_fraction` (5% warmup → cosine-to-0,
L35-41) and the optimizer's initial `lr` (L194); schedule shape unchanged, amplitude scales. No other edits.
Ruff clean; `git diff` = the single `PEAK_LR` line.

### Surprises & Discoveries
None — trivial hyperparameter edit on the restored EXP-012 baseline.

### Decisions
- Tested 0.15 (not 0.1): EXP-016 showed 0.3 regressed (optimum ≤ 0.2), so 0.15 is the maximum-likelihood
  optimum location if it lies below 0.2 — a modest step toward the textbook batch-128 peak (0.1) that hedges
  against under-progress within the fixed budget. If 0.15 ≈ 0.2 or worse, the LR-peak axis is settled.

## Experimental Adjustments

<!-- none yet -->

## Run Log

### Run 1

Metadata:
- **Job ID**: (PID — local background run)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-opus-4-8/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-08
- **Ended**: 2026-06-08

Description:
- Running the EXP-012 baseline recipe with the single change `PEAK_LR` 0.2→0.15 on a single H20. Pure
  hyperparameter change (compute-neutral) → expect ~8ms/step and ~84–91 epochs, a fair same-budget test.
  Hypothesis: the gentler peak settles into a slightly better-generalizing minimum within the budget, lifting
  best_test_acc above the 96.32 bar (corroborated by final_test_loss ≤ 0.195). A null (≈96.22) means peak 0.2
  is already optimal and the LR-peak axis is settled (0.3 worse above, 0.15 not better below).

Observations:
- Clean startup: `params: 4,299,866` (UNCHANGED — correct for a hyperparameter-only change), clean compile,
  no traceback (source: run.log L1-4).
- LR scaled correctly: warmup **peaked at exactly 0.1500** before the cosine descent (source: run.log grep `lr: 0.1500`).
- Ran 77 epochs / 29,744 steps (lower-end of the run-to-run throughput-jitter band ~65–77 per goal-learnings).
  Combined with the gentler LR, plausible mild under-progress — but the regression is clear regardless.
- Run exited 0, total_seconds 398.3 < 600 (source: run.log final summary, background task exit 0).

Key Metrics:
- best_test_acc: 95.58% @ ep 73 region (source: run.log) — vs baseline 96.22 (−0.64pp)
- final_test_acc: 95.53% @ ep 77 (source: run.log)
- final_test_loss: 0.2046 @ ep 77 (source: run.log) — vs EXP-012's 0.195 (loss ROSE most of the LR sweep)
- num_epochs: 77 | num_steps: 29,744 | num_params: 4,299,866 | peak_vram_mb: 453.8 | total_seconds: 398.3
- **LR-peak sweep summary**: 0.15 → 95.58 (EXP-017) | 0.2 → 96.22 (baseline, EXP-012) | 0.3 → 95.77 (EXP-016).
  0.2 is a clear peak; both directions regress → LR-peak axis SETTLED.

## Verification Results

### Conditions Checked
- **Cond 1 — clean completion within budget**: PASS. best_test_acc and total_seconds present; total_seconds
  398.3 < 600; Traceback count 0 (source: run.log final summary).
- **Cond 2 — primary metric clears bar**: **FAIL**. best_test_acc = 95.58% < 96.32 bar. Δ = −0.64pp vs
  baseline 96.22. → verdict no-improvement. (Decisive condition.)
- **Cond 3 — no constraint violations**: skipped — not reached after Cond 2 failed. (For the record: scope clean —
  git diff = train.py only, single PEAK_LR line; eval-count 77 == num_epochs 77; num_params 4,299,866 unchanged;
  seed 42 intact; no new deps.)

### Informational Metrics
- Not collected (only when all necessary conditions pass). For reference: num_epochs 77, final_test_loss 0.2046
  (> EXP-012's 0.195), peak_vram_mb 453.8 (unchanged).

## Errors & Dead Ends

## Human Notes

> (none)
