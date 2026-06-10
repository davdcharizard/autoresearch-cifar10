# EXP-016: LR-schedule micro-tuning — raise peak LR 0.2 → 0.3 on the TA+Cutout recipe

## Execution

Overall Status & Info:
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-016.md
- **Plan**: plans/plan-016.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-016
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary
Single-constant change per plan-016 Milestone 1: `PEAK_LR` 0.2 → 0.3 in train.py line 23 (with an
inline comment noting the EXP-016 retune). `PEAK_LR` feeds both `lr_at_fraction` (the 5% linear
warmup → cosine-to-0 schedule, L35-41) and the optimizer's initial `lr` (L194), so the schedule's
amplitude scales while its shape is unchanged. No other edits. Ruff clean; `git diff` = the single
`PEAK_LR` line.

### Surprises & Discoveries
None — trivial hyperparameter edit on the restored EXP-012 baseline.

### Decisions
- Tested the HIGHER direction (0.3) rather than lower (0.1): EXP-000's 0.2 already beat the textbook
  batch-128 peak of 0.1, evidence the model prefers a higher-than-default LR, and heavy TA+Cutout
  regularization should tolerate/benefit from a more aggressive peak. If 0.3 nulls/regresses, the next
  loop probes the opposite direction (0.1–0.15).

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
- Running the EXP-012 baseline recipe with the single change `PEAK_LR` 0.2→0.3 on a single H20. This is
  a pure hyperparameter change (compute-neutral), so we expect ~8ms/step and ~88–91 epochs, matching
  EXP-012 — a fair same-budget test with no throughput confound (the issue that muddied EXP-015).
  Hypothesis: the higher peak lets the heavily-augmented model explore more before the cosine anneals to
  0, reaching a flatter minimum and lifting best_test_acc above the 96.32 bar (corroborated by
  final_test_loss ≤ 0.195). A null (≈96.22, loss ≈0.195) means the peak is already near-optimal.

Observations:
- Clean startup: `params: 4,299,866` (UNCHANGED vs baseline — correct for a hyperparameter-only change),
  clean compile, no traceback (source: run.log L1-4).
- LR schedule scaled correctly: warmup ramped 0.109→0.164 by 2.7% done and **peaked at exactly 0.3000**
  before the cosine descent (source: run.log warmup steps + grep `lr: 0.3000`).
- Higher-LR early noise as predicted: ep 1 test_acc 39.80% (vs EXP-015's 44.42% at the old peak) — recovered
  normally, no NaN/divergence (source: run.log eval ep 1).
- Ran 84 epochs / 32,620 steps, dt ~8–13ms (within run-to-run throughput jitter; compute-neutral change).
- Run exited 0, total_seconds 400.8 < 600 (source: run.log final summary, background task exit 0).

Key Metrics:
- best_test_acc: 95.77% @ ep 80 (source: run.log)
- final_test_acc: 95.70% @ ep 84 (source: run.log)
- final_test_loss: 0.2018 @ ep 84 (source: run.log) — vs EXP-012's 0.195 (loss ROSE)
- num_epochs: 84 | num_steps: 32,620 | num_params: 4,299,866 | peak_vram_mb: 453.8 | total_seconds: 400.8

## Verification Results

### Conditions Checked
- **Cond 1 — clean completion within budget**: PASS. best_test_acc and total_seconds present; total_seconds
  400.8 < 600; Traceback count 0 (source: run.log final summary).
- **Cond 2 — primary metric clears bar**: **FAIL**. best_test_acc = 95.77% < 96.32 bar (baseline 96.22 + 0.1).
  Δ = −0.45pp vs baseline. → verdict no-improvement. (Decisive condition.)
- **Cond 3 — no constraint violations**: skipped — not reached after Cond 2 failed. (For the record: scope clean —
  git diff = train.py only, the single PEAK_LR line; eval-count 84 == num_epochs 84; num_params 4,299,866 unchanged;
  seed 42 intact; no new deps.)

### Informational Metrics
- Not collected (only when all necessary conditions pass). For reference: num_epochs 84, final_test_loss 0.2018
  (> EXP-012's 0.195), peak_vram_mb 453.8 (unchanged).

## Errors & Dead Ends

## Human Notes

> (none)
