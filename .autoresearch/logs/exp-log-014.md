# EXP-014: RandAugment(2, 9) replacing TrivialAugmentWide (keep Cutout(16) + compile)

## Execution

Overall Status & Info:
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-014.md
- **Plan**: plans/plan-014.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-014
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed (clean run; verification cond 2 failed → no-improvement verdict in analyze)

## Implementation Notes

### Summary
Single-line `train.py` edit (Milestone 1): replaced `transforms.TrivialAugmentWide()` → `transforms.RandAugment()`
(torchvision defaults num_ops=2, magnitude=9 — the standard CIFAR setting), on top of the EXP-012 baseline (Cutout(16)
+ compile, commit 6c417a4). Updated the inline comment. Ruff clean; `git diff` = the augmentation line + comment only.
Tests whether two ops/image (vs TA's single op) — more augmentation strength — improves over the TA recipe.

### Surprises & Discoveries
- (none at implementation time — `RandAugment` confirmed in torchvision 0.24.1; defaults are the CIFAR (2,9) setting.)

### Decisions
- **Used RandAugment() defaults (num_ops=2, magnitude=9)**: the canonical CIFAR-WRN RA setting, so the first probe of
  the "more aug" axis uses the standard point rather than an arbitrary (N,m). If it helps, magnitude/num_ops become
  explicit tunable follow-up knobs; if it over-augments, that bounds the sweet spot at TA's lighter single-op regime.

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
- Running the EXP-012 recipe (k=4 WRN + Cutout(16) + compile) with RandAugment(2,9) in place of TrivialAugmentWide.
  Probes the demonstrated-live augmentation-strength axis (EXP-012 gained adding TA; EXP-013 lost reducing aug) by
  applying two ops/image. Expect params 4,299,866 (unchanged), ~8–9ms/step (RA is CPU PIL ops like TA), ~88–91 epochs.
  KEY corroborator: final_test_loss ≤ 0.195 if RA improved the fit; > 0.195 with acc↓ ⇒ over-augmented. Success = ≥96.32.

Observations:
- Clean startup: **params 4,299,866 (UNCHANGED)**, clean compile, no traceback, no NaN (run.log head).
- **Steady 8ms/step (~15,300 img/s) from step 50 — identical to EXP-012/compiled-k4**: RandAugment(2,9) adds no
  throughput cost (CPU PIL ops, 8 workers keep up). Tracking toward ~90 epochs ⇒ fair converged test. ep1 eval 57.56%
  (source: run.log step 50–900 + ep1 eval line).

Key Metrics:
- **best_test_acc: 96.19%** — BELOW the 96.32 bar and ≈ the 96.22 baseline (−0.03pp, within noise) (run.log summary).
- **num_epochs: 91** / num_steps 35,256 — fair, fully-converged test (eval count 91 == num_epochs ⇒ eval once/epoch).
- **final_test_loss: 0.1972** — ≈ EXP-012's 0.195 (within noise). RA(2,9) and TA produce essentially identical
  fit-quality and accuracy → the auto-aug POLICY choice doesn't matter here. Late evals 96.11–96.19 (ep 87–91), a
  stable cluster ≈ TA's 96.12–96.22 (overlapping) → equivalent, not a regression.
- num_params 4,299,866 (UNCHANGED). peak_vram 453.8 MB. dt 8ms (~15,300 img/s ≈ EXP-012; RA no throughput cost).

## Verification Results

### Conditions Checked

- **Cond 1 — clean completion within budget**: PASS. best_test_acc 96.19% present, total_seconds 411.4 < 600, no
  traceback (run.log; tracebacks=0).
- **Cond 2 — metric ≥ 96.32**: **FAIL**. 96.19 < 96.32 (≈ 96.22 baseline, −0.03pp within noise). → no-improvement.
- **Cond 3 — no constraint violations**: skipped — aborted after Cond 2. (Informational: clean — diff = train.py one
  line, num_params unchanged 4,299,866, seed 42, eval count 91 == num_epochs.)

### Informational Metrics

- num_epochs 91 / num_steps 35,256 (fair converged test). final_test_loss 0.1972 ≈ EXP-012's 0.195 (RA ≈ TA).
  peak_vram_mb 453.8. img/s ~15,300 (dt 8ms ≈ EXP-012 — no throughput confound).

## Errors & Dead Ends

## Human Notes

> (none — autopilot)
