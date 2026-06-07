# EXP-020: Extended TTA (spatial shifts) — eval-only

## Execution

Overall Status & Info:
- **Created**: 2026-05-29
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-020.md
- **Plan**: plans/plan-020.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-020
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: failed

## Implementation Notes

### Summary
Single change: replaced 2-view TTA (original + hflip) with 6-view TTA (original + hflip + ±1px spatial shifts in 4 directions using reflect padding) in ResNet.forward() eval branch. No training changes whatsoever.

### Surprises & Discoveries
None — minimal change.

### Decisions
No deviations from plan.

## Run Log

### Run 1

Metadata:
- **Job ID**: N/A (local)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-cifar10/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-05-29
- **Ended**: 2026-05-29

Description:
- Running baseline ResNet-20 k=4 with 6-view TTA instead of 2-view. Training is identical to baseline. Only eval-mode forward() changed. Target: best_test_acc >= 96.49%.

Observations:
- 57 epochs in 300s — normal variance, confirms no training changes
- 96.13% < 96.39% baseline — 6-view TTA is WORSE than 2-view hflip TTA
- Spatial-shift views add noise rather than useful diversity — the model's conv layers already handle small translations
- Diluting the strong hflip signal (0.5 → 0.167 weight) with lower-quality shift views hurts the average

Key Metrics:
- best_test_acc: 96.13% (source: run.log)
- final_test_acc: 96.08% (source: run.log)
- num_epochs: 57 (source: run.log)

## Verification Results

### Conditions Checked

1. **best_test_acc >= 96.49%**: FAILED — actual 96.13%, 0.26% below baseline. (source: run.log)
2. Remaining conditions skipped.

### Informational Metrics

<!-- Not collected — necessary condition failed. -->

## Errors & Dead Ends

## Human Notes

> {Researcher can add comments, corrections, or context here}
