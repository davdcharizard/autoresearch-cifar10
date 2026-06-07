# EXP-023: Weight decay 1e-3

## Execution

Overall Status & Info:
- **Created**: 2026-05-29
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-023.md
- **Plan**: plans/plan-023.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-023
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: failed

## Implementation Notes

### Summary
Single change: WEIGHT_DECAY 5e-4 → 1e-3.

### Surprises & Discoveries
None.

### Decisions
No deviations.

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
- Running with weight decay doubled from 5e-4 to 1e-3. Testing whether stronger L2 regularization improves generalization.

Observations:
- 96.01% < 96.39% — WD 1e-3 over-regularizes
- WD 5e-4 is the sweet spot; further increase hurts
- 58 epochs, slight best/final gap (96.01% vs 95.89%)

Key Metrics:
- best_test_acc: 96.01% (source: run.log)
- final_test_acc: 95.89% (source: run.log)
- num_epochs: 58 (source: run.log)

## Verification Results

### Conditions Checked

1. **best_test_acc >= 96.49%**: FAILED — actual 96.01%. (source: run.log)
2. Remaining skipped.

### Informational Metrics
<!-- Not collected. -->

## Errors & Dead Ends

## Human Notes

> {Researcher can add comments, corrections, or context here}
