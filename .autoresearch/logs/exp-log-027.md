# EXP-027: CutMix alpha 0.5

## Execution

Overall Status & Info:
- **Created**: 2026-05-29
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-027.md
- **Plan**: plans/plan-027.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-027
- **Commit**: (pending)
- **PR**: (pending)
- **Outcome**: failed

## Implementation Notes

### Summary
Single change: CUTMIX_ALPHA 1.0 → 0.5. Beta(0.5, 0.5) U-shaped distribution for lighter average mixing.

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
- CutMix alpha 0.5 (U-shaped mixing). Same frequency (p=0.5), lighter intensity.

Observations:
- 96.13% < 96.39% baseline. Alpha 0.5 slightly worse than alpha 1.0.
- All CutMix parameters now confirmed: alpha=1.0, p=0.5 is optimal.

Key Metrics:
- best_test_acc: 96.13% (source: run.log)
- num_epochs: 58 (source: run.log)

## Verification Results

### Conditions Checked
1. **best_test_acc >= 96.49%**: FAILED — actual 96.13%. (source: run.log)
2. Remaining skipped.

### Informational Metrics
<!-- Not collected. -->

## Errors & Dead Ends

## Human Notes

> {Researcher can add comments, corrections, or context here}
