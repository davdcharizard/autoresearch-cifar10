# EXP-024: Label smoothing 0.05 + fixed numpy seed

## Execution

Overall Status & Info:
- **Created**: 2026-05-29
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-024.md
- **Plan**: plans/plan-024.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-024
- **Commit**: (pending)
- **PR**: (pending)
- **Outcome**: failed

## Implementation Notes

### Summary
Two changes: LABEL_SMOOTHING 0.1→0.05, added np.random.seed(42) for deterministic CutMix.

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
- Label smoothing halved (0.1→0.05) with fixed numpy seed.

Observations:
- 96.22% < 96.39% baseline. Label smoothing 0.05 didn't help.
- Note: numpy seed changed the CutMix sequence vs baseline, so not a perfectly fair comparison. But clearly not an improvement.

Key Metrics:
- best_test_acc: 96.22% (source: run.log)
- num_epochs: 58 (source: run.log)

## Verification Results

### Conditions Checked

1. **best_test_acc >= 96.49%**: FAILED — actual 96.22%. (source: run.log)
2. Remaining skipped.

### Informational Metrics
<!-- Not collected. -->

## Errors & Dead Ends

## Human Notes

> {Researcher can add comments, corrections, or context here}
