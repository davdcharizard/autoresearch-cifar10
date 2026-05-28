# Experiment Log: EXP-033

## Execution
- **Created**: 2026-05-28
- **Brainstorm**: brainstorm/brainstorm-033.md
- **Plan**: plans/plan-033.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-033
- **Commit**: (pending)
- **PR**: (pending)
- **Outcome**: completed

## Implementation Notes
### Summary
Three changes on current Nesterov+reflect baseline: removed RandomHorizontalFlip, added alternating flip in training loop, reduced WEIGHT_DECAY from 5e-4 to 4e-4.
### Surprises & Discoveries
None.
### Decisions
None.

## Run Log
### Run 1
- **Description**: Alternating flip + WD 4e-4 on Nesterov+reflect baseline. Four-axis stack targeting optimizer, data quality, augmentation pattern, and regularization strength.
- **Job ID**: local
- **Log file**: run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-05-28
- **Ended**: 2026-05-28
- **Observations**: WD reduction from 5e-4 to 4e-4 combined with alternating flip gave 96.52% — worse than alternating flip alone (96.64% in EXP-032). WD reduction caused under-regularization that negated the alternating flip benefit.
- **Key Metrics**: best_test_acc=96.52%, final_test_acc=96.46%, num_epochs=98

## Verification Results
### Conditions Checked
1. **best_test_acc > 96.66%**: FAILED. Actual: 96.52% (-0.04pp below baseline).
### Informational Metrics

## Errors & Dead Ends
(none)

## Human Notes
(autopilot)
