# EXP-021: CutMix probability reduction (0.5 → 0.3)

## Execution

Overall Status & Info:
- **Created**: 2026-05-29
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-021.md
- **Plan**: plans/plan-021.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-021
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: failed

## Implementation Notes

### Summary
Single change: CUTMIX_PROB 0.5 → 0.3. This reduces CutMix-augmented batches from 50% to 30%, giving the model more clean training samples for faster convergence.

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
- Running baseline with CutMix probability reduced from 0.5 to 0.3. Testing over-regularization hypothesis. Target: best_test_acc >= 96.49%.

Observations:
- 96.06% < 96.39% baseline — CutMix prob 0.3 is WORSE than 0.5
- 58 epochs, good T_max alignment (best≈final)
- The model benefits from 50% CutMix — it provides valuable regularization, not over-regularization
- The over-regularization hypothesis from EXP-011 was about ADDING more regularization, not the existing level being too much

Key Metrics:
- best_test_acc: 96.06% (source: run.log)
- final_test_acc: 96.01% (source: run.log)
- num_epochs: 58 (source: run.log)

## Verification Results

### Conditions Checked

1. **best_test_acc >= 96.49%**: FAILED — actual 96.06%, 0.33% below baseline. (source: run.log)
2. Remaining conditions skipped.

### Informational Metrics

<!-- Not collected — necessary condition failed. -->

## Errors & Dead Ends

## Human Notes

> {Researcher can add comments, corrections, or context here}
