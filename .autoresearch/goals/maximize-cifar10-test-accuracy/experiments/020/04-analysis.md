# Report EXP-020: Extend Mixup to 75 Percent
- **Created**: 2026-07-26

## Goal

Raise fixed-seed CIFAR-10 `best_test_acc` above the accepted 94.07% WRN-16-2 baseline within 300 counted training seconds. Success required at least 94.17% with all local H20, scope, timing, seed, and evaluation constraints intact.

## Idea & Hypothesis

The accepted recipe uses batch-shared alpha-0.2 mixup through 65% counted time. EXP-004 showed that ending at 50% lost 0.16 points, providing a directional signal that the 50-65% interval was useful. EXP-020 tested the unexplored longer side by extending mixup to 75% while retaining a 75-second hard-label tail.

## Approach

Only `MIXUP_END_FRACTION` changed from 0.65 to 0.75. Model, initialization, fixed seed, batch-shared coefficient sampling, alpha, optimizer, time-based LR, data transforms, workers, and evaluation remained bit-for-bit accepted. Semantic preflight verified the exact one-line diff, strict boundary behavior, one simulated transition, and identical remaining hyperparameters/LR values.

## Execution

The sole local H20 run completed cleanly with no retries. Mixup disabled exactly once at epoch 106, step 20,569 and 225.0 counted seconds. The run evaluated 29 unique epochs, retained the accepted 691,674 parameters, and emitted no error signature.

## Results

- **Primary metric**: 93.82% (baseline: 94.07%, delta: -0.25 points, -0.27%)
- **Observations**: The candidate completed 27,655 steps, or 141.5936 dataset-equivalent passes, in 300.0 training seconds and 340.5 wall seconds. Best and final accuracy were 93.82%; final loss was 0.2612 versus 0.2432 accepted.
- **Analysis**: The intervention achieved its intended local effect with no exposure or implementation confound: mixup covered ten additional percentage points and the run retained essentially accepted passes. Accuracy and loss both worsened, showing that the 50-to-65 improvement does not continue monotonically. The duration response is an inverted-U around the accepted region: 50% under-regularizes, while 75% leaves too little clean-label refinement or over-regularizes the late schedule. The accepted 65% cutoff is therefore locally bracketed and should remain fixed.
- **Key Learning**: Mixup duration is locally bracketed around 65%; both 50% and 75% preserve exposure but reduce test accuracy.

## Verification

- **Conditions**: Completion and integrity passed; the primary metric condition failed because 93.82% is below 94.17%.
- **Review Notes**: Results are trustworthy. The exact one-line treatment, one H20, fixed seed, single scored execution, 225.0-second transition, unique evaluations, and complete finite summary were confirmed.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid complete run with no hard-constraint violation, but accuracy fell 0.25 points below baseline and 0.35 below threshold.

## Unexplored Avenues

- **Fine cutoff values near 65%**: 60% or 70% could map the local peak more finely, but expected effect is below the acceptance margin and would be low-value without new evidence.
- **Weaker alpha at fixed duration**: alpha 0.1 remains untested, but local under-regularization evidence makes it a weak bet.
- **Replace rather than compound regularization**: a staged alternative augmentation could avoid stacking on mixup, though worker-safe execution is substantially more complex.

## Next Steps

- **Medium confidence - new optimization geometry**: revisit early-window SAM only after a local sharpness diagnostic supports it and timing fits the budget.
- **Medium confidence - non-compounding data invariance**: replace mixup for a bounded early window rather than stacking RandAugment on it.
- **Low confidence - alpha 0.1**: use only as a final controlled closure of mixup strength, since the accepted duration is now bracketed.

## Exit Action Results

