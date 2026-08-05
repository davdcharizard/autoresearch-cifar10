# Report EXP-004: Earlier 50% Mixup Cutoff
- **Created**: 2026-07-24

## Goal

Increase CIFAR-10 `best_test_acc` above the 94.07% moving baseline by at least 0.10 percentage points, testing whether more hard-label cosine refinement improves the accepted alpha-0.2 mixup schedule.

## Idea & Hypothesis

Change only the mixup cutoff from 65% to 50% counted time, giving mixed-target representation learning and hard-label refinement 150 seconds each. EXP-002 ended at its best after a rising hard-label tail, so the hypothesis predicted that 45 additional clean-label seconds would reach at least 94.17% without materially reducing exposure.

## Approach

The experiment changed one line in `train.py`: `MIXUP_END_FRACTION = 0.65` became `0.50`. Alpha, architecture, optimizer, time-based LR schedule, seed and RNG trajectory, loader, transforms, evaluation cadence, finite-loss guard, and output schema were otherwise identical to EXP-002.

## Execution

One full run completed without retries or adjustments. Lint, compile, diff, constant, parameter-count, evaluator-call, scope, and H20 checks passed. Mixup disabled once at 150.0 counted seconds, epoch 71, step 13,674, and LR 0.1092. The run completed 27,954 steps in 340.7 seconds total.

## Results

- **Primary metric**: 93.91% (baseline: 94.07%, delta: -0.16 percentage points, -0.17%)
- **Observations**: The cheaper hard-label branch increased exposure from EXP-002's 141.9 to 143.1 passes, a 0.9% gain, but accuracy still fell. Best accuracy occurred at epoch 135 and final was only 0.01 points lower. Final test loss was 0.2708 versus EXP-002's 0.2432, despite near-zero late training loss.
- **Analysis**: The hypothesis is not supported. The intervention delivered both the intended longer hard-label phase and slightly more total exposure, yet produced a stable lower endpoint and worse test loss. This indicates that mixup between 50% and 65% contributes useful generalization that extra hard-label fitting cannot replace. The 50% operating point is discredited; the broader cutoff question remains open on the later side.
- **Key Learning**: Ending mixup at 50% gains 0.9% exposure but loses 0.16 accuracy points, indicating the 50-65% regularization window remains valuable.

## Verification

- **Conditions**: metric improvement condition failed
- **Review Notes**: Results confirmed trustworthy. The process exited 0 on one H20, used 300.0 counted seconds, stayed below 600 seconds total, evaluated at most once per epoch, and had an exact one-line in-scope diff.
- **Verdict**: no-improvement
- **Verdict Basis**: The valid 93.91% result was 0.16 points below the 94.07% baseline and 0.26 points below the 94.17% threshold.

## Unexplored Avenues

- Test a later 75% cutoff, the opposite arm of the timing question, to determine whether additional mixed-target regularization helps while retaining a 75-second hard-label tail.
- Keep the 65% timing and tune alpha upward moderately, separating regularization strength from duration.

## Next Steps

- **High confidence**: test the pre-registered opposite 75% cutoff while preserving alpha 0.2 and every other accepted setting.
- **Medium confidence**: if 75% also fails, restore 65% and test alpha 0.3-0.4 rather than the weakly supported alpha 0.1 direction.
- **Medium confidence**: retain regularized WRN-16-3 as a higher-upside later bet with a strict exposure gate.

## Exit Action Results

No exit actions were defined for this local-only run.
