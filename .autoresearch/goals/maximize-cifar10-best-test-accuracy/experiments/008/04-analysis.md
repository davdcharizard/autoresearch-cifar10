# Report EXP-008: Width-2 Weight Decay 5e-4
- **Created**: 2026-08-05

## Goal

Increase CIFAR-10 `best_test_acc (%)`, higher is better, from the width-2 moving baseline of 93.55% at `8faf0f3`. A valid improvement required at least 93.65% under the fixed one-H20, 300-second, `train.py`-only protocol.

## Idea & Hypothesis

Increase only coupled SGD weight decay from `1e-4` to the canonical CIFAR Wide ResNet value `5e-4`. Claude selected this compute-neutral point as the cleanest test of whether plateau norm control could carry better width-2 generalization into the weak tail. The hypothesis predicted at least 93.65%, while pre-registering a lower strong checkpoint and elevated tail train loss as evidence that the point was too aggressive.

## Approach

Changed exactly one literal in `train.py`. Width 2, 1,073,962 parameters, N1/M7 through 80%, the deterministic weak-loader transition, hard-label tail, batch 128, LR schedule, momentum, seed, evaluator, and all other code remained unchanged. The existing single optimizer group applied `5e-4` coupled decay to its existing full parameter set, including BN affine values and classifier bias.

## Execution

Mandatory external Claude idea and plan reviews completed with exit code 0; no fallback reviewer was used. Compilation, Ruff, pre-commit, exact diff, model shape/parameter, and optimizer-value checks passed. One fixed-seed H20 run exited 0 without retry in 332.2 seconds total. It completed 26,729 steps, 98.48% of EXP-007 exposure; this small difference is node timing rather than added scalar work.

## Results

- **Primary metric**: `93.38%` (baseline: `93.55%`, delta: `-0.17` percentage points, `-0.18%` relative)
- **Observations**: The strong checkpoint collapsed from EXP-007's 90.08%/0.2283 loss EMA to 81.29%/0.4148. The first weak checkpoint was 91.63% versus 92.96%. The tail then rose steadily to 93.38% at the final epoch; its last-three slope was approximately +0.02 points/epoch. Final test loss was 0.1988, 0.0208 lower than EXP-007's 0.2196, and the minimum was 0.1975 at epoch 67. Final train-loss EMA was 0.0770, still substantially above EXP-007's roughly 0.04 late-tail level.
- **Analysis**: The exact `5e-4` point is too strong for the short fixed-time width-2 regime. It achieved norm/confidence regularization strongly enough to improve NLL, but suppressed fitting throughout the N1/M7 plateau by a large margin and entered the weak tail 1.33 accuracy points behind. The 14-epoch tail recovered most of that gap and was still rising at termination, yet not enough to beat top-1. This is an operating-point failure, not evidence that norm control is irrelevant: the clean one-literal intervention, lower loss, unambiguous underfit signature, and rising endpoint support testing a lower decay that retains more strong-view capacity.
- **Key Learning**: Coupled decay `5e-4` lowered NLL but over-regularized width 2; a lower decay is the justified next point.

## Verification

- **Conditions**: Completion, summary, timing, scope, parameter count, lifecycle, and evaluation uniqueness passed. Primary accuracy failed: 93.38% <93.65%.
- **Review Notes**: Results are trustworthy. The data/RNG stream, operation graph, evaluator, and training protocol were unchanged except for the decay scalar; one idle H20 was used; no reroll occurred. Step variation was informational and did not cause the statistical failure.
- **Verdict**: no-improvement
- **Verdict Basis**: The valid run finished 0.17 points below the moving baseline and 0.27 below the required threshold.

## Unexplored Avenues

- Test `2e-4` coupled decay. It is the most conservative interpolation between accepted `1e-4` and the clearly over-strong `5e-4`, with identical compute and attribution.
- Test `3e-4` only if external review favors stronger norm pressure despite the large plateau-fit collapse; it has more overshoot risk than `2e-4`.
- Exclude BN affine and bias from stronger decay while retaining stronger convolution/linear-weight regularization. This changes parameter-group semantics and requires a separate reviewed experiment.
- Apply extra decay only late. This targets the observed tail directly but low LR makes coupled decay weak there and phase-dependent optimizer mutation is less clean than a scalar point.

## Next Steps

- **High confidence**: restore accepted `1e-4` and adversarially compare a one-literal `2e-4` point against selective or late-only decay.
- **Medium confidence**: prefer `2e-4` over `3e-4` because `5e-4` depressed the strong checkpoint by 8.79 points.
- **Low-medium confidence**: revisit architecture regularization only after the scalar decay range is narrowed; width 3 and block dropout remain poorer fixed-time bets.

## Exit Action Results

- None defined.
