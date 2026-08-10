# Report EXP-017: Learned Pool-First Transition Shortcuts
- **Created**: 2026-08-06

## Goal

Maximize CIFAR-10 `best_test_acc` above the 94.15% frontier at `7c1e7d8` while changing only `train.py` and preserving the fixed seed-42, one-H20, 300-second training and evaluator protocol. A formal improvement required at least 94.25%.

## Idea & Hypothesis

Replace only the two Option-A transition shortcuts with average pooling followed by a learned bias-free 1x1 projection and BatchNorm. The pool-first path was expected to retain more spatial information than strided slicing, learn a useful channel basis, preserve at least 25,500 optimizer steps, and raise best accuracy to at least 94.25% without disturbing the seven identity shortcuts or accepted residual branches.

## Approach

`train.py` gained a marker convolution class and two exact `AvgPool2d(2,2) -> 1x1 Conv -> BatchNorm` shortcuts. Constructor draws were isolated from the global CPU RNG, and both projections were initialized sequentially by a dedicated generator derived from the active seed; the accepted shared tensors and post-construction RNG state remained bitwise equal to control. External Claude adversarially reviewed the exact production diff and ignored controllers and returned `APPROVED` with no blocking issue. Structural, 200-real-batch numerical, five-pair timing/inference, and 1,000-batch loader gates all passed before launch.

## Execution

One fixed-seed production run was launched under the 600-second supervisor with output only in `run.log`; it exited zero and was not retried. The run completed 26,557 steps in exactly 300.0 counted seconds and 331.5 seconds total. It switched at the fixed 80% point, stopped all eight strong workers, realized 49.77% CutMix, and performed 19 evaluations on 19 unique epochs.

## Results

- **Primary metric**: 94.09% (baseline: 94.15%, delta: -0.06 points, -0.06% relative)
- **Observations**: Switch accuracy was 90.20% versus EXP-010's 89.73%, and first-weak accuracy was 93.45% versus 93.16%. The run retained 98.73% of EXP-010's 26,898 steps and exceeded the attribution floor. Best accuracy arrived at epoch 65, then ended at 94.05%; final test loss was 0.2024 versus the baseline's 0.1934.
- **Analysis**: The intervention achieved its local objective: new paths recruited safely, improved strong-phase fit, and did not meaningfully compromise exposure. That gain did not survive as better generalization through the weak cosine tail. The worse final NLL and slightly lower best accuracy indicate that the random normalized transition basis changed representation quality in a way that favored early fit but not late confidence or classification. This rejects the exact learned projection-plus-BN operating point, not every form of anti-aliased shortcut downsampling.
- **Key Learning**: Learned normalized transition shortcuts improve short-horizon fit but slightly worsen late calibration and peak accuracy despite full exposure.

## Verification

- **Conditions**: primary improvement failed; all process-integrity conditions passed.
- **Review Notes**: Results are trustworthy. The run had 10 finite summary fields, exact model/scope/timer/lifecycle invariants, 19 unique evaluations, no retry, and 26,557 steps above the mechanism-support floor.
- **Verdict**: no-improvement
- **Verdict Basis**: `best_test_acc=94.09%` was 0.06 points below the moving baseline and 0.16 below the required 94.25%, while no hard constraint or protocol-integrity failure occurred.

## Unexplored Avenues

- Isolate pool-first anti-aliasing from learned channel mixing by retaining deterministic Option-A channel semantics after average pooling; this could test whether the projection basis or downsampling rule caused the NLL deficit.
- An identity-informed projection without a new normalization layer could preserve channel provenance while learning only missing channels, but it requires a fresh hypothesis and adversarial review because it changes initialization and branch scale.

## Next Steps

- **High confidence**: preserve the accepted Option-A shortcut and test a mechanism outside transition-path normalization; EXP-017 shows that higher switch fit alone is not the current limiter.
- **Medium confidence**: investigate lightweight feature recalibration on identity-preserving residual outputs, with a zero/identity start and production-batch first-update gate.
- **Medium confidence**: test deterministic anti-aliased downsampling separately from learned projection or BN so each transition-path effect is attributable.

