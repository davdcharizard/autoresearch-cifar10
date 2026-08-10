# Report EXP-005: Early Weak-Phase Adaptation
- **Created**: 2026-08-05

## Goal

Raise `best_test_acc` from 92.30% to at least 92.40% under the fixed one-H20 protocol.

## Idea & Hypothesis

Move only the RandAugment-to-crop/flip switch from 80% to 75%, leaving LR at 0.1 until 80%. Claude selected this as a clean test of high-LR clean-objective and BatchNorm adaptation.

## Approach

Added `AUG_SWITCH_FRACTION=0.75` and changed exactly the post-batch break and loader-switch predicates. RandAugment strength, LR schedule, model, loss, workers, evaluation, and seed remained fixed.

## Execution

One fixed-seed run executed on an idle H20 with no retry or error. The switch occurred once at epoch 74 and 75.0%; all eight workers stopped, weak training resumed, and steps through 78.5% retained `lr=0.1`.

## Results

- **Primary metric**: `92.12%` (baseline: `92.30%`, delta: `-0.18` points, `-0.20%` relative)
- **Observations**: Final accuracy was 91.98% and loss 0.2624. The run preserved 38,234 steps, 99 epochs, 300.0 counted seconds, 339.6 total seconds, 330.1 MB VRAM, and 269,722 parameters.
- **Analysis**: The intended clean high-LR interval executed without a throughput or lifecycle confound, but accuracy regressed. The local evidence therefore favors retaining strong augmentation until the LR transition: the extra 15 seconds of weak high-LR training likely replaced useful invariance learning rather than improving clean adaptation. Augmentation draws necessarily changed, so exact effect size is not causal, but the setting clearly failed the declared metric.
- **Key Learning**: Five percent of weak high-LR adaptation reduced accuracy 0.18 points; preserve strong augmentation through the 80% LR boundary.

## Verification

- **Conditions**: Primary accuracy failed (`92.12% < 92.40%`); remaining formal conditions were skipped.
- **Review Notes**: Result confirmed trustworthy: exact scoped diff, fixed seed, idle H20, correct 75% switch and high-LR interval, clean exit, complete summary, and unchanged evaluator.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid metric was 0.18 points below the moving baseline.

## Unexplored Avenues

- Preserve the 80% switch and test a nearby magnitude; phase-duration exploitation is not supported by this result.
- Same-width preactivation remains an orthogonal architecture test with low throughput risk.

## Next Steps

- **High confidence**: preserve the exact EXP-004 phase boundary and let external review compare magnitude tuning with an orthogonal architecture lever.
- **Medium confidence**: test magnitude 8 rather than 9 as a narrower augmentation-strength step.
- **Medium confidence**: revisit preactivation now that boundary tuning regressed.

## Exit Action Results

- None defined.
