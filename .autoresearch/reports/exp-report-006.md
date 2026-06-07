# Report EXP-006: k=4 + TrivialAugment + CutMix
- **Created**: 2026-05-28
- **Goal**: goals/maximize-cifar10-test-accuracy.md

## Goal
Maximize CIFAR-10 test accuracy. Baseline: 95.25% (EXP-004).

## Results
- **Primary metric**: 95.15% (baseline: 95.25%, delta: -0.10%)
- **Key Learning**: TrivialAugment + CutMix combined is too aggressive for 4.3M model with ~60 epochs. Added augmentation diversity prevented convergence to the same level.
- **Verdict**: no-improvement

## Next Steps
1. Try EMA (exponential moving average) + weight decay tuning at k=4
2. Try pre-activation at k=4
3. Try stochastic depth at k=4

## Exit Action Results
