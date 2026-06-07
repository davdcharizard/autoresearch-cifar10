# Report EXP-016: TTA (horizontal flip)
- **Created**: 2026-05-29

## Goal
Maximize CIFAR-10 test accuracy. Baseline: 95.73% (EXP-007).

## Results
- **Primary metric**: 96.39% (baseline: 95.73%, delta: +0.66%)
- **Observations**: TTA with horizontal flip is a pure evaluation-time improvement — training is identical to EXP-007. The model averages predictions from the original and flipped input, reducing left-right asymmetry noise. best/final gap 0.13%.
- **Key Learning**: TTA with horizontal flip adds +0.66% for free. This compounds with all training improvements since it only affects evaluation. The eval is 2x slower but eval time isn't in the training budget.
- **Verdict**: improvement

## Next Steps
1. Add more TTA augmentations (crops, translations) for further gains
2. Combine TTA with other improvements (wider model, different training tweaks)

## Exit Action Results
