# Dual Focal Loss for Calibration
- **Authors**: Linwei Tao, Minjing Dong, Chang Xu
- **Venue**: ICML 2023
- **URL**: https://proceedings.mlr.press/v202/tao23a.html

## Key Contributions
- Adds a second high-probability logit to focal weighting to reduce both over- and under-confidence.
- Reports broad calibration improvements across CIFAR-10/100 models without requiring post-hoc temperature scaling.
- Separates calibration quality from classification accuracy: on CIFAR-10 WideResNet-26-10, test error was 3.96% versus 3.86% for plain cross-entropy, despite much better calibration.

## Relevance

This is a caution against choosing a calibration-specific loss solely because EXP-011 has low test loss. Calibration gains need not raise top-1 accuracy, and the published WideResNet CIFAR-10 result slightly favors cross-entropy. It remains a useful rejected comparator for Poly-1 and classifier-temperature proposals.

## Key Techniques
- Ground-truth and strongest competing probability jointly define the focal factor.
- Adaptive focal variants require validation feedback and are incompatible with this frozen no-extra-validation protocol.
