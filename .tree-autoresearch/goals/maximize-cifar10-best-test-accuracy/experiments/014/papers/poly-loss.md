# PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions
- **Authors**: Zhaoqi Leng, Mingxing Tan, Chenxi Liu, Ekin Dogus Cubuk, Xiaojie Shi, Shuyang Cheng, Dragomir Anguelov
- **Venue**: ICLR 2022
- **URL**: https://openreview.net/forum?id=gSdSJoenupI

## Key Contributions
- Represents classification objectives as weighted polynomial bases in target error probability.
- Proposes Poly-1, cross-entropy plus `epsilon * (1 - p_t)`, as a one-parameter, low-overhead loss change.
- Shows that useful coefficients depend on the dataset and task rather than transferring universally.

## Relevance
EXP-011 is throughput-sensitive and already uses soft CutMix targets, so a one-line loss change is attractive only if target probability and gradient inflation are defined coherently for both hard and mixed labels. A single coefficient must be preregistered from a bounded gradient-scale rationale, not selected from test accuracy.

## Key Techniques
- Compute target probability from the same hard or mixed target distribution used by cross-entropy.
- Audit hard-label and two-label CutMix gradients separately.
- Apply the identical objective to ordinary and both SAM passes.
