# PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions
- **Authors**: Zhaoqi Leng, Mingxing Tan, Chenxi Liu, Ekin Dogus Cubuk, Xiaojie Shi, Shuyang Cheng, Dragomir Anguelov
- **Venue**: ICLR 2022
- **URL**: https://openreview.net/forum?id=gSdSJoenupI

## Key Contributions
- Expresses classification losses as weighted polynomial bases in target error probability.
- Proposes Poly-1, `CE + epsilon*(1-p_t)`, as a one-line low-overhead modification with image-classification gains.
- Emphasizes that the best coefficient is task-dependent.

## Relevance
For one-hot softmax CE, Poly-1 rescales each example's logit gradient by `1 + epsilon*p_t`; it does not change that example's gradient direction. A future test should derive epsilon from a preregistered gradient-inflation budget or normalize the scale, and must define target probability coherently for CutMix.

## Key Techniques
- Adjust the leading `(1-p_t)` polynomial coefficient.
- Audit effective gradient inflation rather than describing the change only as loss geometry.
- Treat coefficient selection as part of the mechanism, not a free default.
