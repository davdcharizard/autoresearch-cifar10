# PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions
- **Authors**: Zhaoqi Leng, Mingxing Tan, Chenxi Liu, Ekin Dogus Cubuk, Xiaojie Shi, Shuyang Cheng, Dragomir Anguelov
- **Venue**: ICLR 2022
- **URL**: https://openreview.net/forum?id=gSdSJoenupI

## Key Contributions
- Expresses classification losses as weighted polynomial bases in `(1-p_t)` and shows cross-entropy/focal loss are special cases.
- Proposes Poly-1, cross-entropy plus one tunable first-order term, as a one-line low-overhead modification.
- Reports top-line gains over cross-entropy on 2D image classification and other perception tasks, while emphasizing that the best coefficient is task-dependent.

## Relevance
Poly-1 directly targets the stable accuracy limiter with essentially no model or loader overhead, preserving the complete CutMix/SAM/EMA schedule. The main uncertainty is adapting target probability to CutMix's two soft labels and choosing the coefficient without a sweep; a faithful candidate can use the area-weighted probability assigned to the mixed target and preregister the paper's simple coefficient.

## Key Techniques
- Cross-entropy has polynomial coefficients `1/j`; Poly-1 changes the coefficient of the first `(1-p_t)` basis.
- The simplest loss is `CE + epsilon_1 * (1-p_t)`.
- The optimal coefficient depends on dataset/task, making an accuracy-blind coefficient rationale important here.
