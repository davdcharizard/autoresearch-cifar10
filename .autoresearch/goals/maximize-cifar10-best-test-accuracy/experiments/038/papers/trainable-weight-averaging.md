# Trainable Weight Averaging: Efficient Training by Optimizing Historical Solutions
- **Authors**: Tao Li, Zhehao Huang, Qinghua Tao, Yingwen Wu, Xiaolin Huang
- **Venue**: ICLR 2023
- **URL**: https://openreview.net/forum?id=8wbnpOJY-f
- **Availability**: OpenReview full text was unavailable; distilled from indexed PDF text and search snippets.

## Key Contributions
- Optimizes coefficients in the low-dimensional subspace spanned by historical checkpoints instead of using fixed uniform or exponential weights.
- Applies projected optimization at the head stage and reports improved generalization with fewer CIFAR training epochs.
- Attributes gains over SWA/EMA to lower coefficient-estimation error and the small subspace's resistance to overfitting.

## Relevance
EXP018 rejected one uniform weak-tail SWA window, not all historical-weight reuse. TWA is distinct but needs multiple checkpoints, extra coefficient optimization, and BatchNorm handling; under the 300-second counter it is probably too complex unless the projected phase replaces rather than supplements ordinary training.

## Key Techniques
- Store a small set of historical parameter vectors.
- Orthonormalize their difference subspace and optimize only averaging coefficients.
- Apply at the training head stage and explicitly manage BatchNorm buffers.
