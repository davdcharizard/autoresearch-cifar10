# Demystifying Loss Functions for Classification
- **Authors**: Simon Kornblith et al.
- **Venue**: ICLR 2021 submission
- **URL**: https://openreview.net/forum?id=jNTeYscgSw8
- **Availability**: Full text was unavailable through OpenReview; distilled from indexed paper snippets.

## Key Contributions
- Compares softmax cross-entropy with squared error, cosine softmax, logit normalization, logit penalties, label smoothing, dropout, and sigmoid losses.
- Shows that classifier-loss choices alter implicit regularization and optimization, including in BatchNorm architectures.
- Reports tuned CIFAR-10 settings; normalized-logit and cosine variants require substantially different learning rates and explicit temperatures.

## Relevance
The deferred cosine head has external support but is not a drop-in scale-8 change: published CIFAR settings jointly tune temperature, LR, and decay. This strengthens the need for an exact bounded-logit oracle and a trajectory preflight, while weakening confidence in a single arbitrarily transferred scale.

## Key Techniques
- Normalize logits or classifier geometry with an explicit temperature.
- Test loss-scale and gradient-scale consequences rather than judging only bounded logits.
- Couple head geometry to optimizer operating point; avoid assuming the accepted global LR transfers unchanged.
