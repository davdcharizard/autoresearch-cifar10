# Demystifying Loss Functions for Classification
- **Authors**: Simon Kornblith et al.
- **Venue**: ICLR 2021 submission
- **URL**: https://openreview.net/forum?id=jNTeYscgSw8

## Key Contributions
- Compares softmax cross-entropy with squared error, cosine softmax, logit normalization, logit penalties, label smoothing, dropout, and sigmoid losses.
- Shows that classifier-loss choices alter implicit regularization and optimization, including in BatchNorm architectures.
- Reports tuned CIFAR-10 settings in which normalized-logit and cosine variants use explicit temperatures and substantially different optimizer settings.

## Relevance
Angular or logit-normalized heads are not scale-free drop-ins. Any local cosine classifier needs an explicit non-accuracy calibration, formula/gradient oracles, and long trajectory checks; matching output scale does not match Jacobian geometry or make the accepted LR/decay automatically safe.

## Key Techniques
- Normalize features/logits or classifier rows with explicit temperature.
- Measure loss and gradient scale, not only a logit bound.
- Treat head geometry and optimizer operating point as coupled.
