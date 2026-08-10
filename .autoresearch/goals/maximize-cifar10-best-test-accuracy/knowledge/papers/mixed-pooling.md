# Generalizing Pooling Functions in Convolutional Neural Networks
- **Authors**: Chen-Yu Lee, Patrick W. Gallagher, Zhuowen Tu
- **Venue**: AISTATS 2016
- **URL**: https://proceedings.mlr.press/v51/lee16a.html

## Key Contributions

- Generalizes conventional pooling by learning mixtures of max and average responses.
- Reports improved CNN performance and invariance across several benchmarks with modest parameter/compute overhead.
- Shows responsive/learned pooling can outperform a universal fixed statistic.

## Relevance

The accepted CIFAR model uses only global average aggregation. A zero-initialized class-specific max-logit branch makes average-only behavior an exact starting special case and lets training learn when peak spatial evidence helps. Local CutMix area semantics, launch overhead, and strong-phase fit still require direct verification.
