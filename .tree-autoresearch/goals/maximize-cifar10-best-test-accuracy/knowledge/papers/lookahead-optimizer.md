# Lookahead Optimizer: k steps forward, 1 step back
- **Authors**: Michael R. Zhang, James Lucas, Geoffrey Hinton, Jimmy Ba
- **Venue**: NeurIPS 2019
- **URL**: https://papers.nips.cc/paper_files/paper/2019/hash/90fd4f88f588ae64038134f1eeaa023f-Abstract.html

## Key Contributions
- Wraps any standard optimizer with `k` fast updates followed by interpolation of slow weights toward the fast endpoint.
- Reduces optimizer variance using one additional parameter copy and amortized copy/interpolation work.
- Reports CIFAR-10/100 and ImageNet benefits with common settings such as `k=5`, `alpha=0.5`.

## Relevance
Lookahead is compatible with a fixed-forward budget and Nesterov SGD, but its slow weights already form an online exponential average. Combining it with EXP-011's evaluation EMA creates nested smoothing; momentum retention, BatchNorm-buffer exclusion, SAM ordering, and EMA sampling order must be explicit.

## Key Techniques
- Interpolate parameters only after the inner optimizer update.
- Retain inner optimizer state under the canonical wrapper policy.
- Audit slow/fast ownership, RNG neutrality, cadence coincidences, BN mismatch, and downstream averaging interaction.
