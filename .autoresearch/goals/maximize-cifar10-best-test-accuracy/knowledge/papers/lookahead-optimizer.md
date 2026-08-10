# Lookahead Optimizer: k steps forward, 1 step back
- **Authors**: Michael Zhang, James Lucas, Jimmy Ba, Geoffrey E. Hinton
- **Venue**: NeurIPS 2019
- **URL**: https://papers.nips.cc/paper_files/paper/2019/hash/90fd4f88f588ae64038134f1eeaa023f-Abstract.html

## Key Contributions
- Maintains ordinary fast optimizer weights plus slow weights; every `k` steps the slow point interpolates toward fast weights and is copied back into the live model.
- Reduces trajectory variance with one extra parameter copy and amortized parameter-wise arithmetic.
- Reports CIFAR-10/100 gains with SGD-family optimizers and robustness around `k=5`, `alpha=0.5`; retaining inner optimizer state is a supported operating mode.

## Relevance
The accepted CIFAR recipe has a long noisy LR-0.1 RandAugment/CutMix phase and a fixed short horizon. Lookahead can change that trajectory without another backward pass or post-hoc stale endpoint. It is materially distinct from uniform SWA: the exponentially recent-weighted slow point feeds back into every later gradient.

Transfer is not guaranteed. The paper used longer horizons and different ResNets; interpolation filters committed displacement and coupled decay, while persistent momentum temporarily refers to the pre-copy trajectory. Local use needs synchronization-level safety and effective-progress diagnostics.

## Key Techniques
- Fixed `k=5`, slow step `alpha=0.5`.
- Preserve the inner momentum state across synchronizations.
- Count interpolation/copy overhead inside the training budget.
- Evaluate the live model; do not swap to a separate shadow endpoint.
