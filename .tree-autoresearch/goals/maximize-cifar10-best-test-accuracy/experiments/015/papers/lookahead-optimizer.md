# Lookahead Optimizer: k steps forward, 1 step back
- **Authors**: Michael R. Zhang, James Lucas, Geoffrey Hinton, Jimmy Ba
- **Venue**: NeurIPS 2019
- **URL**: https://papers.nips.cc/paper_files/paper/2019/hash/90fd4f88f588ae64038134f1eeaa023f-Abstract.html

## Key Contributions
- Wraps a standard optimizer with `k` fast updates followed by interpolation of slow weights toward the fast endpoint.
- Reduces inner-optimizer variance with one extra parameter copy and amortized copy/interpolation work.
- Reports CIFAR-10/100 and ImageNet improvements, including robustness around the common `k=5`, `alpha=0.5` setting.

## Relevance
Lookahead is a low-forward-cost way to change the online trajectory while retaining Nesterov SGD, and therefore fits the fixed wall-clock budget better than multi-view or extra-gradient methods. However, its slow weights are themselves an EMA of fast endpoints, so combining it with EXP-011's charged-time evaluation EMA may double-smooth the trajectory. Momentum-state policy and ordering relative to SAM and EMA samples must be explicit.

## Key Techniques
- Maintain slow parameters initialized from online parameters.
- Every `k` optimizer steps, update `slow += alpha*(fast-slow)` and copy slow back to fast.
- Preserve the inner optimizer's momentum state unless a different policy is preregistered.
- Audit interpolation cadence, online displacement, RNG neutrality, optimizer ownership, and interaction with evaluation EMA.
