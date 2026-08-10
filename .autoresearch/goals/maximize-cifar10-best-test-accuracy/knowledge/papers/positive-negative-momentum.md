# Positive-Negative Momentum: Manipulating Stochastic Gradient Noise to Improve Generalization
- **Authors**: Zeke Xie, Li Yuan, Zhanxing Zhu, Masashi Sugiyama
- **Venue**: ICML 2021
- **URL**: https://proceedings.mlr.press/v139/xie21h.html

## Key Contributions
- Alternates two momentum streams and combines them with positive/negative coefficients to amplify stochastic-gradient noise without a second gradient evaluation.
- Reports CIFAR-10 ResNet-18 error `4.48 +/- 0.09` versus `5.01 +/- 0.03` for momentum SGD.
- Uses default `beta0=1` and normalization by `sqrt(5)`; official code exposes coupled or decoupled decay and defaults to decoupled.

## Relevance
PNM is close external evidence for low-cost optimizer-shaped generalization, but its paper recurrence is not scale-compatible with PyTorch momentum at the same numeric LR. With `mu=0.9`, each alternating EMA approaches `d`, producing direction `d/sqrt(5)`, while PyTorch's buffer approaches `10d`: roughly a 22.36x deterministic drift gap. A local PNM experiment must explicitly solve effective-scale matching and decay semantics before claiming a noise-only comparison.

## Key Techniques
- Maintain odd/even buffers with decay `mu**2` and injection `1-mu**2`.
- Combine current/previous buffers as `((1+beta0)*current-beta0*previous)/sqrt((1+beta0)^2+beta0^2)`.
- Persist exact post-transform safety batches and verify alternating recurrence, coupled/decoupled decay, and effective update scale before a fixed-budget run.
