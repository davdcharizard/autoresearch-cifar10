# Positive-Negative Momentum: Manipulating Stochastic Gradient Noise to Improve Generalization
- **Authors**: Zeke Xie, Li Yuan, Zhanxing Zhu, Masashi Sugiyama
- **Venue**: ICML 2021
- **URL**: https://proceedings.mlr.press/v139/xie21h.html

## Key Contributions
- Splits momentum into odd/even gradient streams and combines them with positive and negative coefficients to amplify stochastic-gradient noise without a second gradient evaluation.
- Normalizes the update by `sqrt((1 + beta0)^2 + beta0^2)`; the paper uses `beta0=1` as its default, corresponding to a fivefold noise-variance multiplier before normalization.
- Reports ResNet-18 CIFAR-10 error `4.48 +/- 0.09` for PNM versus `5.01 +/- 0.03` for momentum SGD, plus gains across other models/datasets.
- Gives convergence and PAC-Bayesian arguments for stronger gradient noise and improved generalization, while acknowledging the long-time/local-quadratic assumptions.

## Relevance
The accepted run is already generalization-limited and uses momentum SGD at batch 128. PNM directly changes stochastic optimization noise at low model cost, unlike Nesterov's ambiguous extra current-gradient weighting. Its reported CIFAR/ResNet comparison is unusually close to this goal, although the architecture, augmentation, schedule, decay semantics, and fixed-time horizon differ. The official implementation defaults to decoupled decay, whereas accepted PyTorch SGD uses coupled decay and local EXP008/009 show decay semantics matter; a locally isolated PNM test must use the official `decoupled=False` path and treat the reported gain as non-exact evidence.

## Key Techniques
- Maintain two alternating momentum streams with decay `momentum**2` and injection `1-momentum**2`.
- Combine current and previous stream as `((1+beta0)*m_current - beta0*m_previous) / sqrt((1+beta0)^2 + beta0^2)`.
- Apply coupled decay consistently before the PNM recurrence and preflight the first two alternating updates exactly.
