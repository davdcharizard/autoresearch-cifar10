# ShakeDrop Regularization for Deep Residual Learning
- **Authors**: Yoshihiro Yamada, Masakazu Iwamura, Takuya Akiba, Koichi Kise
- **Venue**: ICLR 2018 Workshop / IEEE Access 2019
- **URL**: https://arxiv.org/abs/1802.02375

## Key Contributions
- Perturbs single residual branches with distinct stochastic forward and backward coefficients.
- Uses a stochastic-depth-style switch to stabilize stronger residual perturbations.
- Reports applicability to ResNet, Wide ResNet, PyramidNet, and ResNeXt on CIFAR.

## Relevance
ShakeDrop directly targets residual-network generalization with little parameter cost, but its intentionally strong disturbance is risky here: stronger CutMix, preactivation, and zero-gamma already suppressed short-horizon strong-phase fit. It is therefore best treated as evidence for a tightly gated, low-rate residual-noise proposal rather than copied at paper-default strength.

## Key Techniques
- Sample a depth-dependent Bernoulli gate for each residual branch.
- When gated, use randomized forward and backward multipliers; use expected scaling at evaluation.
- Increase disturbance with depth while retaining an unperturbed-network probability for stability.
