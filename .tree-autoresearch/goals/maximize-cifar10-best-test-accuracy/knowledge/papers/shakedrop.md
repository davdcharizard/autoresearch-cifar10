# ShakeDrop Regularization for Deep Residual Learning
- **Authors**: Yoshihiro Yamada, Masakazu Iwamura, Takuya Akiba, Koichi Kise
- **Venue**: IEEE Access 2019; ICLR 2018 workshop precursor
- **URL**: https://arxiv.org/abs/1802.02375

## Key Contributions
- Introduces stochastic residual-branch scaling for ResNet, Wide ResNet, PyramidNet, and ResNeXt.
- Identifies stabilization as essential when residual disturbance is strong.

## Relevance
Short-budget CIFAR experiments can use conservative, expectation-preserving residual dropping, but should ablate it first when optimization underfits.

## Key Techniques
- Per-block stochastic residual scaling.
- Depth-dependent disturbance schedules and stabilization.
