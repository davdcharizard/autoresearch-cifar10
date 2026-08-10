# ShakeDrop Regularization for Deep Residual Learning
- **Authors**: Yoshihiro Yamada, Masakazu Iwamura, Takuya Akiba, Koichi Kise
- **Venue**: IEEE Access 2019; ICLR 2018 workshop precursor
- **URL**: https://arxiv.org/abs/1802.02375

## Key Contributions
- Introduces stochastic residual-branch scaling for ResNet, Wide ResNet, PyramidNet, and ResNeXt.
- Emphasizes a stabilizer that permits strong stochastic regularization without destabilizing training.

## Relevance
The baseline is a small residual network and can accept stochastic residual regularization without dependencies. The method is higher risk under a short budget because its published gains often accompany deeper or wider networks and longer schedules.

## Key Techniques
- Per-block stochastic residual scaling during training.
- Depth-dependent survival or disturbance schedule.
