# Gradually Updated Neural Networks for Large-Scale Image Recognition
- **Authors**: Siyuan Qiao, Zhishuai Zhang, Wei Shen, Bo Wang, Alan Yuille
- **Venue**: ICML 2018
- **URL**: https://proceedings.mlr.press/v80/qiao18b.html

## Key Contributions
- Adds a computation ordering across channels to increase effective depth without additional nominal computation.
- Argues that the ordering removes overlap singularities and improves convergence.
- Reports gains on CIFAR and ImageNet.

## Relevance
The work supports changing how an existing convolution uses channels rather than adding auxiliary attention paths. Its custom channel-wise topology is less implementation-safe than macro-level block reallocation, but it motivates prioritizing representation changes that preserve the number of major convolutions.

## Key Techniques
- Channel groups updated in a fixed sequential order.
- Structured reuse of existing convolutional computation.
- Drop-in changes at convolution or block granularity.
