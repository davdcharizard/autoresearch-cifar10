# ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks
- **Authors**: Qilong Wang, Banggu Wu, Pengfei Zhu, Peihua Li, Wangmeng Zuo, Qinghua Hu
- **Venue**: CVPR 2020
- **URL**: https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_ECA-Net_Efficient_Channel_Attention_for_Deep_Convolutional_Neural_Networks_CVPR_2020_paper.html

## Key Contributions

- Uses global average pooling followed by a short one-dimensional convolution across channels to generate channel gates.
- Avoids the dimensionality-reduction bottleneck of squeeze-and-excitation while retaining local cross-channel interaction.
- Adds only a handful of parameters and negligible published FLOPs; ECA-Net50 reports 80 added parameters and about a 2.28-point ImageNet top-1 gain over ResNet-50.

## Relevance

EXP-004 has extreme memory headroom but a tight wall-clock step budget, so ECA is an unusually cheap additive representation mechanism. Its direct evidence is not matched to a shallow CIFAR WRN, and standard sigmoid gating initially scales features by about one half; an experiment therefore needs a preregistered placement and initialization contract rather than assuming published ImageNet gains transfer.

## Key Techniques

- Global average pooling to a channel descriptor.
- Odd-kernel one-dimensional convolution over channels without dimensionality reduction.
- Sigmoid channel gating with kernel size selected from channel width.
