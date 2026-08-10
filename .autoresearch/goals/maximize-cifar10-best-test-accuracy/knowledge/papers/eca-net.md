# ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks
- **Authors**: Qilong Wang, Banggu Wu, Pengfei Zhu, Peihua Li, Wangmeng Zuo, Qinghua Hu
- **Venue**: CVPR 2020
- **URL**: https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_ECA-Net_Efficient_Channel_Attention_for_Deep_Convolutional_Neural_Networks_CVPR_2020_paper.html

## Key Contributions
- Introduces channel attention without the dimensionality-reduction bottleneck used by squeeze-and-excitation blocks.
- Uses global spatial pooling followed by a tiny one-dimensional convolution across channels and a sigmoid gate.
- Reports consistent accuracy gains across ResNet backbones with negligible parameter and FLOP additions at ImageNet scale.

## Relevance

The accepted width-2 ResNet has enough capacity to fit strong views but may not allocate it optimally across channels under RandAugment and CutMix. ECA can recalibrate each block's residual features before addition without initializing the branch toward identity and with only a few parameters. Its global descriptor may interact poorly with spatially mixed CutMix regions, and the paper does not directly validate a 20-layer CIFAR network under a 300-second budget, so paired timing and first-update gates remain necessary.

## Key Techniques
- Global-average-pool a block output to one descriptor per channel.
- Apply a small odd-kernel `Conv1d(1, 1, padding=k//2, bias=False)` along the channel axis.
- Sigmoid the result and multiply it into residual features before shortcut addition.
- Choose the channel-interaction kernel deterministically from channel count rather than tuning it on the target run.
