# Bag of Tricks for Image Classification with Convolutional Neural Networks
- **Authors**: Tong He, Zhi Zhang, Hang Zhang, Zhongyue Zhang, Junyuan Xie, Mu Li
- **Venue**: CVPR 2019
- **URL**: https://openaccess.thecvf.com/content_CVPR_2019/html/He_Bag_of_Tricks_for_Image_Classification_with_Convolutional_Neural_Networks_CVPR_2019_paper.html

## Key Contributions
- Separates training refinements from small ResNet architecture tweaks and evaluates their accuracy effects.
- ResNet-D replaces lossy stride-2 shortcut projection with average pooling followed by a stride-1 `1x1` projection.
- Frames downsampling placement as an information-preservation choice that can improve accuracy at small relative model cost.

## Relevance

The accepted CIFAR ResNet20 uses an even more austere Option-A shortcut: raw `::2` spatial sampling followed by zero channel padding. An average-pool plus learned `1x1` projection and BatchNorm at the two transition blocks would preserve all input positions and learn channel transport while leaving every ordinary postactivation residual branch active. The paper's direct experiments are ImageNet bottleneck ResNets, so it supports the mechanism but not the effect size for this width-2 CIFAR model or CutMix recipe.

## Key Techniques
- At each stage transition, downsample the shortcut with fixed `2x2`, stride-2 average pooling.
- Apply a learned stride-1 `1x1` convolution after pooling rather than striding the projection itself.
- Normalize the projected shortcut before residual addition and keep the main residual path unchanged.
