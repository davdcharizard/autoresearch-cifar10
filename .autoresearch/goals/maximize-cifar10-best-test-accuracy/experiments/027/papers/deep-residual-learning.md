# Deep Residual Learning for Image Recognition
- **Authors**: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
- **Venue**: CVPR 2016
- **URL**: https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html

## Key Contributions
- Introduces residual learning with identity shortcuts to make substantially deeper convolutional networks optimizable.
- Defines three dimension-changing shortcut choices: zero-padded identity (Option A), projection only when dimensions increase (Option B), and all projections (Option C).
- Uses economical Option A for the reported CIFAR models, while ImageNet ResNets use learned Option-B projections at stage changes.

## Relevance

The accepted model uses Option A at its two stage transitions. EXP017 and EXP021 tested pool-first ResNet-D variants, not the original stride-2 `1x1` Option-B shortcut. A standard projection is therefore a distinct, literature-native way to learn channel transport while retaining the accepted shortcut's spatial sampling geometry. Its CIFAR benefit is not established by the paper, and the extra BatchNorm/projection path must pass exact recruitment and fixed-time gates.

## Key Techniques
- At a dimension-changing block, replace `x[:, :, ::2, ::2]` plus zero padding with a stride-2 `1x1` convolution and BatchNorm.
- Keep identity shortcuts for same-shape blocks.
- Compare added projection cost and early optimization behavior rather than assuming an ImageNet result transfers to CIFAR.
