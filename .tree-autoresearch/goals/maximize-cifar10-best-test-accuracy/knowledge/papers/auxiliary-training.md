# Auxiliary Training: Towards Accurate and Robust Models
- **Authors**: Linfeng Zhang, Muzhou Yu, Tong Chen, Zuoqiang Shi, Chenglong Bao, Kaisheng Ma
- **Venue**: CVPR 2020
- **URL**: https://openaccess.thecvf.com/content_CVPR_2020/html/Zhang_Auxiliary_Training_Towards_Accurate_and_Robust_Models_CVPR_2020_paper.html

## Key Contributions
- Uses auxiliary classifiers during training and discards them at inference.
- Transfers auxiliary information into the primary classifier with input-aware self-distillation and classifier-alignment regularization.
- Reports clean-accuracy and robustness gains on CIFAR-10, CIFAR-100, and ImageNet.

## Relevance
This independently supports training-only auxiliary classifiers as a representation lever. Its full gains depend on corrupted inputs, selective BatchNorm, distillation, and alignment, so a plain intermediate companion CE inherits only a mechanism prior, not the reported effect size.

## Key Techniques
- Disposable auxiliary classifiers.
- Input-aware auxiliary-to-primary distillation.
- Selective normalization and late classifier alignment.
