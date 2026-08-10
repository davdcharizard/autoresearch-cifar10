# Auxiliary Training: Towards Accurate and Robust Models
- **Authors**: Linfeng Zhang, Muzhou Yu, Tong Chen, Zuoqiang Shi, Chenglong Bao, Kaisheng Ma
- **Venue**: CVPR 2020
- **URL**: https://openaccess.thecvf.com/content_CVPR_2020/html/Zhang_Auxiliary_Training_Towards_Accurate_and_Robust_Models_CVPR_2020_paper.html

## Key Contributions
- Adds auxiliary classifiers during training and discards them at inference.
- Transfers auxiliary information into the primary classifier with input-aware self-distillation and classifier-alignment regularization.
- Reports clean-accuracy and robustness gains on CIFAR-10, CIFAR-100, and ImageNet; the full method also uses corrupted samples and selective BatchNorm.

## Relevance
Together with the existing AISTATS Deeply-Supervised Nets note, this supplies independent evidence that training-only auxiliary classifiers can improve the deployed primary network. A plain pooled intermediate companion CE is much cheaper and cleaner for EXP004, but it omits the paper's corruption, distillation, selective-BN, and alignment package, so those headline gains do not transfer directly.

## Key Techniques
- Auxiliary classifiers used only during training.
- Input-aware distillation from auxiliary to primary predictions.
- Late classifier-weight alignment and separate normalization for corrupted samples.
