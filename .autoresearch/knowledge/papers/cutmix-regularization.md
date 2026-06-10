# CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features
- **Authors**: Sangdoo Yun, Dongyoon Han, Seong Joon Oh, Sanghyuk Chun, Junsuk Choe, Youngjoon Yoo
- **Venue**: ICCV 2019
- **URL**: https://arxiv.org/abs/1905.04899

## Key Contributions
- Introduces CutMix, which replaces a rectangular patch in one image with a patch from another image and mixes labels by the replaced area.
- Retains informative pixels unlike Cutout while preserving a regional regularization effect.
- Reports consistent classification gains on CIFAR and ImageNet benchmarks.

## Relevance
CutMix is a distinct alternative to the failed whole-image mixup and erased-patch Cutout experiments in this repo. It keeps local image statistics realistic while adding regional label mixing inside `train.py`, with no evaluation-harness changes.

## Key Techniques
- Sample a beta-distributed lambda and convert it to a rectangular patch area.
- Permute the batch on-device, paste source patches into target images, and recompute lambda from the actual clipped box area.
- Train with a weighted two-target cross-entropy loss using the area-adjusted lambda.
