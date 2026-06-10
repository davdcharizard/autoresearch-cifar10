# Improved Regularization of Convolutional Neural Networks with Cutout

- Source: https://arxiv.org/abs/1708.04552
- Authors: Terrance DeVries, Graham W. Taylor
- Relevance: CIFAR image-classification regularization.

## Key Takeaway

Cutout masks square image regions during training and can be combined with ordinary augmentation and other regularizers. For this repo, the closest low-risk implementation is tensor-space `transforms.RandomErasing` after `ToTensor()` and before normalization-sensitive model input.

## Use In This Project

Try cutout-style masking before architecture changes because it is cheap, stays fully inside `train.py`, and directly targets the baseline's minimal crop/flip augmentation.
