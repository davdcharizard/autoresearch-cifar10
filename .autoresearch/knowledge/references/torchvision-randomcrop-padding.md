# Torchvision RandomCrop Padding Modes

- Source: https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.RandomCrop.html
- Relevance: CIFAR train-time crop augmentation variants.

## Key Takeaway

`transforms.RandomCrop` supports `padding_mode` values beyond the default constant fill, including reflected boundary padding. This lets CIFAR crop-border statistics be changed with a one-line `train.py` edit, without altering crop size, padding amount, model code, optimizer settings, or dependencies.

## Use In This Project

EXP-029 validated `padding_mode="reflect"` on the 28/56/112 ResNet-20 anchor, reaching 93.58% best test accuracy. Future augmentation-boundary probes can compare sibling modes such as symmetric padding while preserving the successful reflection-padding anchor as the default.
