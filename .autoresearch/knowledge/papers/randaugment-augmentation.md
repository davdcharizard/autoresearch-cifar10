# RandAugment: Practical Automated Data Augmentation

- Source: https://papers.neurips.cc/paper_files/paper/2020/file/d85b63ef0ccb114d0a3bb7b7d808028f-Paper.pdf
- Implementation reference: https://docs.pytorch.org/vision/main/generated/torchvision.transforms.RandAugment.html
- Relevance: CIFAR-style image augmentation without adding dependencies.

## Key Takeaway

RandAugment reduces automated augmentation to two main parameters: how many augmentation operations to apply and the magnitude of those operations. Torchvision provides a built-in `transforms.RandAugment` implementation, so this repo can test policy augmentation by editing only the existing training transform in `train.py`.

## Use In This Project

Use conservative settings first, such as `num_ops=1` and a low magnitude, because this fixed-budget loop already has strong crop/flip/reflection, label smoothing, and weight decay. Strong policies may over-regularize or add dataloader overhead, so verify total wall-clock stays under 10 minutes and compare against the +0.10 percentage-point threshold.
