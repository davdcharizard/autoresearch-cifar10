# Plan EXP-067: AutoAugment CIFAR10 policy
## Changes
Replace `transforms.RandomHorizontalFlip()` with `transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10)` in the training transform pipeline.
