# mixup: Beyond Empirical Risk Minimization

**Source**: https://arxiv.org/abs/1710.09412

## Summary
Mixup trains on convex combinations of input examples and their labels. For CIFAR-style image classification, it is a cheap regularization mechanism that smooths the empirical training distribution and can improve generalization without changing the model architecture or evaluation path.

## Relevance to This Project
- Fits the `train.py`-only constraint: image tensors and labels can be mixed inside the training loop.
- Targets late generalization and overfitting behavior rather than model capacity.
- Adds per-batch tensor work, so fixed-budget experiments must measure whether it delays LR milestones or reduces useful step coverage.

## Autoresearch Notes
EXP-042 attempted mild `alpha=0.1` mixup but crashed before final metrics, so the idea remains unproven in this repo. Any retry should first establish a reliable launch path and check whether mixup overhead still reaches the first LR drop.
