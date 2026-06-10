# SGDR: Stochastic Gradient Descent with Warm Restarts

- Source: https://arxiv.org/abs/1608.03983
- Authors: Ilya Loshchilov, Frank Hutter
- Relevance: SGD learning-rate scheduling for CIFAR-style training.

## Key Takeaway

Cosine annealing and restart schedules can improve anytime performance for SGD on CIFAR tasks. For this fixed wall-clock benchmark, a no-restart cosine decay over the expected step horizon is a simple first test before introducing restart complexity.

## Use In This Project

Replace abrupt `MultiStepLR` drops with `CosineAnnealingLR` over the expected maximum number of optimizer steps, then evaluate whether smoother decay improves peak test accuracy.
