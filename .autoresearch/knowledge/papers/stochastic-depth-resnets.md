# Deep Networks with Stochastic Depth

**Source**: https://arxiv.org/abs/1603.09382

## Summary
Stochastic depth trains residual networks with some residual branches randomly bypassed during training while using the full network at evaluation time. The method is designed for residual architectures and can act as a structural regularizer without changing inference-time parameters.

## Relevance to This Project
- Fits the `train.py`-only constraint by adding training-only behavior inside `BasicBlock`.
- Targets generalization through residual-branch co-adaptation rather than image augmentation, scalar LR tuning, width, batch size, or weight decay.
- Should add little parameter or evaluation overhead, but per-block randomness can still affect `torch.compile` behavior and fixed-budget throughput.

## Autoresearch Notes
Use very mild drop probabilities for this short ResNet-20-style model. The local history already shows identity-biased residual initialization can undertrain, so stochastic depth should be treated as a distinct train-time regularizer and kept conservative.
