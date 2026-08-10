# Deep Networks with Stochastic Depth

- **Authors**: Gao Huang, Yu Sun, Zhuang Liu, Daniel Sedra, Kilian Weinberger
- **Venue**: ECCV 2016
- **URL**: https://arxiv.org/abs/1603.09382

## Key Contributions

- Randomly bypasses residual blocks during training and uses the full network with survival-probability scaling at evaluation.
- Shortens the expected training graph, regularizes the ensemble of effective depths, and enabled very deep residual networks to train.
- Reports strong CIFAR-10 results, including 4.91% error for a 1202-layer network.

## Relevance

Stochastic depth can jointly attack the measured backward bottleneck and generalization, unlike a pure regularizer that only consumes more counted time. Transfer is weak: the published headline gains are on much deeper networks, whereas the accepted network has only nine residual blocks and already needs strong-phase fit. Dropping even one block is therefore a large representation perturbation.

## Local Requirements

- Keep the stem, transitions, channel padding, accepted initialization, data, optimizer, and schedule unchanged.
- Use one predeclared survival schedule and deterministic per-step RNG semantics; never choose rates after observing accuracy.
- Include bypass/scaling operations inside counted time and equalize evaluation looks.
- Gate exact-corpus class geometry, branch-use frequency, RNG replay, expected compute, and full-function evaluation before production.
