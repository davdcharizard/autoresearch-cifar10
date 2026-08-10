# Generalization Analysis of Stochastic Weight Averaging with General Sampling
- **Authors**: Peng Wang, Li Shen, Zerui Tao, Shuaida He, Dacheng Tao
- **Venue**: ICML 2024
- **URL**: https://proceedings.mlr.press/v235/wang24bl.html

## Key Contributions
- Establishes stochastic-weight-averaging stability bounds for nonconvex optimization with both replacement and without-replacement sampling.
- Derives sharper generalization bounds for SWA than for the corresponding SGD iterate.
- Supports the theoretical result with experiments across several benchmarks.

## Relevance

The accepted CIFAR run uses shuffled without-replacement minibatches and a noisy weak cosine tail whose best and final checkpoints can differ. Late averaging could improve the evaluated solution with negligible training-kernel cost, directly targeting late generalization rather than further increasing strong-phase fit. The local BatchNorm buffers and very short 60-second tail make averaging-window and buffer semantics the principal risks.

## Key Techniques
- Average parameter iterates from a deliberately selected portion of the optimization trajectory.
- Treat without-replacement sampling explicitly rather than assuming independent gradient samples.
- Compare the averaged solution against the terminal SGD iterate under the same training path.

