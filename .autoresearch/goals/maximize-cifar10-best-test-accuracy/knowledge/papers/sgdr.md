# SGDR: Stochastic Gradient Descent with Warm Restarts
- **Authors**: Ilya Loshchilov, Frank Hutter
- **Venue**: ICLR 2017
- **URL**: https://arxiv.org/abs/1608.03983

## Key Contributions
- Introduces cosine learning-rate annealing with warm restarts for strong anytime performance.
- Demonstrates gains on CIFAR-10 and CIFAR-100.
- Uses a smooth per-batch schedule instead of sparse hand-chosen step drops.

## Relevance
The baseline's milestones are fixed at steps 32,000 and 48,000, even though the wall-clock budget may produce a different total step count. A schedule parameterized by elapsed budget fraction can anneal reliably to a small learning rate before termination and should improve final convergence at negligible cost.

## Key Techniques
- Anneal learning rate with a cosine curve within a cycle.
- Optionally restart to a larger learning rate for anytime ensembling behavior.
- Express schedule progress in the actual training horizon rather than assumed epochs.

