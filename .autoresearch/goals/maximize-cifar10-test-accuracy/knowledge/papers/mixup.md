# mixup: Beyond Empirical Risk Minimization
- **Authors**: Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz
- **Venue**: ICLR 2018
- **URL**: https://arxiv.org/abs/1710.09412

## Key Contributions
- Trains on convex combinations of pairs of examples and labels.
- Improves generalization on CIFAR-10 and other classification datasets by encouraging linear behavior between examples.

## Relevance
Mixup is implementable entirely in `train.py` with little GPU overhead and directly targets the baseline's generalization gap, though it may delay peak hard-label accuracy under a short training budget.

## Key Techniques
- Sample a beta-distributed interpolation coefficient.
- Interpolate shuffled image pairs and combine their cross-entropy losses with the same coefficient.
