# mixup: Beyond Empirical Risk Minimization
- **Authors**: Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz
- **Venue**: ICLR 2018
- **URL**: https://arxiv.org/abs/1710.09412

## Key Contributions
- Trains on convex combinations of image pairs and their one-hot targets.
- Encourages locally linear behavior between examples and reduces memorization.
- Improves generalization on CIFAR-10 across PreAct ResNet, WideResNet, and DenseNet models.

## Relevance
Mixup is a low-memory, dependency-free regularizer that fits entirely in `train.py`. It is especially plausible for lifting the 91.67% ResNet20 baseline, although soft targets add implementation complexity and overly strong mixing can slow convergence inside a fixed 300-second budget.

## Key Techniques
- Sample lambda from a Beta distribution, commonly with alpha in the 0.1-0.4 range.
- Permute each minibatch and mix both inputs and labels.
- Compute the lambda-weighted cross-entropy of both target sets.

