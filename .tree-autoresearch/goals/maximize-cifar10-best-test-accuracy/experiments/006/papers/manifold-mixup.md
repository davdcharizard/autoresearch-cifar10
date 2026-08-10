# Manifold Mixup: Better Representations by Interpolating Hidden States

- **Authors**: Vikas Verma et al.
- **Venue**: ICML 2019
- **URL**: https://proceedings.mlr.press/v97/verma19a.html

## Key Contributions

- Samples one eligible layer and one `lambda ~ Beta(alpha, alpha)` per minibatch, permutes the minibatch, and linearly mixes hidden representations and one-hot labels at that layer.
- Backpropagates through layers before and after the mix; the operation adds no model forward and negligible arithmetic.
- Encourages flatter class-conditional representations and smoother decision boundaries.
- Uses `alpha=2` in the main CIFAR experiments and finds robust gains across several eligible-layer sets.

## Relevance

The parent is limited by generalization while preserving nearly 25,560 steps and independent images. Manifold mixup retains every image and optimizer step while moving regularization from pixels into learned features. On CIFAR-10, error falls from 4.83 to 2.95 on PreActResNet-18, 4.64 to 2.54 on PreActResNet-34, and 3.99 to 2.55 on WRN-28-10. It also beats input Mixup materially. The paper does not test composition with CutMix, stochastic depth, or late SAM.

## Key Techniques

- Use a single permutation, lambda, and eligible layer per selected batch.
- The strongest reported CIFAR-10 eligible set is input plus the first two hidden boundaries `{0,1,2}`; deeper-only sets are weaker.
- A parent-compatible candidate should keep the existing overall early mixing probability and clean final quarter, while preregistering how CutMix and manifold mixup share selected batches.
- Exact target coefficients must follow the linear hidden mixture; CutMix area correction cannot be reused for the manifold branch.
