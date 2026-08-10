# Manifold Mixup

- Source: Vikas Verma et al., ICML 2019, https://proceedings.mlr.press/v97/verma19a.html

## Core Method

Sample one eligible layer, one minibatch permutation, and one `lambda ~ Beta(alpha, alpha)`. Linearly interpolate the hidden representation at that layer and train the remaining network against the same interpolation of the paired one-hot labels. Layers before and after the mixing boundary both receive gradients, and the method requires no extra model forward.

## Evidence

With `alpha=2`, CIFAR-10 error improves from 4.83 to 2.95 on PreActResNet-18, 4.64 to 2.54 on PreActResNet-34, and 3.99 to 2.55 on WRN-28-10. It materially outperforms input Mixup and improves test NLL. The strongest reported eligible set contains input plus the first two hidden boundaries.

## Reusable Caveats

- A policy using CutMix at input instead of linear Mixup is a hybrid adaptation, not a reproduction of the paper's strongest layer set.
- Mixing policy, boundary, lambda, permutation, and label coefficient must be coupled exactly; no area correction applies to linear hidden mixing.
- When composing with a validated input-space augmentation, hold total mixed exposure fixed and preregister the replacement share to keep attribution honest.
- The large source gains use longer and weaker-baseline recipes, so discount effect size heavily on a strong CutMix/SAM parent.
