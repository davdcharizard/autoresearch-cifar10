# CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features (Yun et al.)

- **arXiv**: 1905.04899 (ICCV 2019)
- **Status**: TESTED AND REFUTED for this project's regime (EXP-060, no-improvement — 96.69 family null at matched dose p=0.5, byte-clean signatures; absorption extends to augmentation TYPE; see reports/exp-report-060.md)

## Paper claim

Cut-and-paste occlusion: with probability p, sample λ ~ Beta(α, α), cut a box of area
(1−λ)·HW at a random center, replace the region with the same-position patch from another
training image (batch permutation), and supervise with the area-weighted mixed loss
`λ·CE(y_a) + (1−λ)·CE(y_b)` (λ adjusted to the actual clamped box area). Unlike Cutout/erasing
(region → noise/zeros, information DELETED) the region keeps real image statistics, and unlike
mixup (global alpha-blend, ghosted off-manifold inputs) the inputs stay locally in-domain.
CIFAR-10/100 fixed-epoch results (crop+flip baselines): beats Cutout and mixup consistently,
e.g. PyramidNet CIFAR-100 +1.5 over baseline where Cutout gains +0.5; CIFAR-10 gains
+0.5–1.0 over erasing-class augmentation on ResNet-family nets. Canonical CIFAR setting α=1.0
(λ uniform), applied with p=0.5–1.0.

## Project-relevant arithmetic

- α=1.0 → Beta(1,1) = Uniform(0,1): λ is a plain uniform draw, no Beta sampler needed.
- Implementation must stay out of the compiled graph and sync-free in the charged region:
  box coordinates from CPU RNG as Python ints (no `.item()` on GPU tensors), then only
  `torch.randperm(B, device)` + one in-place slice assignment + a second CE on the SAME logits.
  Expected charged cost ~0.1–0.3ms/step on a 22.4ms step (deferral-law price ≤ −0.02pp).
- Composes with label smoothing (DeiT-class recipes use CutMix+LS together).
- Project priors: mixup STACKED as a 4th regularizer lost −0.46 (EXP-009); absorption law
  (external transfer 0-for-16) discounts the published gain heavily — this is the TYPE
  substitution probe (occlusion-with-signal vs occlusion-with-noise at constant dose p=0.5).
