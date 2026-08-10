# Adaptive Sharpness-Aware Minimization (ASAM)

- Source: Jungmin Kwon et al., ICML 2021, https://proceedings.mlr.press/v139/kwon21b.html

## Core Method

ASAM makes SAM's perturbation neighborhood scale-aware. For element-wise p=2 adaptation with scale `s=abs(w)+eta`, the perturbation is `epsilon = rho * s^2 * grad / ||s*grad||`. Biases are preferably left unadapted with unit scale. The adaptive-coordinate radius `||epsilon/s||` rather than the Euclidean norm is fixed at rho.

## Evidence

With CIFAR defaults `rho=0.5`, `eta=0.01`, ASAM improves over SAM by 0.20 points on WRN-28-2, 0.30 on WRN-28-10, 0.24 on ResNet-56, and 0.46 on ResNeXt29-32x4d. The reported method runs throughout training.

## Reusable Caveats

- Full-run ASAM increments should be discounted sharply when only a sparse or late subset of steps uses adaptive perturbations.
- The second scale multiplication is essential; `rho*s*grad/||s*grad||` is not the p=2 algorithm.
- Rho is geometry- and schedule-dependent. A literature value transferred to a low-LR tail needs a non-metric perturbation safety bound and exact adaptive-radius smoke.
- Weight decay belongs only in the restored base-optimizer update, not the first gradient that defines the adversarial perturbation.
