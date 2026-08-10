# ASAM: Adaptive Sharpness-Aware Minimization

- **Authors**: Jungmin Kwon, Jeongseop Kim, Hyunseo Park, In Kwon Choi
- **Venue**: ICML 2021
- **URL**: https://proceedings.mlr.press/v139/kwon21b.html

## Key Contributions

- Replaces SAM's scale-sensitive spherical parameter neighborhood with a weight-adaptive neighborhood.
- For element-wise p=2 ASAM, compute `scaled_grad = (abs(w)+eta) * grad`, normalize it globally, then perturb by `rho * (abs(w)+eta) * scaled_grad / ||scaled_grad||`, equivalently `rho * (abs(w)+eta)^2 * grad / ||(abs(w)+eta)*grad||`.
- Uses `eta=0.01`; does not adapt bias parameters in its preferred configuration.
- On CIFAR-10, reports ASAM above SAM across ResNet, ResNeXt, WRN, DenseNet, and PyramidNet families.

## Relevance

EXP-004 already proves that period-two SAM in the final clean quarter adds 0.17 points. ASAM changes the geometry of that validated intervention without another pass or more pulses. Reported CIFAR-10 gains over SAM are 0.24 points on ResNet-56, 0.46 on ResNeXt29-32x4d, 0.20 on WRN-28-2, and 0.30 on WRN-28-10. These magnitudes exceed the 0.10-point gate, although the paper uses full-run ASAM rather than a late periodic dose.

## Key Techniques

- Candidate fixed settings: `ASAM_RHO=0.5`, `ASAM_ETA=0.01`, element-wise p=2, no bias adaptation.
- Preserve EXP-004's final-quarter, period-two schedule and all two-pass RNG/BatchNorm/restoration invariants.
- The adaptive perturbation norm is not 0.5 in ordinary Euclidean space; verify the normalized adaptive-coordinate relation and exact restore instead.
- The paper tuned rho over a large grid. Using its common CIFAR value is literature transfer, not evidence that 0.5 is optimal for a late-only schedule.
