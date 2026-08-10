# Sharpness-Aware Minimization for Efficiently Improving Generalization
- **Authors**: Pierre Foret, Ariel Kleiner, Hossein Mobahi, Behnam Neyshabur
- **Venue**: ICLR 2021
- **URL**: https://openreview.net/forum?id=6Tm1mposlrM

## Key Contributions
- Optimizes both loss value and neighborhood sharpness through an adversarial parameter perturbation.
- Reports strong generalization improvements across CIFAR models and other vision tasks.

## Relevance
SAM has meaningful upside for a well-fitted CIFAR classifier, but its second forward/backward pass must be sparsified or phase-limited under a strict wall-clock budget. RNG replay and BatchNorm buffer handling are essential in stochastic residual networks.

## Key Techniques
- Global gradient-normalized weight perturbation.
- Second gradient at perturbed parameters.
- One optimizer update after exact parameter restoration.
