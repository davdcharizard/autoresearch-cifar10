# Sharpness-Aware Minimization for Efficiently Improving Generalization
- **Authors**: Pierre Foret, Ariel Kleiner, Hossein Mobahi, Behnam Neyshabur
- **Venue**: ICLR 2021
- **URL**: https://openreview.net/forum?id=6Tm1mposlrM

## Key Contributions
- Optimizes loss jointly with neighborhood sharpness by taking an adversarial parameter perturbation before the update.
- Reports improved generalization across CIFAR-10/100 and other vision tasks.
- Provides a model- and dataset-agnostic optimizer wrapper.

## Relevance
The mechanism directly targets generalization rather than augmentation strength, but standard SAM requires two sequential gradient calculations. Under this fixed 300-second training budget, full SAM would roughly halve optimizer exposure, so only a sparse or late periodic variant is operationally plausible.

## Key Techniques
- Compute a normalized gradient perturbation of radius rho.
- Recompute the gradient at perturbed parameters before the optimizer update.
- Trade extra per-step compute for flatter-neighborhood optimization.
