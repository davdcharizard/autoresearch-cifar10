# A Modern Look at the Relationship between Sharpness and Generalization
- **Authors**: Maksym Andriushchenko et al.
- **Venue**: ICML 2023
- **URL**: https://proceedings.mlr.press/v202/andriushchenko23a.html

## Key Contributions
- Compares multiple sharpness definitions across CIFAR-10, ImageNet, vision transformers, and language models.
- Finds sharpness often tracks training choices such as learning rate and is not a universal generalization predictor.

## Relevance
The result cautions against selecting ASAM only because adaptive sharpness is reparameterization-aware. Experiments need matched optimizer evidence, fixed geometry, and direct accuracy verification; lower measured sharpness alone is not success.

## Key Techniques
- Control learning rate and parameterization when comparing sharpness.
- Separate in-distribution accuracy from sharpness and OOD correlations.
