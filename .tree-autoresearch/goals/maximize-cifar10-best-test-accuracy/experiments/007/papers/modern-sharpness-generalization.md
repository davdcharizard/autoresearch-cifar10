# A Modern Look at the Relationship between Sharpness and Generalization
- **Authors**: Maksym Andriushchenko, Francesco Croce, Maximilian Mueller, Matthias Hein, Nicolas Flammarion
- **Venue**: ICML 2023
- **URL**: https://proceedings.mlr.press/v202/andriushchenko23a.html

## Key Contributions
- Evaluates multiple sharpness definitions across CIFAR-10, ImageNet, transformers, and language tasks.
- Finds that measured sharpness often tracks training choices such as learning rate rather than generalization itself.
- Shows that reparameterization-invariant adaptive sharpness is not a universal predictor of better generalization.

## Relevance
EXP-004 validates one sparse late SAM operating point, but this paper cautions against assuming that replacing it with ASAM must improve accuracy merely because the geometry is scale-invariant. Any ASAM proposal needs direct matched-regime evidence and a concrete effect expectation rather than a flatness-only argument.

## Key Techniques
- Compare sharpness definitions under controlled changes in learning rate and parameterization.
- Separate correlation with optimization settings from correlation with in-distribution and out-of-distribution generalization.
- Treat sharpness metrics as data- and setup-dependent diagnostics.
