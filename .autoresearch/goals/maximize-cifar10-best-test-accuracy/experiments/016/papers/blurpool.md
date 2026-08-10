# Making Convolutional Networks Shift-Invariant Again
- **Authors**: Richard Zhang
- **Venue**: ICML 2019
- **URL**: https://proceedings.mlr.press/v97/zhang19a.html

## Key Contributions
- Identifies aliasing in strided convolution, average pooling, and max pooling.
- Integrates fixed low-pass filtering before subsampling across common CNN downsampling paths.
- Reports improved shift consistency and ImageNet classification accuracy with modest compute.

## Relevance
The accepted CIFAR ResNet downsamples twice through both stride-2 residual convolutions and Option-A shortcut slicing. A valid anti-alias candidate must keep the two paths spatially aligned and filter before subsampling; changing only the shortcut is not the paper's full mechanism.

## Key Techniques
- Convert a strided operator to dense stride one, then fixed blur/subsample.
- Apply consistent downsampling semantics across residual paths.
- Measure kernel overhead rather than infer it from fixed coefficients.
