# Regularization in ResNet with Stochastic Depth
- **Authors**: Soufiane Hayou and colleagues
- **Venue**: NeurIPS 2021
- **URL**: https://papers.nips.cc/paper_files/paper/2021/file/82ba9d6eee3f026be339bb287651c3d8-Paper.pdf

## Key Contributions
- Analyzes stochastic depth as explicit and gradient regularization in residual networks.
- Shows full residual blocks can be skipped during training, reducing effective depth and potentially training time, unlike ordinary dropout.
- Derives survival-probability guidance and documents a compute/regularization tradeoff on CIFAR-10 ResNets.

## Relevance
Batchwise stochastic depth could attack the dominant convolution/BN backward cost while adding architectural noise. Yet this model is only ResNet-20, the accepted strong phase already risks underfit, and identity-oriented local changes have repeatedly reduced switch accuracy. Any candidate must prove both actual conditional-kernel savings and sufficient updates per residual branch.

## Key Techniques
- Sample residual-block survival masks per training iteration and use the full/scaled network at evaluation.
- Prefer a shallow-model, low-drop fixed policy with exact RNG and BatchNorm semantics.
- Measure both global optimizer exposure and per-block effective update counts rather than claiming skipped compute as free progress.
