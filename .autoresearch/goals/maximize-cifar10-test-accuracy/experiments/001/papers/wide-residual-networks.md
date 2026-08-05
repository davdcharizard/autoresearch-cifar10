# Wide Residual Networks
- **Authors**: Sergey Zagoruyko, Nikos Komodakis
- **Venue**: BMVC 2016
- **URL**: https://arxiv.org/abs/1605.07146

## Key Contributions
- Replaces extreme residual-network depth with fewer, wider residual blocks.
- Reports better accuracy and computational efficiency than thin, very deep residual networks on CIFAR.

## Relevance
The baseline is a thin ResNet-20 and the objective is accuracy under a wall-clock budget. A shallower, wider network may spend H20 compute more effectively while retaining enough optimizer steps in 300 seconds.

## Key Techniques
- Widen residual stages while reducing or retaining modest depth.
- Use CIFAR-style residual blocks and standard crop/flip augmentation.
