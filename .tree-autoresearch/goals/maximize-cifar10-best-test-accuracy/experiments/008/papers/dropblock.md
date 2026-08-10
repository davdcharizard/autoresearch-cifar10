# DropBlock: A Regularization Method for Convolutional Networks
- **Authors**: Golnaz Ghiasi, Tsung-Yi Lin, Quoc V. Le
- **Venue**: NeurIPS 2018
- **URL**: https://papers.nips.cc/paper_files/paper/2018/hash/7edcfb2d8f6a659ef4cd1e6c9b6d7079-Abstract.html

## Key Contributions

- Drops contiguous spatial regions rather than independent activations, addressing spatial correlation that weakens ordinary dropout in convolutional maps.
- Reports that applying DropBlock to skip paths as well as convolutional paths improves accuracy.
- Gradually increasing the dropped fraction improves accuracy and robustness to the strength setting; the paper reports a 1.6-point ImageNet top-1 gain for ResNet-50.

## Relevance

DropBlock directly regularizes feature maps and can be implemented without an extra model pass, so it fits the time budget. Its mechanism overlaps EXP-004's stochastic residual drop path and introduces block-size and keep-rate choices; a fixed single run would need a literature-derived, time-scheduled package with dedicated RNG isolation.

## Key Techniques

- Bernoulli seed mask expanded to contiguous blocks.
- Rescale surviving activations to preserve expectation.
- Increase regularization strength gradually during training.
