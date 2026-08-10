# DropBlock: A regularization method for convolutional networks
- **Authors**: Golnaz Ghiasi, Tsung-Yi Lin, Quoc V. Le
- **Venue**: NeurIPS 2018
- **URL**: https://papers.nips.cc/paper_files/paper/2018/hash/7edcfb2d8f6a659ef4cd1e6c9b6d7079-Abstract.html

## Key Contributions
- Drops contiguous feature-map regions instead of independent activations.
- Finds structured dropout more effective for spatially correlated convolutional features.
- Recommends gradually increasing the dropped fraction during training for robustness.

## Relevance
The method offers feature-space occlusion distinct from failed input Cutout, but adds masking work to a backward-bound loop and may compound the accepted RandAugment+CutMix regularization. Any local adaptation should be late-stage, low-rate, and phase-gated rather than applied throughout training.

## Key Techniques
- Generate contiguous spatial masks in selected feature stages.
- Rescale surviving activations to preserve expectation.
- Schedule drop probability rather than applying its maximum from initialization.
