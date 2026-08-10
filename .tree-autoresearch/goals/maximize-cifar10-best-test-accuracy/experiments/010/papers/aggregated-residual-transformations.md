# Aggregated Residual Transformations for Deep Neural Networks
- **Authors**: Saining Xie, Ross Girshick, Piotr Dollar, Zhuowen Tu, Kaiming He
- **Venue**: CVPR 2017
- **URL**: https://openaccess.thecvf.com/content_cvpr_2017/html/Xie_Aggregated_Residual_Transformations_CVPR_2017_paper.html

## Key Contributions
- Introduces cardinality, the number of parallel transformations, as a capacity axis alongside depth and width.
- Shows increased cardinality can outperform deeper or wider alternatives at matched complexity.
- Gives an equivalent grouped-convolution implementation for the multi-branch residual transformation.

## Relevance
Grouped convolutions could increase representational diversity without increasing nominal FLOPs. On this small WRN, however, changing both block topology and kernel efficiency is risky; a grouped-convolution candidate requires a strict same-harness latency preflight and should retain dense input/output projections.

## Key Techniques
- Bottleneck residual block with grouped 3x3 convolution.
- Homogeneous repeated topology.
- Complexity-matched depth, width, and cardinality comparisons.

_Full PDF fetch returned HTTP 403; distilled from the official CVPR abstract and indexed PDF snippet._
