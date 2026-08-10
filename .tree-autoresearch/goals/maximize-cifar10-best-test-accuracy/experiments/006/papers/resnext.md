# Aggregated Residual Transformations (ResNeXt)

- **Authors**: Saining Xie, Ross Girshick, Piotr Dollar, Zhuowen Tu, Kaiming He
- **Venue**: CVPR 2017
- **URL**: https://openaccess.thecvf.com/content_cvpr_2017/html/Xie_Aggregated_Residual_Transformations_CVPR_2017_paper.html

## Key Contributions

- Introduces cardinality, implemented with grouped 3x3 convolutions inside bottleneck residual blocks, as a representation dimension distinct from width and depth.
- At matched ImageNet complexity, increasing cardinality outperforms merely going wider or deeper.
- A 29-layer CIFAR architecture uses three stages of three bottleneck blocks after an initial 3x3 convolution.

## Relevance

The H20 has enormous memory headroom and the current six-basic-block WRN may leave representational gains available. On CIFAR-10, ResNeXt-29 8x64d reports 3.65% error with 34.4M parameters versus 4.17% for a 36.5M Wide ResNet; 16x64d reaches 3.58%. Results average ten runs. However, these models are more than twelve times the parent's 2.75M parameters and the evidence does not cover a 300-second budget, so a compact cardinality adaptation has uncertain effect and throughput.

## Key Techniques

- Use 1x1 reduce, grouped 3x3 transform, and 1x1 expand residual branches.
- Preserve preactivation, time-based schedules, front-loaded CutMix, and late SAM where possible.
- A compact configuration must be microbenchmarked before a full run; parameter count alone does not predict grouped-convolution throughput at 32x32.
- Any reduced-width/cardinality recipe departs from the reported configuration and should be treated as an exploratory architecture bet, not a directly reproduced result.
