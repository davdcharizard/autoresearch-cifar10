# Squeeze-and-Excitation Networks
- **Authors**: Jie Hu, Li Shen, Gang Sun
- **Venue**: CVPR 2018
- **URL**: https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper

## Key Contributions

- Uses global pooling and a bottleneck MLP to model full channel interdependencies.
- Establishes channel recalibration as an effective residual-backbone mechanism.

## Relevance

SE has a higher mechanism ceiling than local ECA, but standard sigmoid gates are not identity-preserving and small FP32 excitation paths can be launch-bound on compact CIFAR models. Use isolated initialization and measure production latency rather than estimating from parameter count.

## Key Techniques

- Global average pooling, reduction MLP, and sigmoid excitation.
- Residual-block integration with channel-wise scaling.
