# Gradient Centralization: A New Optimization Technique for Deep Neural Networks
- **Authors**: Hongwei Yong, Jianqiang Huang, Xiansheng Hua, Lei Zhang
- **Venue**: ECCV 2020
- **URL**: https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/2471_ECCV_2020_paper.php
- **Official optimizer**: https://github.com/Yonghongwei/Gradient-Centralization/blob/master/GC_code/CIFAR100/algorithm/SGD.py

## Key Contributions
- Centralizes eligible weight gradients to zero mean and interprets the operation as projected gradient descent under a constrained loss.
- Reports improved optimization smoothness and generalization across multiple vision tasks.
- Adds a linear gradient transformation without another model forward or architecture change.

## Quantitative CIFAR Evidence
- On CIFAR-100 with SGDM, batch 128, 200 epochs, and weight decay `5e-4`, ten-run mean accuracy improved by `+1.95` points for ResNet-18 (76.87 -> 78.82), `+1.39` for ResNet-101, `+0.83` for ResNeXt-29, `+0.75` for VGG-11, and `+0.37` for DenseNet-121 (main paper Table 1).
- ResNet-50 SGDM improved `+0.91` points (78.23 -> 79.14) under the same CIFAR-100 regime (main paper Table 2). The supplementary weight-decay study reports gains at every tested value, including `+2.65` points at `1e-4`, but absolute baselines vary strongly with weight decay.
- The paper reports 0.6 seconds of GC overhead within a 71-second ResNet-50 CIFAR-100 epoch (~0.85%). This supports a low-overhead prior, not a guarantee for the compact H20 WRN where 17 reductions may be launch-bound.
- These are CIFAR-100 conventional-epoch results, not CIFAR-10 under a saturated CutMix/drop-path wall-clock recipe. They establish plausible effect size above this goal's 0.10-point gate while leaving transfer magnitude uncertain.

Primary source: https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460613.pdf (Table 1/2 and Section 3.4); supplementary: https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460613-supp.pdf.

## Relevance
Gradient centralization is a low-forward-cost optimization lever for the CIFAR WRN. The official `SGD_GC` adds coupled L2 weight decay before centralizing eligible convolutional/linear directions, then applies momentum/Nesterov. Preserve BN/bias directions without centralization, charge the reductions, and measure both removed mean norm and throughput.

## Key Techniques
- Subtract each output unit's mean gradient over its remaining tensor dimensions.
- For coupled-decay SGD, compose in the order `data gradient + L2 -> GC -> momentum/Nesterov`.
