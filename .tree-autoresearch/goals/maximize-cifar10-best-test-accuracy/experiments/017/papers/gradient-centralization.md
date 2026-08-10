# Gradient Centralization: A New Optimization Technique for Deep Neural Networks
- **Authors**: Hongwei Yong, Jianqiang Huang, Xiansheng Hua, Lei Zhang
- **Venue**: ECCV 2020
- **URL**: https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/2471_ECCV_2020_paper.php
- **Official optimizer**: https://github.com/Yonghongwei/Gradient-Centralization/blob/master/GC_code/CIFAR100/algorithm/SGD.py

## Key Contributions
- Centralizes eligible weight gradients to zero mean and interprets the operation as projected gradient descent under a constrained loss.
- Reports improved loss/gradient smoothness, training stability, and generalization across image classification, fine-grained recognition, detection, and segmentation.
- Requires only linear gradient processing and no model architecture, activation, or data-path change.

## Relevance
EXP-002 is limited by stable generalization under a strict forward budget. Gradient centralization can be inserted between backward and the existing Nesterov step without another forward, new data, or model state. The official `SGD_GC` adds coupled L2 before centralizing eligible directions and before momentum. The relevant isolated test should match that order for convolutional/linear weights, leave BN/bias directions uncentralized, charge the reduction work, and audit the removed mean component. Literature establishes plausibility, not the coefficient-free method's effect under this exact CutMix/drop-path/time-cosine recipe.

## Key Techniques
- For eligible regularized direction tensor `d = g + weight_decay * parameter`, apply `d -= mean(d, dim=all dimensions except output-channel dimension, keepdim=True)`.
- Compose coupled decay and GC in the order `data gradient + L2 -> GC -> momentum/Nesterov`.
- Verify post-transform centralization residuals and measure charged overhead rather than assuming the linear pass is free.
