# Squeeze-and-Excitation Networks

**Source**: Hu et al., CVPR 2018, https://openaccess.thecvf.com/content_cvpr_2018/papers/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.pdf

## Key Takeaway
Squeeze-and-Excitation (SE) blocks are lightweight channel-attention modules that use global feature statistics to recalibrate channel responses. They can be inserted into existing CNN/residual blocks and are reported to improve classification accuracy with modest additional compute.

## Relevance to This Project
The CIFAR-10 loop has exhausted many scalar recipe changes around the current ResNet-20 anchor, including LR, weight decay, smoothing, batch size, cosine tails, mixup, and simple residual regularizers. SE blocks test a distinct architecture mechanism while preserving the proven optimizer, data augmentation, schedule, validation cadence, and fixed evaluation harness.

## Implementation Notes
- Insert the SE gate after the residual branch's final normalization and before adding the shortcut.
- Use global average pooling followed by a small bottleneck MLP or `1x1` conv stack with ReLU and sigmoid.
- Keep the reduction ratio conservative, such as 16, to limit parameter and runtime overhead.
- Watch the fixed-budget step count; any accuracy gain must justify added compute.
