# ECA-Net: Efficient Channel Attention

- Source: Qilong Wang et al., CVPR 2020, https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_ECA-Net_Efficient_Channel_Attention_for_Deep_Convolutional_Neural_Networks_CVPR_2020_paper.html
- Relevance: a low-overhead architectural change that is orthogonal to the parent's augmentation and optimizer regularization.

## Mechanism

After global average pooling, ECA applies a short 1D convolution across the channel descriptor and a sigmoid gate. It avoids the channel bottleneck used by squeeze-and-excitation, preserving direct channel information while allowing local cross-channel interaction. The kernel size can be chosen from channel width; for the parent's small widths, a fixed odd kernel such as 3 is the conservative implementation.

## Evidence

The paper reports only 80 extra parameters and 4.7e-4 GFLOPs when applied to ResNet-50, with a greater than two-point ImageNet top-1 improvement over the plain backbone. The study covers residual backbones and shows that removing dimensionality reduction is important. It does not provide a matched result for this exact CIFAR-10, PreAct WRN-16-4, 300-second regime.

## Experiment implications

- Add the gate after the residual branch's final convolution and before residual addition.
- Applying it to every block is simple and cheap, but changes the function of all eight blocks at once.
- Main risks are sigmoid gates initially scaling residual branches near 0.5 and an architecture change interacting with the already tuned optimization schedule.
- A neutral initialization or `2 * sigmoid(...)` gate could preserve the initial residual scale, but that departs from the paper and should be reviewed explicitly.

## Verdict

Keep as a finalist: plausible effect size, negligible time cost, and orthogonal to prior experiments, but exact CIFAR-10 benefit is uncertain.
