# BlurPool — Anti-aliased downsampling (Zhang 2019)

**Paper**: "Making Convolutional Networks Shift-Invariant Again", ICML 2019, arXiv:1904.11486
**Topic**: Architecture / generalization — anti-aliased downsampling
**Why relevant**: a convergence-NEUTRAL generalization mechanism (no learnable params, no stochastic penalty) — fits
the project's diagnosed need after every regularizer failed and all scalar knobs bracketed. First applied EXP-024.

## Core idea
Strided downsampling (strided conv, max-pool) samples every s-th pixel with no low-pass first → aliasing → CNNs are
surprisingly shift-VARIANT, hurting generalization. Fix: insert a low-pass (blur) filter BEFORE every subsampling.
Replace a "stride-s op" with "stride-1 op → Blur → stride-s subsample". The blur kernel is a FIXED binomial filter
(e.g. 3×3 from [1,2,1]⊗[1,2,1]/16), applied depthwise (groups=channels) — zero learnable parameters.
Reported small but consistent accuracy gains + large shift-consistency gains across architectures/datasets.

## Implementation (PyTorch, no new dep)
```python
class BlurPool2d(nn.Module):
    def __init__(self, channels, stride=2):
        super().__init__()
        self.stride, self.channels = stride, channels
        a = torch.tensor([1., 2., 1.])
        k = (a[:, None] * a[None, :]); k = k / k.sum()       # 3x3 binomial, sums to 1
        self.register_buffer("filt", k[None, None].repeat(channels, 1, 1, 1))  # (C,1,3,3) buffer, not a param
    def forward(self, x):
        return F.conv2d(x, self.filt, stride=self.stride, padding=1, groups=self.channels)
```
- Downsample BasicBlock (stride=2): make `conv1` stride-1, then `relu(bn1(conv1(x))) → BlurPool(stride2) → conv2`.
- Projection shortcut: `BlurPool(in_ch, stride2)(x) → 1×1 conv stride1 → BN`.
- Spatial check (3×3, stride2, padding1): out=floor((H-1)/2)+1 → 32→16, 16→8, 8→4 — matches the strided-conv output,
  so conv-path and shortcut stay aligned. Params unchanged (filters are buffers).

## The cost caveat for THIS project (critical)
Moving the downsampling conv to stride-1 makes it compute at the HIGHER (pre-subsample) resolution → ~4× FLOPs for
that conv. At the two downsample sites (layer2 64→128, layer3 128→256, the heaviest convs) this is a meaningful FLOPs
increase (~+1G). At a FIXED 300s budget this risks the project's most robust failure mode: more compute → fewer
epochs → underfit → regress (cf. capacity EXP-004/009, and the compile-graph epoch loss EXP-015). MITIGATING fact:
k=4 is LAUNCH-bound (8ms/step, GPU mostly idle), so spare FLOPs MAY be partly absorbed without proportional wall-clock
cost — genuinely uncertain. Therefore ALWAYS check realized epoch count vs baseline 91: if epochs crater the result is
compute-confounded, not a clean test of anti-aliasing.

## CIFAR ceiling
Documented CIFAR-10 gains are modest (small 32×32 images, only 2 downsample stages) vs ImageNet — expect a small
effect at best even if epochs hold.
