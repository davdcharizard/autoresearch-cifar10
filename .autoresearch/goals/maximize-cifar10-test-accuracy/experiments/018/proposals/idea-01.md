# Proposal idea-01: BlurPool anti-aliased downsampling (MaxBlurPool) — shift-equivariance inductive bias

## Core change (train.py only)
Replace each `nn.MaxPool2d(2)` in `layer1/2/3` (and optionally the final `nn.MaxPool2d(4)`) with **MaxBlurPool**: keep the max operation at **stride 1** (anti-alias-friendly, preserves the max-feature selection), then low-pass filter with a **fixed binomial blur kernel** applied **depthwise** at **stride 2** to subsample. The blur kernel is a fixed (non-learned) normalized binomial filter (Zhang's default `[1,2,1]⊗[1,2,1]/16`, 3×3), registered as a buffer (no params, no optimizer entry, no new deps — pure `F.conv2d` with `groups=C`). A `BLUR_KSIZE` env (3 default; 2/5 optional) and a `BLUR_FINAL` toggle for the 4×4 head pool let the plan sweep the operating point.

```python
class BlurPool2d(nn.Module):
    def __init__(self, channels, ksize=3, stride=2):
        super().__init__()
        a = {2: [1.,1.], 3: [1.,2.,1.], 5: [1.,4.,6.,4.,1.]}[ksize]
        k = torch.tensor(a); k = (k[:,None]*k[None,:]); k = k/k.sum()
        self.register_buffer("kernel", k[None,None].repeat(channels,1,1,1))
        self.stride, self.ksize, self.channels = stride, ksize, channels
    def forward(self, x):
        x = F.pad(x, [self.ksize//2]*4, mode="reflect")
        return F.conv2d(x, self.kernel.to(x.dtype), stride=self.stride, groups=self.channels)

class MaxBlurPool2d(nn.Module):           # drop-in for nn.MaxPool2d(2)
    def __init__(self, channels, ksize=3):
        super().__init__()
        self.mp = nn.MaxPool2d(2, stride=1)        # dense max, stride 1
        self.blur = BlurPool2d(channels, ksize, stride=2)
    def forward(self, x):
        return self.blur(self.mp(x))
```
Wired into `ResNet9.__init__`: `nn.MaxPool2d(2)` → `MaxBlurPool2d(128/256/512, BLUR_KSIZE)` at layer1/2/3 (channels = the `conv_bn` output feeding that pool: 128, 256, 512). The blur buffer rides with `AveragedModel(use_buffers=True)` harmlessly (constant). channels_last/bf16 preserved (depthwise conv keeps memory format; cast kernel to x.dtype).

## Mechanism — why this is a genuinely DIFFERENT lever
The current net subsamples with naive `MaxPool2d(2)` ×3 + `MaxPool2d(4)` — each violates the Nyquist sampling theorem, so the feature map is **aliased** and the network is **not shift-equivariant**: a 1-px input translation can flip downsampled activations. This is the ONE structural weakness never touched in EXP-001–017. Zhang 2019 (ICML, "Making Convolutional Networks Shift-Invariant Again") shows inserting a low-pass blur **between** the dense pool and the subsample restores approximate shift-equivariance AND **increases clean classification accuracy** across ImageNet architectures — the paper explicitly frames anti-aliasing as **effective regularization** that improves generalization, not just robustness. This is a change to the network's **inductive bias** (what features it can represent stably), categorically distinct from the saturated axes: input-aug (EXP-008/011/015), wd/LS (EXP-012), optimizer (EXP-009/010), loss-geometry (EXP-013), capacity/epochs (EXP-005/007/014), and BN/activation noise (EXP-016/017).

## Why it targets the limiter
The diagnosed limiter is a **generalization ceiling** (~96.3–96.5) for this architecture at 300s — EXP-014 bought +12% epochs and +capacity with flat accuracy, proving the bottleneck is the architecture's representational/generalization quality, not compute. BlurPool lifts that ceiling by improving the representation's stability prior rather than adding capacity or regularization noise. It is the canonical, literature-validated way to make a strong convnet generalize better **without** adding capacity (no epoch cost from params) — exactly the class the strategic mandate calls for (project-insights High, EXP-014/017: "the high-EV move is a different architectural inductive bias").

## Throughput (the #1 failure mode — pre-registered check)
Cost = 3 depthwise 3×3 blur convs at spatial 32→16, 16→8, 8→4 (stride-1 maxpool is cheap; depthwise blur is ~C·9 MACs/pixel, far below the dense 3×3 convs' C_in·C_out·9). These are **standard cuDNN convs** — unlike GhostBN (EXP-016) they do NOT break the fused BN kernel. Expectation: ≤~3–5% throughput cost (≥~142 epochs). MUST be measured in an M1 throughput smoke; if a cell drops <~135 epochs, restrict to layer1/2 only (the large-spatial sites where aliasing matters most) or drop `BLUR_FINAL`. Per EXP-014, ±10 epochs near 150 ≈ 0.02–0.03pp, well under the bar, so a small cost does not confound.

## Design — SAME-SESSION multi-cell
- c0: unchanged baseline (`nn.MaxPool2d`) — full-speed same-session anchor (stored 96.38 too weak at the noise floor; same-session control mandatory per project-insights).
- cA: MaxBlurPool at layer1/2/3, ksize=3, final 4×4 pool unchanged — PRIMARY (anti-alias the three aliasing-prone strided subsamples).
- cB: cA + blur the final 4×4 head pool too (AvgBlur or MaxBlur on the 4×4), OR ksize=2 (lighter "rect-2" filter, Zhang's cheapest) — second operating point chosen in the plan from the throughput smoke.

## Correctness / EMA / eval
- Blur kernel is a fixed buffer (sums to 1, DC-preserving) — no params, excluded from optimizer automatically (not a `nn.Parameter`). `num_params` rises only by the buffer count (not counted in `requires_grad` params).
- Eval path identical to train (no train/eval divergence; deterministic op). flip-TTA still valid (blur is symmetric, commutes with horizontal flip up to the reflect pad).
- `reflect` pad avoids zero-padding bias at borders; keeps output size = floor(H/2) matching the old MaxPool(2) output exactly → downstream shapes unchanged.
- bf16/channels_last preserved; cast kernel to activation dtype each call.
- Smoke: (i) output shapes equal the old MaxPool path at every layer; (ii) blur kernel sums to 1.0; (iii) one-step backward produces finite grads on trainable params (blur buffer has no grad); (iv) ksize is odd-symmetric.

## Verification
- Best blurpool cell ≥ **96.48** AND > same-session c0 by a clear (>0.1pp) margin, replicated with a confirmation re-run on any apparent win (per the low-c0-draw lesson EXP-016/017).
- num_epochs ≥ ~135 (throughput check; reject/restrict if a cell under-anneals); ep25 within ~0.5pp of c0; fully annealed (best≈final).
- Integrity: train.py-only; prepare.py byte-unchanged; ≤1 eval/epoch; seed 42; baseline (ksize→identity is not a no-op, so use the c0 cell as the regression anchor).
- ON A WIN: bake MaxBlurPool as the default downsampling.

## Hypothesis
Replacing the aliasing `MaxPool2d` subsampling with anti-aliased MaxBlurPool restores approximate shift-equivariance and, per Zhang's "anti-aliasing as regularization" finding, lifts the generalization ceiling: best_test_acc ≥96.48 over the same-session control at near-full epochs. If it ties at healthy epochs/ep25, the ceiling is not movable by the downsampling inductive bias either, and the residual headroom (if any) lies in the stem/representation or is a genuine data/architecture limit at 300s.

## Effort: low-medium. Risk: (1) throughput — depthwise blur at full spatial could cost more than expected (mitigated: standard conv, layer-restriction ladder, num_epochs gate); (2) on CIFAR with already-strong aug + flip, the shift-equivariance gain may be small/within noise (the honest prior — Zhang's gains are ImageNet, larger images, more aliasing); (3) blurring the 4×4 head pool may over-smooth (mitigated: cB toggle, keep head unblurred in cA).
## Sources: Zhang 2019 "Making Convolutional Networks Shift-Invariant Again" ICML (arXiv:1904.11486); Adobe `antialiased-cnns` reference impl (github.com/adobe/antialiased-cnns); project-insights High (generalization ceiling EXP-014, backbone-pivot mandate); experiments/014/04-analysis.md; train.py:149-152 (the MaxPool downsample sites).
