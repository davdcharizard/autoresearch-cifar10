# BlurPool / anti-aliased downsampling (Zhang 2019, "Making Convolutional Networks Shift-Invariant Again")

Standing reference for the FIRST architectural-inductive-bias lever on this goal (chosen EXP-018), distinct from the saturated capacity/optimizer/regularization/aug/BN-noise axes.

## What it is
Naive downsampling — `MaxPool2d`, strided conv, `AvgPool2d` — is "dense op then subsample"; the subsample step violates the Nyquist sampling theorem → **aliasing** → the network loses **shift-equivariance** (a small input translation can drastically change downsampled activations). The classic signal-processing fix: **low-pass filter (blur) between the dense op and the stride-2 subsample**. Zhang 2019 (ICML, arXiv:1904.11486) shows that, integrated correctly, this:
- keeps the original op (e.g. max), just anti-aliases its subsampling → **MaxBlurPool** = `MaxPool(stride=1)` then blur(stride=2);
- **increases clean classification accuracy** across ImageNet architectures (not just robustness) — the paper frames anti-aliasing as **effective regularization**;
- uses a **fixed (non-learned) binomial blur kernel**, sizes 2–5: size-2 `[1,1]` (rect), size-3 `[1,2,1]` (triangle, default), size-5 `[1,4,6,4,1]`; the 2D kernel is the outer product, normalized to sum 1 (DC-preserving).
Reference impl: Adobe `antialiased-cnns` (github.com/adobe/antialiased-cnns).

## Why it's a fresh axis on this goal
Our whitened ResNet-9 downsamples with naive `MaxPool2d(2)`×3 (layer1/2/3) + `MaxPool2d(4)` head — never touched in EXP-001–017. It is the one structural weakness on a net diagnosed as generalization-ceiling-bound (EXP-014). BlurPool changes the *inductive bias* (shift-equivariance / what features the net represents stably), categorically different from input-aug (EXP-008/011/015), wd/LS (012), optimizer (009/010), loss-geometry (013), capacity/epochs (005/007/014), and BN/activation noise (016/017 — closed).

## Implementation on THIS harness (load-bearing points)
- **Drop-in for `nn.MaxPool2d(2)`** at layer1/2/3 (channels 128/256/512). `MaxBlurPool2d` = `nn.MaxPool2d(2, stride=1)` → depthwise `F.conv2d` with the fixed kernel, `groups=C`, `stride=2`. Output size = floor(H/2), matching the old MaxPool(2) exactly → downstream shapes unchanged.
- **Blur kernel = registered buffer**, NOT a `nn.Parameter` → excluded from the optimizer automatically; rides `AveragedModel(use_buffers=True)` harmlessly (constant). Adds 0 trainable params.
- **Precompute the buffer in the activation dtype** (bf16) once — do NOT call `self.kernel.to(x.dtype)` per forward (allocation/launch overhead; reviewer concern).
- **`reflect` pad** by `ksize//2` on all sides before the blur to avoid zero-pad border bias and keep size arithmetic clean.
- **channels_last/bf16**: depthwise conv preserves memory format; keep the buffer channels_last.
- **THROUGHPUT IS THE #1 RISK** (project-insights High; EXP-005/007/013/016 under-anneal). BlurPool uses STANDARD convs (no fused-BN-kernel break, unlike GhostBN EXP-016), but 3 depthwise blurs at full spatial (32/16/8) + `F.pad` can be kernel-launch/memory-bound. MUST run an M1 throughput smoke and gate `num_epochs ≥ ~135`; fallback ladder: layer1/2-only (large-spatial sites where aliasing matters most) → ksize=2 (cheapest). Per EXP-014, ±10ep near 150 ≈ 0.02–0.03pp (below bar), so a small cost does not confound.
- Do NOT blur the final 4×4 head pool in the primary cell (over-smoothing/shape risk; reviewer concern) — defer to a follow-up.

## Strength / operating points
Primary: MaxBlurPool layer1/2/3, ksize=3 (triangle). Second point: ksize=2 (rect, lightest). Watch ep25 (within ~0.5pp of c0) + full anneal (best≈final).

## Status on this goal
Tested **EXP-018 → no-improvement** (the first architectural-inductive-bias experiment). The faithful MaxBlurPool (dense max + blur) was THROUGHPUT-DISQUALIFIED: the dense `MaxPool2d(stride=1)` at full res + `F.pad(reflect)` ran ~0.40× baseline (reflect) / 0.884× (zero-pad) → 132ep < the 135 under-anneal gate. The THROUGHPUT-FREE **BlurPool-only** form (drop the dense max; binomial blur stride-2 replaces max — zero-pad in conv) ran at 153/146ep (≥ c0's 150) and was tested: cA(ks3) 96.23 = **−0.08pp** vs same-session c0 96.31 (within noise); cB(ks5) 96.16 = **−0.15pp** — MONOTONICALLY WORSE with stronger blur, fully annealed, ep25 healthy (cA 92.41 ≥ c0 92.18, not under-fit). The honest prior held: strong `RandomCrop(pad4)`+flip aug already supplies translation invariance and 32×32 has little aliasing (vs Zhang's 224² ImageNet gains), so the shift-equivariance prior is redundant and the low-pass mildly hurts (lost detail).
**Implementation note for any re-entry**: use zero-pad IN the conv (`F.conv2d(..., padding=ksize//2)`), NOT `F.pad(reflect)` (the reflect kernel+alloc was the dominant cost: 0.40× vs 0.70×). Odd ksize only (clean symmetric padding). The dense `MaxPool2d(stride=1)` is the remaining cost that disqualifies max-mode.
**Do NOT re-run BlurPool variants** — the downsampling-operator axis is closed. The only untested faithful variant (compile-funded MaxBlurPool to preserve max at full epochs) is LOW priority: the monotonic blur-strength degradation argues a masked anti-aliasing benefit is implausible (at best a tie). See experiments/018/04-analysis.md.

## Sources
- Zhang 2019 https://arxiv.org/abs/1904.11486 (ICML) ; Adobe antialiased-cnns https://github.com/adobe/antialiased-cnns
