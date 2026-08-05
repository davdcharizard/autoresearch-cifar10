# Mixing augmentation for CIFAR-10 (CutMix / mixup) at short schedules

Standing reference for label-mixing data augmentation, with the short-schedule evidence that decides which variant fits this goal's ~150-epoch / 300s budget.

## The variants
- **mixup** (Zhang et al., arXiv:1710.09412): convex pixel blend `x̃=λx_i+(1-λ)x_j`, `ỹ=λy_i+(1-λ)y_j`, λ~Beta(α,α). Global blend.
- **CutMix** (Yun et al., ICCV 2019, **arXiv:1905.04899**): paste a rectangular region from image j into image i; label weight = **area fraction** of the paste. λ~Beta(α,α), α=1.0 default; box side = √(1-λ)·H,W via `rand_bbox` with edge-clamp + λ-recompute to the *exact* pasted area; two-term loss `λ·CE(out,y_i)+(1-λ)·CE(out,y_j)` (one forward/backward — criterion called twice on the same logits). Standard apply prob p≈0.5.

## Short-schedule evidence (the load-bearing finding)
On ResNet-18/CIFAR-10 across 200→1200 epoch schedules (OpenMixup benchmarks; TransformMix arXiv:2403.12429; Mixup-Without-Hesitation arXiv:2101.04342):
- **CutMix converges FAST and gives its best *relative* advantage EARLY**: ≈96.1–96.2% @200ep vs vanilla ~94.9 and mixup ~95.6. It plateaus/declines with very long training.
- **plain mixup needs LONG schedules** (800–2000ep) to pay off; can even hurt in late epochs.
⇒ For a short/fast-training budget (~150 epochs here), **use CutMix, not mixup.**

## Mechanism vs occlusion aug (why it can stack on EXP-008)
Cutout/RandomErasing = single-image *occlusion* (delete info, keep hard label). CutMix = two-image *region mixing* with *soft (area-split) labels* — pastes real cross-class content (richer signal than zeros) and adds label-mixing regularization. Mechanistically distinct ⇒ plausibly complementary, not redundant.

## Implementation notes for THIS harness (constraints)
- **Throughput**: keep it free — draw the box center and λ on **CPU** (seeded torch CPU RNG / Python ints). Do NOT `.item()` a CUDA tensor inside the timed step (forces a sync → can cut `num_epochs`). `torch.randperm` for the batch permutation can be on-device (indexing only, no sync). Guard: `num_epochs` must stay in the normal band (~142–155).
- **Label-smoothing interaction**: CutMix soft labels compose linearly with `CrossEntropyLoss(label_smoothing=s)` (LS smooths the area-mixed two-hot target). Both soften targets → risk of over-softening/under-fit when stacked with LS=0.2 + Cutout12 + RandomErasing. Treat LS as a co-variable: test LS 0.2 vs 0.1 with CutMix.
- **Curriculum**: disabling CutMix in the low-LR tail (e.g. final 15%) lets EMA average clean-image iterates (most accuracy lands in the tail here) — common in fast recipes; weigh vs constant-p.
- α=1.0 (uniform λ) is the CutMix default; lower α → mixup-like bimodal λ (avoid).

## Status on this goal
Tested in **EXP-011** (`experiments/011/04-analysis.md`): **no-improvement**. CutMix ADDED on top of Cutout12+RandomErasing only matched baseline (best 96.40 @ LS0.2 vs 96.38; 96.32 @ LS0.1) — throughput-free and fully annealed (142 ep), but depressed early convergence (ep25 91.21 vs 92.31) with an unchanged annealed ceiling → redundant with the existing occlusion aug on this saturated net (input-space aug near saturation). LS retune (0.2→0.1) did not help. Untried: CutMix as a REPLACEMENT for Cutout/RE rather than an addition.

## Sources
- CutMix: https://arxiv.org/abs/1905.04899 ; mixup: https://arxiv.org/abs/1710.09412
- Short-schedule benchmarks: OpenMixup (github.com/Westlake-AI/openmixup); TransformMix https://arxiv.org/abs/2403.12429 ; Mixup-Without-Hesitation https://arxiv.org/abs/2101.04342
