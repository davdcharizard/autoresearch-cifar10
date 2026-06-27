# Squeeze-and-Excitation Networks (Hu, Shen, Sun, CVPR 2018, arXiv 1709.01507) + ECA-Net (Wang et al., CVPR 2020)

## Core method
- Per residual block: squeeze = global average pool over H×W → (N, C); excite = Linear(C→C/r) → ReLU → Linear(C/r→C) → sigmoid → per-channel scale of the block output (after last BN, BEFORE the residual add). Canonical r=16.
- Cost: ~C²/8 params per block (r=16), <1% FLOPs. The real cost on launch-bound hardware is the ~6 extra kernels per block (pool, 2 matmuls, 2 pointwise, broadcast mul) — small-tensor launches, not math.

## Published results (fixed-epoch regime)
- CIFAR-10 (paper's own table): ResNet-110 error 6.37 → 5.21 (+1.16 acc); consistent +0.5–1.2 across CIFAR ResNet depths. ImageNet: ResNet-50 +~1.0 top-1. Gains are converged-LEVEL effects — input-conditioned channel gating is added functional capacity, not a transit accelerant.
- ECA-Net: replaces the MLP with a k=3 1D conv across channels (~zero params), comparable accuracy — but the LAUNCH count is similar, so on launch-bound GPUs ECA saves little vs SE (project law EXP-034: per-block cost is launch-dominated, width-independent).

## Implementation notes for this repo (planned EXP-037)
- Insert after bn2, before the shortcut add, in BasicBlock.
- **Near-identity init against the deferral law (EXP-018: γ=0 start −0.99)**: fc2.weight zero-init + fc2.bias = 2.0 → sigmoid ≈ 0.881 constant at step 0 (mild uniform damping ≈ a 0.88 γ-scale the residual add tolerates); gradients to fc2.weight are nonzero (∝ ReLU(fc1·s)), so gates learn immediately. Exclude SE linears from the global `_weights_init` kaiming pass via a `skip_kaiming` attribute — kaiming on fc2 (fan_in C/16 → std up to 0.7) would randomize gates at init.
- Params at this repo's widths (64/128/256, r=16, 9 blocks): 1,740 + 6,552 + 25,392 = 33,684 → total 4,319,710.
- dt estimate on H20 (launch-bound, EXP-026/034 pricing): +1.5–3ms on the 22.4ms step. Deficit law: every +1ms ≈ −6 epochs ≈ −0.08pp. Net-positive window requires measured dt ≲ 26ms.

## MEASURED RESULT (EXP-037, 2026-06-10)
- Two clean runs: best 96.34/96.37 (129/128 epochs) = baseline mean − epoch deficit exactly; test_loss 0.188 ≈ family 0.185. SE's published LEVEL gain appeared at ZERO strength under TA+RE + completed one-cycle anneal (SENet CIFAR baselines are crop+flip) — the heavy-augmentation absorption law covers attention modules too.
- Pricing datum: 9 SE modules cost +1.7ms total (~0.19ms/module) on the 22.4ms step — micro-attachments fuse cheaply under default compile, an order below the 2.5ms/block whole-block launch cost (EXP-034).
- Near-identity init pattern VALIDATED as engineering: ep1 ~35 vs family ~38 (no EXP-018-class deferral); reusable for any gated module. Conclusion: attention axis closed in this regime; ECA/stage-3-only doses inherit the null.
