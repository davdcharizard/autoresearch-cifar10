# Fast CIFAR-10 training recipes (DavidNet → hlb-CIFAR10 → airbench)

Standing reference for the fast-CIFAR-10 lineage that reaches 94–96% on a single GPU in seconds-to-minutes. Validated in part by EXP-001 (DavidNet → 95.22% in 300s on H20).

## Lineage & headline numbers
- **David Page, "How to Train Your ResNet" / cifar10-fast** (DAWNBench winner): wide-shallow **ResNet-9 / "DavidNet"** (~6.5M params), one-cycle LR, Cutout, label smoothing → **~94%** in ~24 epochs / <94s on a V100. Reimpl: davidcpage/cifar10-fast, 99991/cifar10-fast-simple, johanwind "94% in 94 lines/94s".
- **tysam hlb-CIFAR10**: hand-tuned reproduction; GELU activations, 3×3→2×2 conv, ~94% in ~6.3 A100-s.
- **Keller Jordan, cifar10-airbench** (arXiv:2404.00498): record family — **94% ~2.6s, 95% ~10s, 96% ~27–35s** on A100. Core trick = **frozen whitening initial conv** (eigendecomposition of ~5000 training patches, 3→24 ch, kernel 2, `torch.linalg.eigh`, `requires_grad=False`), GELU ConvGroup blocks (Conv→MaxPool→BN→GELU), flip TTA inside forward.

## Key techniques (all torch/torchvision-only, no new deps)
- **DavidNet/ResNet-9 architecture**: prep 3→64; stages 64→128(+Residual)→256→512(+Residual), MaxPool(2) each; global MaxPool; bias-free Linear; **logits ×0.125** ("output scale is important").
- **One-cycle LR** (triangular): linear ramp 0→peak over ~15% then decay to ~0. Peak ≈0.4 in the *mean-loss* convention (johanwind sums loss → its 0.6/512 LR + wd 5e-4×512 translate to mean-loss peak ~0.4–0.6, wd 5e-4). **EXP-001 used a time-based variant** keyed on `total_training_time/TIME_BUDGET_S` so it completes within a fixed time budget.
- **Regularization**: Cutout (after Normalize), label smoothing 0.1–0.2, SGD+Nesterov. **EXP-008 validated cutout 8→12 + light `torchvision.RandomErasing(p=0.25, scale=0.02–0.15, value=0)` → +0.38pp (96.00→96.38), throughput-free (CPU workers); now in the base recipe.** airbench96 uses cutout=12. Stronger aug is the proven lever to spend the ~4× epoch surplus on this saturated net.
- **Throughput**: bf16 autocast (no GradScaler) + channels_last + cudnn.benchmark; batch 512+.
- **Scaling axes for higher accuracy (airbench, arXiv:2404.00498)**: 94→95 is the WIDTH trick (block channels 64→128, 256→384) + a few more epochs; 95→96 is the DEPTH trick — "add a third convolution to each block" — plus 12px Cutout (we have) and more epochs (we have ~4×). **EXP-021 TESTED this here and it does NOT transfer**: a compile-funded 2nd ReZero GatedResidual@8×8/layer2 (+1.18M, 152 ep fully annealed) TIED the same-session compiled control (96.26 vs 96.29, −0.03pp). Why airbench's depth helps at 96 but not us: airbench runs ~40 ep and IS capacity-bound there; our net runs ~150 ep and is already at a generalization ceiling (EXP-014 width + EXP-021 depth both flat). Do NOT add depth/width to THIS backbone — the within-DavidNet capacity surface is exhausted. The only untested depth variant is airbench-faithful "3rd conv in EVERY block" (vs our layer2-only), but the prior is now poor.
- **Alternating (derandomized) flip (airbench)**: replace i.i.d. RandomHorizontalFlip with antithetic per-image flip parity (`flip iff (epoch+idx)%2`) so every image is seen once in each orientation per 2-epoch window. "Improves the performance of every training considered" where flip helps. Throughput-free variance reduction, distinct from the saturated aug-content lane. UNTRIED — strong cheap follow-up lever (EXP-021 idea-02).
- **Whitening front-end** (airbench): biggest convergence accelerator; UNTRIED here — top next-step candidate (see experiments/001 proposals/idea-03.md).
- **Flip TTA**: average logits of x and x.flip(-1) inside `forward`, gated on `not self.training`; legitimate since the frozen eval calls `model(inputs)` directly. ~+0.2–0.4pp. UNTRIED.

## Constraint notes for THIS goal
- Normalization must match the frozen eval exactly: mean=(0.4914,0.4822,0.4465), std=(1,1,1). Whitening patch stats must be computed in this same space.
- Only `train.py` editable; no new packages; no seed hacking; ≤1 val/epoch; eval harness frozen.

## Sources
- https://johanwind.github.io/2022/12/28/cifar_94.html
- https://myrtle.ai/how-to-train-your-resnet-3-regularisation/ ; https://myrtle.ai/2018/09/24/how-to-train-your-resnet-5
- https://github.com/davidcpage/cifar10-fast ; https://github.com/99991/cifar10-fast-simple
- https://github.com/KellerJordan/cifar10-airbench ; https://github.com/tysam-code/hlb-CIFAR10 ; https://arxiv.org/abs/2404.00498
- "Bag of Tricks for Image Classification" He et al. CVPR 2019, arXiv:1812.01187 (zero-γ BN, label smoothing, cosine, no-decay-on-bias)
