# ACNet: Asymmetric Convolution Blocks / structural reparameterization (arXiv 1908.03930, ICCV 2019)
- **Authors**: Ding, Guo, Ding, Han (family: RepVGG CVPR 2021, DBB CVPR 2021 by the same group)
- **URL**: https://arxiv.org/abs/1908.03930 | reference impl https://github.com/DingXiaoH/ACNet
- **Status in project**: measured EXP-064 — **NO LAUNCH, family cost-closure**: full ACB toll ratio 1.930 (43.15ms vs control 22.36ms; required gain 1.19 > published max 1.11); minimum DBB-lite variant (3x3 ∥ 1x1) 28.61ms still requires ≈ 0.69 > single-branch ablation gains. On a launch-bound box the 1D convs hit slow odd-shape kernels; "free at inference" inverts under a train-time budget. Zero charged seconds.

## Claim

Replace every KxK conv with an Asymmetric Convolution Block (ACB): three parallel branches —
KxK + BN, 1xK + BN, Kx1 + BN — summed before the nonlinearity. Train the branched net; after
training the three branches fold EXACTLY into one KxK conv (+bias) by BN-fusion + center-aligned
kernel addition, so inference cost is unchanged. Top-1 gains on CIFAR-10 at fixed epochs:
**+0.35 to +1.11** across VGG, ResNet-56, WRN-16-8, DenseNet-40 (WRN-16-8 notable: gains hold
on a wide net like our 4x). ImageNet: +0.5–1.5 across AlexNet/ResNet-18/DenseNet-121.

## Mechanism

NOT regularization and NOT eval-time capacity: the folded eval model is byte-shape-identical to
the plain net. The gain is attributed to (a) per-branch BN giving the optimizer independently
adaptive scales for the kernel skeleton (the central cross of the 3x3) vs its corners —
training-dynamics reparameterization; (b) enhanced robustness to rotational/flip distortion.
The same group's DBB (CVPR 2021) generalizes the branch set (1x1, 1x1-KxK, 1x1-avgpool) with
similar gains — confirming the family mechanism is the reparameterized optimization geometry,
not the specific branch shapes.

## Project-relevant readings

- Uniquely evades two closed axes at once: capacity (eval params unchanged — not a capacity
  increase) and reg-dose (branches are not a regularizer; train loss typically DROPS).
- The toll is train-time dt: +2 convs (+2 BNs) per conv site, 18 sites in ResNet-20 + stem.
  On our launch-bound box (2.5–2.8ms/block, EXP-034/040) extra small kernels price at launches,
  not FLOPs — MUST be probe-gated with a pre-registered inequality before any charged run.
- No folding machinery needed for our Eval: evaluating the branched module in eval mode is
  mathematically identical to evaluating the folded net (BN-eval is affine; fold is exact
  algebra) — fold only if eval wall-time matters.
- Absorption risk: published baselines use standard crop+flip aug (~94–95% level), NOT
  TA+RE-heavy recipes; external transfer record here is 0-for-18. Counterargument for the
  screen: the mechanism (per-branch adaptive kernel-part scaling) is something heavy aug
  cannot supply — it changes the optimizer's parameterization, not the data distribution.
