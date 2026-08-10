# Proposal: Canonical Full Preactivation, Without Zero-Gamma

## Decision

Test a **paper-faithful full-preactivation width-2 ResNet-20** on the complete accepted EXP-010 training recipe, and do **not** combine it with zero-initialized BN scale in this experiment. This is an architectural reorder only: preserve p=0.5 alpha-1 CutMix during the N1/M7 plateau, the 80% hard weak-tail transition, all-parameter `1e-4` decay, SGD, LR schedule, seed, timer, and evaluator.

The executable residual unit is:

```text
pre = ReLU(BN_in(x))
residual = Conv3x3_stride(pre)
residual = Conv3x3(ReLU(BN_out(residual)))
out = shortcut + residual                 # no activation after addition
```

Use the following boundary semantics, which are part of the specification rather than implementation latitude:

- Remove the current stem BN/ReLU. The stem is only `Conv3x3(3, 32)`.
- For ordinary same-shape units, `shortcut = x`, preserving an operation-free identity path across residual sums.
- For the first unit after the stem, apply its `BN_in-ReLU` before the split and use `shortcut = pre`. This implements the first-unit special case described in the Identity Mappings appendix; there is no preceding residual sum whose raw identity path must be preserved.
- For the two dimension-changing units, apply the existing parameter-free Option-A stride-2 slice and right-channel zero pad to `pre`, not raw `x`. The spatial rule remains `[:, :, ::2, ::2]`, and 32 then 64 channels are appended on the high-channel side. This adapts the paper's preactivated projection-shortcut rule to the accepted Option-A shortcut without adding learnable parameters.
- After the ninth addition, apply one new `BatchNorm2d(128)` and ReLU, then retain adaptive average pooling and the existing linear classifier.

The implementation should make the exceptional shortcut source explicit (for example, `preactivate_shortcut` for the network's first unit and `need_pad` units); it must not accidentally send `pre` through every same-shape shortcut.

## Why Preactivation, Not the Naive Combination

He et al.'s primary identity-mapping study derives a direct forward and backward path when both the shortcut and the after-addition function are identities. Its full `BN-ReLU-weight-BN-ReLU-weight-add` unit improved CIFAR-10 error from 6.61% to 6.37% for ResNet-110 and from 5.93% to 5.46% for ResNet-164; the benefit became much larger at 1001 layers. The authors attribute the result to easier optimization plus the regularizing placement of BN. This is direct CIFAR evidence for the proposed ordering, although it is evidence at substantially greater depth than the local ResNet-20.

Goyal et al. provide separate primary evidence for zero-initializing the scale of the last BN in a **postactivation** residual branch: on ImageNet ResNet-50 at batch 256, top-1 error improved from `23.84 +/- 0.18` to `23.60 +/- 0.12`. The technique is compatible with the current block because its `bn2` is after `conv2` and feeds addition directly, so gradient reaches its scale even when the residual output starts at zero.

It is not directly composable with canonical full preactivation. In `BN_in-ReLU-Conv-BN_out-ReLU-Conv`, there is no BN after the final convolution. Zeroing `BN_out.weight` with its default zero bias makes the input to the second ReLU exactly zero. PyTorch's ReLU derivative at zero is zero, so the BN scale/bias, `conv2`, and all earlier residual-path parameters receive no update; the branch remains identically dead. Adding a third post-`conv2` BN with zero scale would avoid that dead branch because its scale sees gradient through addition, but it would add nine BN operations, 1,344 affine parameters, and a new normalization mechanism. That hybrid is neither the paper's controlled reorder nor a compute-neutral test under the fixed-time protocol.

Therefore initialize all candidate BN scales to one and biases to zero, retain Kaiming initialization for every convolution/linear layer, and isolate full preactivation. A later experiment can test zero-gamma on the accepted postactivation architecture if this candidate loses; it should not be folded into EXP-012.

## Capacity and Fixed-Time Cost

The candidate has exactly **1,073,962 trainable parameters**, matching EXP-010:

- Convolution and classifier shapes are unchanged.
- Current BN affine parameters are stem `64` plus block totals `384 + 768 + 1,536 = 2,752`.
- Preactivation BN affine parameters are block totals `384 + 704 + 1,408` plus final `256 = 2,752`.

It also preserves exactly 19 convolutions, 19 BN calls, and 19 ReLU calls per forward pass. Even the total number of per-example BN activation elements is unchanged at 376,832: moving normalization from the stem to the classifier boundary offsets the larger pre-stride `BN_in` tensors at the two stage transitions. Runtime should consequently be close to neutral, but altered kernel shapes/order still require measurement rather than assumption.

Before the full run, use one idle 97,871 MiB H20 for an interleaved paired control/candidate benchmark with batch 128 and the accepted soft-target CE path. Warm both models, then collect at least 500 synchronized forward/backward/SGD steps per model with alternating order. Require:

- candidate/control median synchronized step time `<= 1.03`;
- both timing distributions stable (CV `<= 2%`) and finite losses/gradients;
- projected retention `floor(26,898 * control_time / candidate_time) >= 26,091` steps (97% of EXP-010);
- peak allocated VRAM `<= 650 MB`; and
- conservative projected end-to-end runtime `< 540s`.

Any miss is a no-go, not permission to change width, batch size, compilation, loader mechanics, or the candidate definition. The data pipeline is byte-identical, so a repeated worker-throughput benchmark is unnecessary; static diff and the normal full-run lifecycle checks remain required.

## Correctness Gates

Before H20 timing, run focused structural tests in a disposable process:

- require exactly 1,073,962 parameters and logits `[128, 10]` for `[128, 3, 32, 32]` input;
- assert every block's first BN has `in_channels` features and second BN has `out_channels` features;
- assert all BN scales are one, including every second BN, and all BN biases are zero;
- with residual convolutions zeroed in eval mode, require an ordinary same-shape unit's output to equal raw `x`, the first unit's output to equal its preactivation, and transition outputs to equal the exact preactivated Option-A slice/pad tensor;
- hook all additions and prove there is no post-add ReLU; require only the final network BN-ReLU before pooling;
- verify finite nonzero gradients for both convolutions and both BN affine tensors in every block after one ordinary backward pass; and
- require the only tracked diff to be the model architecture in `train.py`. CutMix, transforms, loaders, optimizer groups, schedule, timing, evaluator cadence, and seed must remain unchanged.

## Hypothesis and Decision Rule

**Hypothesis:** clean identity propagation through all same-shape residual sums, coupled with final BN-ReLU feature conditioning, will improve representation/optimization enough for the accepted width-2 p=0.5 CutMix recipe to reach **at least 94.25% best test accuracy**, with a point prediction of **94.30%**, while retaining at least 97% of EXP-010's optimizer exposure.

The primary decision is unchanged: one seed-42 run is an improvement only at `best_test_acc >= 94.25%`; a lower complete run is preserved as no-improvement with no reroll. Record the 80% clean checkpoint, first weak checkpoint, final NLL, endpoint slope, step retention, and best/final gap as mechanism evidence, never as alternate acceptance gates.

## Risks and Interpretation

- **Depth mismatch is the main risk.** The primary preactivation gains were measured at 110+ layers, where direct paths solve a larger optimization problem. A 20-layer network may gain less than the required 0.10 point or may prefer the current post-add nonlinearity.
- **The shortcut exception is consequential.** Preactivating the first and two transition shortcuts follows the paper's boundary logic but changes the values transported by Option A. Results identify the specified full-preactivation network, not the within-stage reorder in isolation.
- **The accepted recipe is already strongly regularized.** BN before every residual convolution and signed post-add features may change CutMix/RandAugment fitting and the hard-tail recovery rate even at equal exposure.
- **Zero-gamma is intentionally untested.** A loss does not reject zero-initialized final BN on the current postactivation block; it rejects this full-preactivation specification. Conversely, a win should not be attributed to identity initialization because residual branches do not start at zero.
- **The margin is small relative to one-seed variation.** A bare 94.25-94.35 result satisfies the fixed protocol but should be reported as weak causal evidence; the protocol forbids seed rerolls.

## Primary Sources

- K. He, X. Zhang, S. Ren, and J. Sun, [Identity Mappings in Deep Residual Networks](https://arxiv.org/pdf/1603.05027), ECCV 2016. See Eq. 9, Tables 2-3, and the appendix's first/last-unit and preactivated projection details.
- K. He et al., [official `resnet-pre-act.lua` reference implementation](https://github.com/KaimingHe/resnet-1k-layers/blob/master/resnet-pre-act.lua). It uses BN-ReLU before weight layers, raw identity shortcuts for same-shape units, preactivation before dimension-changing splits, and final BN-ReLU before pooling.
- P. Goyal et al., [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/pdf/1706.02677), 2017. Section 5.1 and Table 2b isolate zero initialization of the final BN scale in postactivation ResNet-50.
