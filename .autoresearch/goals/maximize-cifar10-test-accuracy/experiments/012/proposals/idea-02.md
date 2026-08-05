# Proposal: Exact 8x8 Bottleneck Residual Refinement

## Experiment

Append exactly one half-width pre-activation bottleneck residual unit after the accepted model's complete `layer3` and immediately before its existing final `BN-ReLU-pool-fc` path. Keep the accepted WRN stage widths `[32,64,128]`, stage depths `[2,2,2]`, FP32 numerics, initialization policy, optimizer, schedule, alpha-0.2 mixup through 65%, hard-label tail, seed, data path, and evaluation cadence unchanged.

This is one fixed topology, not a bottleneck-width search. There is no adaptive fallback to a different ratio, insertion point, endpoint initialization, width, depth, LR, or regularizer.

## Exact Topology

Let `x` be the `[N,128,8,8]` output of accepted `layer3`. Add a new `PreActBottleneck(128,64)` whose residual branch is:

```text
x --------------------------------------------------------------------+
 |                                                                     |
 +-> BN(128) -> ReLU -> Conv1x1(128->64, s1, p0, bias=False)           |
     -> BN(64) -> ReLU -> Conv3x3(64->64, s1, p1, bias=False)          |
     -> BN(64) -> ReLU -> Conv1x1(64->128, s1, p0, bias=False) --------+ -> add
```

The shortcut is the literal input tensor `x`: no projection, normalization, pooling, or parameterized scaling. There is no activation after the addition inside the new unit. The accepted model's existing final `BatchNorm2d(128)` and ReLU therefore consume the residual sum exactly as they consume accepted `layer3` today. All three convolutions preserve 8x8 spatial resolution.

Implement it as a distinct module and call it once in `WideResNet.forward` between `layer3` and the final `bn`. Do not express it by changing the accepted stage block count: this guards against accidentally recreating closed `[2,2,3]`.

## Initialization

Use the accepted `WideResNet._weights_init` without any special case:

- all three new convolution weights: Kaiming normal, `mode="fan_out"`, `nonlinearity="relu"`;
- all three new BatchNorm scales: one;
- all three new BatchNorm biases: zero;
- convolution biases: absent.

Do not zero-initialize `conv3` or the final bottleneck BatchNorm scale. Zero-endpoint initialization is a separate optimization-geometry treatment from the collected ideas; combining it here would prevent attribution to the compute-efficient topology.

## Exact Cost Arithmetic

Trainable parameters in the new unit, including affine BatchNorm parameters and excluding non-trainable running buffers:

| Component | Arithmetic | Parameters |
|---|---:|---:|
| `BN(128)` | `2 * 128` | 256 |
| `Conv1x1 128->64` | `128 * 64 * 1 * 1` | 8,192 |
| `BN(64)` | `2 * 64` | 128 |
| `Conv3x3 64->64` | `64 * 64 * 3 * 3` | 36,864 |
| `BN(64)` | `2 * 64` | 128 |
| `Conv1x1 64->128` | `64 * 128 * 1 * 1` | 8,192 |
| **Added total** | | **53,760** |

The accepted model has 691,674 trainable parameters, so the exact candidate total is **745,434**, an increase of 7.772%.

Convolution MACs per image at 8x8, counting one multiply-accumulate per output-channel/kernel-input pair and excluding BN, ReLU, and addition:

| Component | Arithmetic | MACs/image |
|---|---:|---:|
| `Conv1x1 128->64` | `8 * 8 * 128 * 64` | 524,288 |
| `Conv3x3 64->64` | `8 * 8 * 64 * 64 * 9` | 2,359,296 |
| `Conv1x1 64->128` | `8 * 8 * 64 * 128` | 524,288 |
| **Added total** | | **3,407,872** |

Including the accepted model's 101,106,944 convolution/linear MACs per image, the candidate total is **104,514,816**, an increase of 3.371%.

For comparison, closed EXP-011's full 128-channel basic block added 295,424 parameters and 18,874,368 MACs/image. This bottleneck uses 18.20% of that block's added parameters and 18.06% of its added MACs. Unlike closed EXP-010, it does not widen the stage output, transition projection, final normalization, or classifier; unlike closed EXP-011, it does not add two dense 128-channel 3x3 transforms. It adds a three-convolution reduction-transform-expansion path with a fixed rank-64 internal representation.

## Mechanism

The two closed capacity probes suggest that additional low-resolution transformation is directionally useful, but the full extra block's 0.2782 test loss indicates that dense raw depth can amplify confident specialization. This candidate adds another nonlinear residual refinement while sharply restricting its internal channel dimension and preserving nearly all accepted exposure. The 1x1 reduction learns which 64-dimensional channel subspace to refine, the 3x3 transform mixes that subspace spatially, and the final 1x1 expansion returns a correction in the accepted 128-channel representation.

The treatment is therefore not a smaller retry of either closed setting. It tests whether a constrained low-rank residual correction has a better generalization-per-MAC tradeoff than dense width or depth.

## Preflight And Gates

Use the same fail-closed, evaluator-free production-path benchmark established by EXP-010/011. Compare exact accepted and candidate models under reset initialization seeds and independent initially identical training RNG streams. Reproduce the complete timed FP32 step, selective SGD/Nesterov groups, pinned host copies, nonblocking transfers, LR writes, mixup sampling/permutation, finite-loss check, backward, optimizer step, and final synchronization. Measure separate 50%-progress mixup and 80%-progress hard-label windows in the fixed interleaved order used by EXP-011, then combine median step times with the preregistered 65/35 time weights.

The static design cost predicts at least 95% production throughput retention and at least `141.9 * 0.95 = 134.805` passes. Because the proposal's central claim is efficiency, require all of the following before the one scored run:

- every per-model/per-regime population CV ratio is at most 5%;
- aggregate candidate throughput retention is at least **92%**;
- calibrated exposure projection is at least **130.5 passes** (`141.9 * 0.92 = 130.548` before display rounding);
- the standard mechanism-interpretation floor of 120 passes is consequently satisfied;
- exact parameter counts, shapes, topology, finite FP32 loss/update, local data, one H20, and fail-closed evaluator conditions all pass.

The 95%/134.805 figures are the engineering expectation, while 92%/130.5 are the preregistered launch gates. If a gate fails, do not score and do not substitute another bottleneck. BN/ReLU and small-kernel launch overhead are the main reasons measured latency may exceed the 3.371% MAC increase.

## Required Topology Tests

- Accepted and candidate totals are exactly 691,674 and 745,434 trainable parameters.
- Both models produce finite FP32 logits of shape `[256,10]`; the candidate supports a finite cross-entropy backward and accepted SGD step.
- The accepted stem and all three stages remain byte-for-byte structurally equivalent: widths `[32,64,128]`, depths `[2,2,2]`, original first-block strides/projections, and original identity shortcuts thereafter.
- The new module is called exactly once after `layer3`; hooks observe input and output shape `[N,128,8,8]`.
- `bn1/conv1` are 128 and `128->64`, kernel 1, stride 1, padding 0, no bias.
- `bn2/conv2` are 64 and `64->64`, kernel 3, stride 1, padding 1, no bias.
- `bn3/conv3` are 64 and `64->128`, kernel 1, stride 1, padding 0, no bias.
- The shortcut has no module or parameters. With the residual branch output intercepted as zero, the module output equals its input exactly; there is no post-add activation.
- The existing final BN remains `BatchNorm2d(128)` and the classifier remains `Linear(128,10)`.
- Initialization assertions under a fixed construction seed confirm all new BN scale/bias tensors are one/zero and that no convolution is zero-initialized; construction does not consume any conditional or data-dependent choice.
- A module-level MAC assertion reproduces 3,407,872 added and 104,514,816 total MACs/image.

## Risks And Interpretation

- CIFAR WRNs favor wide basic blocks; a half-width bottleneck may constrain rather than enrich the representation.
- Three extra BN/ReLU/conv sequences can cost more wall time than MAC arithmetic predicts, especially because 1x1 kernels are small at 8x8.
- The immediately active randomly initialized branch may perturb a well-calibrated accepted optimization path; this is intentional for clean attribution and is not rescued with zero initialization.
- Additional BN running statistics and capacity can still worsen confidence/generalization even though parameter growth is modest.
- A positive delta smaller than 0.10 points is still a formal no-improvement and must not be rerun.

A stable run below 94.17% with at least 130.5 projected and 120 realized passes rejects only this exact post-stage-3, ratio-2 bottleneck with accepted initialization. It does not reject other bottleneck ratios, in-stage replacement designs, or endpoint initialization, but none may be tried as an adaptive fallback in EXP-012.

## Falsifiable Hypothesis

If constrained nonlinear refinement, rather than dense low-resolution capacity, is the useful signal behind EXP-010/011, then the exact post-stage-3 `128->64->64->128` bottleneck will retain at least 92% matched production throughput, realize at least 130.5 projected and 120 scored data passes, and achieve **`best_test_acc >= 94.17%`** in one fixed-seed 300-second run. Failure of either the efficiency gates or the accuracy threshold falsifies this exact proposal without a ratio, initialization, or placement retry.

## Evidence

- `experiments/010/04-analysis.md`: selective final-stage width scored 94.11% at 132.16 passes with 0.2457 final test loss.
- `experiments/011/04-analysis.md`: a dense extra 8x8 block scored 94.15% at 132.92 passes but worsened final test loss to 0.2782.
- `knowledge/papers/wide-residual-networks.md`: shallow/wide residual allocation is generally more compute-efficient than raw depth, motivating a constrained transform rather than another dense block.
- `experiments/012/01-brainstorm.md`: diagnosis identifies generalization and representation efficiency, not memory or raw exposure, as the current gap.
