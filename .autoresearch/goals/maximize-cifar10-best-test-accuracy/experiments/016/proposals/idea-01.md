# Idea: Full-Path Anti-Aliased Transition Blocks

## Summary

Replace both stride-2 stage transitions in the accepted width-2 postactivation
ResNet-20 with full-path anti-aliased transitions. In each transition residual
branch, change the existing learned `3x3` convolution from stride 2 to stride 1,
then apply a fixed depthwise `3x3`, stride-2 binomial blur before `bn1`. In the
matching Option-A shortcut, replace raw `[:, :, ::2, ::2]` decimation with the
same fixed blur and sampling phase, then retain the exact existing high-channel
zero pad.

Use one exact kernel everywhere:

```text
1/16 * [[1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]]
```

The candidate changes only the two downsampling blocks. All seven same-shape
blocks, residual BN/ReLU ordering, Option-A channel policy, parameters,
initialization, width, classifier, EXP-010 data/CutMix recipe, optimizer,
schedule, timer, evaluator, workers, and seed remain accepted. It is one
anti-aliasing experiment with no combination, fallback, kernel search, or rescue
variant.

## Diagnosis and Local Evidence

EXP-010 remains the 94.15% frontier: width-2 postactivation ResNet-20 with
p=0.5 alpha-1 CutMix on N1/M7 views through 80%, followed by a hard weak tail.
It reached 89.73% at the switch, 93.16% on the first weak checkpoint, and
94.15% final/best with 0.1934 NLL and 26,898 updates.

The local history favors preserving active residual representation learning:

- Width 2 gained 1.25 points despite fewer updates, showing that a meaningful
  representation change can justify moderate fixed-time cost.
- Stronger CutMix, full preactivation, and selective zero-gamma all suppressed
  strong-phase fit. EXP-015 was compute-neutral and recruited every branch, yet
  its switch accuracy fell 3.25 points and final accuracy fell to 93.80%.
  EXP-016 must keep the accepted residual ordering, initialization, and all
  branch activity from step one.
- EXP-014's new raw-max path collapsed on its first update. A fixed normalized
  spatial filter adds no learned high-scale path, but finite first-update and
  short-fit checks remain required.
- The system profile finds backward at 75.46% of the 10.927 ms GPU-stage step.
  Dense transition convolutions and blur backward can cost real exposure;
  fixed coefficients and small parameter count are not timing evidence.

Zhang, *Making Convolutional Networks Shift-Invariant Again* (ICML 2019),
identifies aliasing when strided convolution or pooling discards samples without
low-pass filtering. Its applicable construction is to make a strided operator
dense, then insert fixed low-pass filtering before subsampling. The accepted
model downsamples the same transition input through two paths: a learned
stride-2 residual convolution and a raw Option-A stride-2 slice. Applying blur
to only one path would leave the addition phase-inconsistent and would not test
the paper's full-path mechanism.

Roshtkhari et al., *Balanced Mixture of Supernets for Learning the CNN Pooling
Architecture* (AutoML 2023), directly studies downsampling choices in ResNet-20
on CIFAR-10 and finds that placement/operator choices affect accuracy. It
supports the transition point as a real lever, but not this kernel's effect size;
EXP-016 must validate the exact configuration end to end rather than infer from
a supernet proxy.

Sources:

- [Making Convolutional Networks Shift-Invariant Again](https://proceedings.mlr.press/v97/zhang19a.html)
- [Balanced Mixture of Supernets for Learning the CNN Pooling Architecture](https://proceedings.mlr.press/v224/roshtkhari23a.html)
- `experiments/016/papers/blurpool.md`
- `experiments/016/papers/resnet20-downsampling-search.md`

## Exact Blur Operator

Implement a parameter-free `BlurPool` with a channel-specific nonpersistent
buffer:

```python
class BlurPool(nn.Module):
    def __init__(self, channels):
        super().__init__()
        kernel_1d = torch.tensor([1.0, 2.0, 1.0])
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        kernel = (kernel_2d / kernel_2d.sum()).view(1, 1, 3, 3)
        self.register_buffer(
            "kernel", kernel.repeat(channels, 1, 1, 1), persistent=False
        )

    def forward(self, x):
        return F.conv2d(x, self.kernel, stride=2, padding=1, groups=x.shape[1])
```

Pin every semantic choice:

- coefficients are the separable order-2 binomial filter above, normalized to
  DC gain one in the interior;
- kernel size is exactly 3; do not test sizes 2, 4, or 5;
- padding is symmetric one-pixel zero padding via `F.conv2d(..., padding=1)`;
  do not use reflection, replication, circular padding, or explicit cropping;
- stride is exactly 2 and dilation exactly 1;
- filtering is depthwise with one fixed kernel per channel and no learned
  weight, bias, affine scale, or optimizer membership;
- buffers are `persistent=False`, so they move with the model but add no
  checkpoint state key;
- buffers use the model's accepted FP32 dtype. Do not add autocast or dtype
  conversion logic for an untested precision path.

Zero padding is deliberate. It matches the accepted convolution boundary
convention, avoids adding a reflection-boundary mechanism, and gives output
centers at original coordinates `0, 2, 4, ...`. At boundaries the kernel's
effective DC sum is below one; that attenuation is part of this exact candidate
and must not be renormalized spatially.

## Exact Transition Semantics

Construct blur modules only when `stride == 2`. The local architecture has
exactly two such blocks and both also double channels:

| Block | Input | Dense residual conv | Residual blur output | Shortcut blur output | Final output |
|---|---|---|---|---|---|
| `layer2[0]` | `32x32x32` | `32 -> 64`, `3x3`, s1, p1 | `64x16x16` | `32x16x16` | `64x16x16` |
| `layer3[0]` | `64x16x16` | `64 -> 128`, `3x3`, s1, p1 | `128x8x8` | `64x8x8` | `128x8x8` |

Use two channel-specialized blur modules per transition so the kernels are
materialized once, not repeated in every timed forward:

```python
is_transition = stride == 2
conv_stride = 1 if is_transition else stride
self.conv1 = nn.Conv2d(
    in_channels, out_channels, 3,
    stride=conv_stride, padding=1, bias=False,
)
if is_transition:
    self.residual_blur = BlurPool(out_channels)
    self.shortcut_blur = BlurPool(in_channels)
```

Forward must be exactly:

```python
out = self.conv1(x)
if self.stride == 2:
    out = self.residual_blur(out)
out = F.relu(self.bn1(out))
out = self.bn2(self.conv2(out))

shortcut = x
if self.stride == 2:
    shortcut = self.shortcut_blur(shortcut)
    shortcut = F.pad(shortcut, (0, 0, 0, 0, 0, self.pad_channels))

return F.relu(out + shortcut)
```

This ordering is consequential. The residual branch computes its learned
`3x3` response densely, low-pass filters that response, then samples and
normalizes it. Do not place blur before `conv1`, after `bn1`, after ReLU, or
after `conv2`. The shortcut low-pass filters the raw transition input before the
same stride-2 sampling phase and then applies the accepted channel pad. Both
paths therefore arrive at addition on centers corresponding to input coordinates
`0, 2, 4, ...`.

All nontransition blocks retain byte-equivalent forward code. Do not blur the
stem, final pooling, same-shape shortcuts, or second convolutions. Do not replace
Option A with a learned projection, average pool, or blur-plus-projection.

## Parameters, RNG, State, and Compute

The learned convolution shapes are unchanged; only two `conv1.stride` values
change. The model therefore remains exactly **1,073,962 trainable parameters**.
The four fixed blur buffers contain:

```text
layer2 residual: 64 * 9 = 576 values
layer2 shortcut: 32 * 9 = 288 values
layer3 residual: 128 * 9 = 1,152 values
layer3 shortcut: 64 * 9 = 576 values
total: 2,592 FP32 buffer values = 10,368 bytes
```

They are nonpersistent and absent from `state_dict()`. Constructing fixed
`torch.tensor` values, outer products, normalization, and `repeat` consumes no
RNG. Module and learned-parameter construction order stays accepted, so cloned
seed-42 control/candidate builds must have bitwise-equal learned state and
identical post-construction CPU RNG. The new buffers must equal the declared
coefficients exactly and have `requires_grad=False`; they never enter SGD.

The two learned transition convolutions become four times denser spatially:

```text
accepted transition conv MACs: 4,718,592 + 4,718,592 per image
candidate transition conv MACs: 18,874,368 + 18,874,368 per image
increment: 28,311,552 MACs per image
```

The four depthwise blurs add another 331,776 MACs per image:

```text
64*16*16*9 + 32*16*16*9 + 128*8*8*9 + 64*8*8*9
= 331,776
```

The candidate thus adds about 28.64M forward MACs per image, roughly 17.8% of
the accepted approximately 161.3M forward path, before backward and kernel
effects. It also saves dense residual activations for backward. Parameter count
is not a cost proxy; paired H20 timing and memory are mandatory launch gates.

## Preserved Accepted Recipe

Outside `BlurPool` and the two transition paths, preserve EXP-010 exactly:

- width-2 postactivation ResNet-20, all BN scales one and biases zero, active
  residual branches from step one, Option-A high-channel zero padding, adaptive
  average pooling, and the accepted classifier;
- Kaiming-normal Conv/Linear initialization, unchanged BN epsilon/momentum, and
  no zero-gamma, preactivation, projection, pooling-readout, or attention change;
- batch 128, ordinary FP32 SGD, momentum 0.9, all-parameter coupled decay
  `1e-4`, and no Nesterov;
- N1/M7 crop/flip plus p=0.5 alpha-1 CutMix through 80%, then the hard crop/flip
  weak tail, with unchanged worker-side RNG isolation and target semantics;
- LR 0.1 through 80%, step to 0.01, cosine to `1e-4`, seed 42, 300 counted
  seconds, and 64,000 max steps;
- eight persistent workers, explicit shutdown/rebuild, checkpoints
  `(0.2, 0.4, 0.6, 0.7)`, dense weak-tail evaluation at most once per epoch,
  fixed evaluator, timeout, and summary schema.

No BF16 funding, compilation, fused optimizer, larger batch, LR compensation,
extra augmentation, kernel tuning, shortcut-only fallback, or second candidate
is allowed.

## Structural and Numerical Gates

Before H20 timing, require all of the following in disposable tests:

1. Exactly two transition blocks; their `conv1` modules have kernel `3x3`,
   padding one, stride one, unchanged channel shapes and bitwise-aligned weights.
   The other seven `conv1` strides remain one and their forwards are untouched.
2. Exactly four `BlurPool` modules with channel counts 64/32 and 128/64 at the
   two residual/shortcut paths. Every buffer is nonpersistent, nontrainable,
   FP32, grouped by channel, and coefficient-identical to the declared kernel.
3. For seeded tensors, require exact branch/output shapes in the table above,
   no off-by-one crop, and addition-compatible residual/shortcut tensors.
4. An impulse and coordinate-ramp test must prove both blur paths sample centers
   at `0, 2, 4, ...`; reject any one-pixel phase disagreement. A spatially
   constant interior must remain constant within numerical tolerance.
5. The shortcut must equal `F.pad(shortcut_blur(x), channel_pad)` exactly. It
   must not equal raw slicing on nonconstant inputs, proving blur is active.
6. Exactly 1,073,962 learned parameters, unchanged state-dict keys/shapes,
   bitwise-equal candidate/control learned initialization, identical CPU RNG,
   and no blur buffer in the optimizer.
7. Hard `[128]` and probability `[128, 10]` targets must produce finite logits,
   losses, gradients, and one-step parameters. Both transition residual and
   shortcut inputs must receive finite nonzero gradients; blur buffers must have
   no gradient.
8. Syntax, Ruff, formatting, pre-commit, and diff checks pass. The only tracked
   file is `train.py`, and data, optimizer, schedule, timing, evaluator, worker,
   and logging code remain unchanged.

Use synthetic tensors for structure and impulse/phase proofs. Following the
EXP-015 protocol learning, use materialized production-distribution N1/M7
hard/soft batches for optimization-collapse checks.

## Strong-Fit Safety Gate

Anti-aliasing is not expected to have EXP-014's new-path scale failure, but the
full-path filter can remove useful CIFAR detail. Materialize one fixed sequence
of 64 real strong batches and train aligned accepted/candidate models separately
on the identical sequence. Require finite state and loss throughout, no
one-class collapse, and candidate terminal loss EMA no more than 1.5x control.

This gate rejects only immediate numerical or gross fit failure. EXP-015 proved
that favorable 64-step loss can invert over the full 240-second strong phase;
it is not evidence of generalization and cannot select a kernel. Once the full
run launches, do not stop on a low finite checkpoint. The 87.08% switch marker
is a mechanism diagnostic, not an early-stop or retry trigger.

## Mandatory Paired H20 Timing Gates

On the single idle NVIDIA H20 near 97,871 MiB, benchmark the actual reviewed
candidate against accepted `7c1e7d8` in five alternating fresh-process pairs.
Use cloned seed-42 learned weights, separate ordinary SGD state, batch 128 pinned
host inputs, alternating hard/probability targets, 100 warmup steps, and 500
measured complete production-region steps: H2D, zero-grad, forward, CE,
backward, SGD, and terminal synchronization.

Record trial mean, median, p95, CV, projected steps, and peak allocation. Require:

- candidate/control median trial-mean training ratio at most **1.12**;
- `floor(26_898 * control_mean / candidate_mean) >= 24_016`, retaining at least
  89.3% of accepted updates;
- candidate p95 at most 1.18x control p95 and trial-mean CV below 3% for both;
- candidate peak allocation below 850 MB and no more than 256 MB above control;
- all timed losses and parameters finite; and
- a separate five-pair eval-mode inference ratio at most 1.15, CV below 3%, and
  conservative projected total runtime below 540 seconds.

The 1.12 ceiling is deliberately broader than compute-neutral candidates but
still requires the representation mechanism to retain roughly 24k updates and
more than twelve weak-tail epochs. The literature reports modest classification
gains, so a larger update sacrifice would overwhelm the likely signal. If any
timing gate fails, do not launch and do not fall back to shortcut-only blur,
blur-before-conv, a smaller kernel, or BF16.

At the gate boundary, about 24,016 steps imply 61.6 total dataset passes and
roughly 12.3 weak-tail passes. Expect approximately 17-18 unique evaluations
versus EXP-010's 19 because the unchanged 390-step epoch takes longer. Fewer
evaluation opportunities disadvantage the max metric but are intrinsic to this
fixed-time candidate. More than 19 is invalid; do not alter cadence to recover
opportunities.

## Testable Hypothesis

**Primary hypothesis:** consistently low-pass filtering both learned and
Option-A paths before each stride-2 sample will reduce transition aliasing and
preserve class-bearing spatial structure well enough to raise
`best_test_acc` from 94.15% to at least **94.25%**, while retaining at least
24,016 projected optimizer steps on the H20.

A plausible successful range is 94.25-94.50%. The upper end assumes improved
shift consistency complements crop/flip, N1/M7, and CutMix without erasing their
useful detail. Larger claims are unsupported because CIFAR images are only
32x32, the accepted network has just two transitions, and added compute reduces
optimization exposure.

Mechanism diagnostics are:

- 80% clean checkpoint above the 87.08% strong-underfit marker and preferably
  near EXP-010's 89.73%;
- first weak checkpoint near or above EXP-010's 93.16%;
- final NLL near or below 0.1934 and a rising late trajectory;
- actual steps consistent with timing, 45-55% strong CutMix, one worker switch,
  and 17-19 unique evaluations.

Only `best_test_acc >=94.25%` under all goal integrity conditions is an
improvement. Shift consistency is not the evaluator metric and no auxiliary
translation test can override primary accuracy.

## Risks and Failure Interpretation

- **Fine-detail suppression.** CIFAR objects occupy few pixels. The fixed blur
  may remove edges and textures needed for class separation, reproducing the
  strong-phase underfit pattern without changing residual initialization.
- **CutMix boundary dilution.** CutMix's area-proportional labels assume two
  class-bearing regions. Blur mixes activations across region boundaries at both
  transitions and can weaken small donor regions or misalign evidence with area.
- **RandAugment interaction.** Some N1/M7 operations intentionally create sharp
  local contrast. Filtering may improve invariance or erase useful augmented
  training signal.
- **Loss of exact Option-A transport.** The accepted shortcut copies old
  channels exactly at even coordinates. Blur replaces that unparameterized
  sample with a local weighted mixture, weakening literal identity gradients at
  the two most consequential boundaries.
- **Dense-convolution cost.** About 28.64M added forward MACs and saved dense
  activations can reduce both strong updates and hard-tail refinement. Timing
  gates limit but do not remove this fixed-time trade.
- **Boundary attenuation.** Zero padding gives less than unit kernel sum on image
  edges. Crop padding and CIFAR's small maps can make this more material than on
  ImageNet.
- **BN distribution change.** `bn1` now normalizes blurred dense-convolution
  outputs rather than directly sampled outputs. Its running statistics and
  weak-tail adaptation can change even with identical affine initialization.
- **Phase correctness bug.** A different pad convention or blur placement can
  offset one path by a pixel, making addition incoherent. Explicit impulse/ramp
  tests are mandatory.
- **Short-fit false confidence.** EXP-015's favorable 64-step fit did not predict
  the full strong phase. Only the switch trajectory and final run diagnose
  representation quality.
- **Single-seed threshold.** A bare 94.25-94.35 result is a valid but weak causal
  signal. Never reroll, change the kernel, or run a fallback after the result.

## One-Run Verification and Decision Rule

After every structure, safety, timing, memory, and wall gate passes, run exactly
once on the confirmed sole idle H20:

```bash
CUDA_VISIBLE_DEVICES=0 timeout 600s uv run train.py > run.log 2>&1
```

Require exit zero, finite summary fields, approximately 300 counted seconds,
total below 600 seconds, exactly 1,073,962 parameters, one augmentation switch
near 80%, eight stopped workers, unique evaluation epochs, no more than one
evaluation per epoch, correct hard/soft target provenance, and only the reviewed
`train.py` diff. Require seed 42 and never rerun a mechanically valid result.

The timing gate requires at least 24,016 projected steps before launch. If the
valid production run lands below that because of ordinary jitter, keep the
accuracy result but report weakened exposure attribution; do not compensate LR,
budget, or evaluation cadence post hoc.

- **Improvement:** accept only if `best_test_acc >=94.25%` and every formal goal
  condition passes.
- **Valid no-improvement:** revert the anti-aliased transitions after a correct
  result below 94.25%; use switch fit, first weak accuracy, NLL, exposure, and
  trajectory to distinguish smoothing harm from compute loss.
- **Mechanical failure:** repair only an implementation/environment defect that
  leaves the exact kernel, placement, padding, and candidate scope unchanged.
  Any alternate blur, path scope, precision, or optimizer change requires a new
  reviewed experiment.

## Attribution

The candidate changes both downsampling branches together because full-path
consistency is the anti-aliasing mechanism. It preserves learned parameter
values/count, RNG, residual activity, data, optimizer, and evaluation. The net
result includes three inseparable consequences: low-pass filtering, the dense
transition-convolution representation, and fewer fixed-budget updates.

It cannot attribute a result to residual blur versus shortcut blur separately,
compare kernel sizes/padding modes, or support a cheaper blur-before-convolution
variant. Those are explicitly outside EXP-016, and no fallback is permitted.
