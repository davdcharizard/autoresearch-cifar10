# Proposal: Fixed-Temperature Cosine-Normalized Classifier

## Summary

Replace only EXP-011's final affine logit computation with a fixed-scale cosine
classifier. L2-normalize each 256-dimensional pooled feature and each of the ten
classifier weight rows, compute their cosine similarities, and multiply by
`40.0` (temperature `0.025`) before the unchanged cross-entropy or CutMix loss.

Keep the existing `nn.Linear(256, 10, bias=True)` object and its initialization
to preserve the parent's parameter construction and RNG state. Ignore its bias
in forward and freeze it at its existing zero initialization. This leaves the
state-dict and EMA inventory unchanged while making the cosine classifier
bias-free as intended.

Everything else remains EXP-011: WRN-16-4, random crop/flip, front-loaded
probabilistic CutMix, drop path, clean-tail period-two SAM, cadence-31
charged-time full-state EMA, one evaluation per epoch, BF16/channels-last,
seed 42, 300 charged seconds, and physical GPU 0.

## Motivation

EXP-011 is the global-best parent at 95.61%, but its last 16 EMA checkpoints
average 95.493 and end at 95.46. The current limiter is a stable decision-boundary
gain rather than memory or raw throughput. EXP-012's spatial regularizer
regressed, so a classifier-geometry mechanism is appropriately orthogonal.

Standard affine softmax lets class-weight norm and feature norm affect logits.
Cosine normalization removes both radial degrees of freedom and trains angular
class separation directly. A fixed scale then controls softmax sharpness. This
can improve class boundaries with no additional feature extractor, model pass,
data view, target transform, or evaluator change.

`experiments/013/papers/temperature-cosine-softmax.md` supplies unusually
relevant evidence: on ResNet-34/CIFAR-10, standard softmax reached 95.56%, while
fixed scale 40 reached 95.85%, a +0.29-point gain. The paper also shows strong
scale sensitivity and that its learned-scale method reached only 95.49%, so a
learned temperature is not justified here.

## Preregistered Scale Without Validation Search

Set:

```python
COSINE_SCALE = 40.0
COSINE_TEMPERATURE = 1.0 / COSINE_SCALE  # 0.025
COSINE_EPS = 1e-6
```

Scale 40 is fixed because it is the supplied paper's directly reported
competitive operating point on the same dataset and a residual backbone. Do
not test scales 10, 20, 30, or 60 and select by test accuracy; do not make scale
learnable.

The choice is also geometrically plausible for this model. At initialization,
approximately random unit vectors in 256 dimensions have cosine standard
deviation near `1/sqrt(256) = 0.0625`; multiplying by 40 gives logit standard
deviation around 2.5. Scale 1 would yield nearly uniform logits and weak
optimization, consistent with the paper's small-scale failures. Scale 60 would
start around 3.75 and make early softmax substantially sharper. Scale 40 is the
single literature-anchored midpoint, not a locally tuned scalar.

The cosine bound gives logits in `[-40, 40]`, which is numerically safe for
FP32 cross-entropy. `COSINE_EPS=1e-6` is far below normal feature and Kaiming
weight norms but protects against division by zero.

## Exact Classifier Mechanism

Keep model construction:

```python
self.fc = nn.Linear(256, num_classes)
```

Keep the parent's `_weights_init` call unchanged, so `fc.weight` receives the
same Kaiming-normal draw and `fc.bias` is zero. Immediately after model-wide
initialization, call:

```python
self.fc.bias.requires_grad_(False)
```

The frozen bias remains in `named_parameters()` and the state dict but is not a
trainable SAM parameter and is never used in forward. The existing optimizer
may receive it through `model.parameters()`; PyTorch skips parameters whose
gradient is `None`. Add an assertion that the bias stays exactly zero.

Replace only the final return:

```python
out = F.relu(self.bn(out))
out = F.adaptive_avg_pool2d(out, 1).flatten(1)

with torch.autocast(device_type=out.device.type, enabled=False):
    features = F.normalize(out.float(), p=2, dim=1, eps=COSINE_EPS)
    weights = F.normalize(
        self.fc.weight.float(), p=2, dim=1, eps=COSINE_EPS
    )
    logits = COSINE_SCALE * F.linear(features, weights, bias=None)
return logits
```

Disabling autocast only for the final 256-dimensional normalization and 10-way
matrix multiply makes cosine norms and logits FP32 under BF16 training. It does
not recast any spatial activation. Evaluation already runs FP32 and follows the
same code path. If the exact installed PyTorch build rejects CPU autocast with
this generic context in a smoke test, use a small device-type branch; do not
silently run CUDA normalization in BF16.

Do not normalize intermediate feature maps, add an angular margin, add label
smoothing, train a scale, retain affine logits as an auxiliary loss, or change
cross-entropy. This experiment tests exactly fixed-temperature cosine softmax.

## Initialization and RNG Parity

Constructing the same `nn.Linear` with bias and running the same model-wide
initializer consumes exactly the parent's CPU random draws in the same order.
Every shared initialized tensor, including raw `fc.weight` and zero `fc.bias`,
should be bit-identical to EXP-011 under seed 42, and the post-construction CPU
and CUDA RNG states should match.

The forward normalization is deterministic and consumes no RNG. Therefore,
for every shared step prefix:

- DataLoader order and crop/flip streams remain parent-equivalent;
- CutMix gate/geometry/permutation generators remain untouched;
- global CUDA drop-path masks remain unchanged;
- SAM's captured and replayed CUDA state remains exact;
- EMA cadence and state operations consume no new randomness.

Timing can change the number of steps before wall-clock phase boundaries, so
exact terminal CutMix/SAM/EMA counts may differ. The preflight bounds this
protocol-level effect.

The forward function is intentionally not parent-identical at initialization:
normalizing features and weights is the intervention. Preserving raw directions
and RNG isolates that intervention from a second random initialization change.

## CutMix, SAM, and EMA Compatibility

### CutMix

Both ordinary and paired cross-entropy terms consume the same scaled cosine
logits. Preserve the parent area-corrected weighting exactly. No target or
augmentation RNG changes. The scale is shared across both label terms, so the
CutMix interpolation semantics remain intact.

### SAM

`fc.weight` remains trainable and must have a finite nonzero gradient on both
ordinary and CutMix batches. Its gradient is predominantly tangent to each raw
weight row because forward uses normalized directions. It joins the existing
global SAM norm, rho-0.05 perturbation, snapshot, exact restore, and sole
Nesterov update.

The frozen unused bias is deliberately excluded by the current
`parameter.requires_grad` filter, avoiding the parent's strict missing-gradient
assertion. Add an explicit state audit that it is the only model parameter with
`requires_grad=False`, has no gradient, remains zero, and is absent from
`sam_parameters`.

On a SAM step, the second forward renormalizes the perturbed raw classifier
weight. CUDA RNG replay still reproduces all six drop masks; no classifier RNG
exists. BN suppression and parameter restoration remain unchanged.

### EMA

The `ChargedTimeEMA` state coverage remains complete because the same
`fc.weight` and `fc.bias` names/shapes exist. EMA averages raw classifier
weights with the same charged-time coefficient; every EMA evaluation then
normalizes the averaged rows. The frozen zero bias is copied/averaged/restored
as a constant state tensor.

No new persistent parameter or buffer is needed for scale: keep it as a module
constant, and log/assert `40.0`. The existing cadence-31 balance, BN buffer
handling, evaluation swap, optimizer identity, exact restore, and RNG audits
remain unchanged.

## Parameter, Compute, and Latency Estimate

Stored parameter count remains 2,748,890 because the ten-element bias stays in
state. Trainable count becomes 2,748,880. VRAM should be indistinguishable from
EXP-011's 1,222.4 MiB.

Per batch, added work is approximately:

- normalize 256 feature rows of length 256 (65,536 elements);
- normalize ten weight rows (2,560 elements);
- FP32 256x256 by 256x10 multiply (655,360 MACs).

This is tiny relative to roughly 100.5 billion convolution MACs per batch, but
several FP32 reduction/cast kernels replace one autocast linear kernel. Launch
latency, not arithmetic, is the risk. The classifier runs twice on periodic SAM
steps and on every evaluator batch.

Expected weighted charged overhead is below 2%, retaining about 25,300-25,800
steps. Peak VRAM should rise by less than 2 MiB. Total runtime should stay near
EXP-011's 447.9 seconds and below 600 seconds.

## Parent-Relative GPU-0 Preflight

Compare actual EXP-011 and cosine-classifier models in the same physical-GPU-0
harness, batch 256, BF16/channels-last, no compilation. Use identical initialized
shared tensors and synthetic inputs/targets. After at least 50 warmups, run five
alternating randomized-order paired rounds containing:

1. at least 200 ordinary forward/backward/Nesterov steps per arm;
2. at least 100 production-faithful two-pass SAM steps per arm;
3. at least 200 evaluation forwards per arm;
4. cadence-31 EMA updates and one swap/restore transaction for both arms.

Weight training latency using EXP-011's observed approximately 90.4% ordinary
and 9.6% SAM step mix. Proceed only if:

- candidate weighted median is at most `1.03 * parent`;
- candidate weighted p90 is at most `1.05 * parent`;
- `25,798 / median_ratio >= 25,000` projected steps;
- projected total runtime is below 600 seconds;
- peak VRAM remains below 1.25 GiB;
- all finite-gradient, SAM, EMA, RNG, and restore assertions pass.

Use only paired parent-relative gates; no absolute throughput floor. If the
fixed FP32 cosine implementation fails, reject the proposal rather than
changing scale or precision based on prospective metric payoff.

## Expected Accuracy and Testable Hypothesis

Parent EXP-011 has `best_test_acc=95.61%`; formal improvement requires at least
95.71%. Its last-16 EMA mean is 95.493. The preregistered hypothesis is:

> Fixed-scale-40 cosine softmax will reach best accuracy of 95.80-96.00%, raise
> the last-16 EMA mean to at least 95.70, retain at least 25,000 steps, and keep
> all parent CutMix/SAM/EMA invariants intact.

The expected +0.19 to +0.39 best-accuracy range is anchored by the paper's
+0.29 result on CIFAR-10/ResNet-34. A similar gain would clearly pass 95.71 and
would target the stable plateau rather than another isolated maximum.

Run one fixed scale and seed. Below 95.71 is a tree no-improvement. A best at
or above 95.71 with last-16 EMA mean below 95.59 (less than +0.10 over the
parent plateau) is formal max-selected success but falsifies the stable-boundary
hypothesis. Do not test another scale, temperature, learned scale, margin, or
bias after observing the metric.

## Risks

- **Scale sensitivity:** the supplied paper shows fixed scales can range from
  failed optimization to competitive accuracy; scale 40 may not transfer from
  ResNet-34 to this PreAct WRN/CutMix/SAM schedule.
- **Norm information loss:** pooled feature magnitude may encode confidence or
  example difficulty. Cosine normalization removes that signal.
- **Over-sharp mixed labels:** scale 40 can make CutMix's two target terms push
  strongly against a sharp distribution, slowing early optimization.
- **Weight-decay degeneracy:** radial changes to `fc.weight` do not change
  logits, so weight decay mostly changes raw norm and thereby gradient/SAM
  scaling rather than the represented classifier.
- **SAM geometry interaction:** Euclidean perturbation of raw weights followed
  by renormalization emphasizes tangent directions and changes the effective
  classifier perturbation relative to the affine parent.
- **EMA geometry interaction:** averaging raw vectors and then normalizing is
  not the same as averaging unit directions; antipodal or dispersed rows could
  shorten before normalization and become unstable.
- **Frozen-bias bookkeeping:** any accidental optimizer/SAM use or nonzero bias
  breaks the bias-free contract; audits must fail loudly.
- **FP32 launch overhead:** tiny reductions can be launch-bound even when FLOPs
  are negligible, as prior attention experiments showed.
- **Protocol noise:** historical max/tail variation reaches 0.29 points, so a
  bare +0.10 best delta is weak without a higher tail mean.

## Instrumentation

Log classifier type, scale, temperature, epsilon, stored/trainable parameter
counts, and frozen-bias status. After charged training, without another data
forward, report for online and EMA classifier states:

- raw row-norm min/mean/max and finite status;
- pairwise off-diagonal cosine min/mean/max between class rows;
- frozen bias max absolute value (must be zero);
- classifier online-to-EMA raw and normalized-direction distances.

Preserve all existing CutMix/SAM/EMA dose, decay, BN, RNG, evaluation-source,
and swap/restore audits. Durably transcribe the final summary, last-16 EMA
accuracies/mean/range, preflight ratios, and classifier diagnostics before
transient log cleanup.

## Verification

1. Unit-test that logits equal `40 * cosine`, lie within `[-40,40]`, are FP32,
   and are invariant to positive rescaling of features or individual weight
   rows within tolerance.
2. Verify zero/near-zero feature handling is finite under `eps=1e-6` and
   nondegenerate examples produce finite nonzero feature and weight gradients.
3. Construct parent and candidate under seed 42; assert every stored tensor and
   post-construction CPU/CUDA RNG state match exactly before forward, and the
   only trainability difference is frozen `fc.bias`.
4. Verify CutMix area-weighted loss uses the same logits and targets as parent,
   with no augmentation/RNG changes.
5. On a full BF16/channels-last SAM smoke, assert every trainable parameter has
   a gradient, bias has none, rho is 0.05, drop masks replay, BN updates once,
   and all trainable parameters restore exactly before one optimizer update.
6. Verify EMA inventory includes raw weight and frozen bias, cadence samples
   remain ordinary/SAM balanced, swapped cosine evaluation is finite, and full
   online state/RNG/optimizer identities restore exactly.
7. Pass the parent-relative GPU-0 latency gate above.
8. Launch once with
   `timeout 600s env CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`.
9. Confirm physical GPU 0 is the 97,871 MiB H20, approximately 300 charged
   seconds, under 600 total seconds, one evaluation per epoch, at least 25,000
   steps, complete parent/classifier audits, and both best/tail hypotheses.
10. Verify only `train.py` changed, evaluator/dependencies/seeds remain fixed,
    and durable evidence is recorded before transient cleanup.

