# Idea: Learned Anti-Loss Transition Shortcuts

## Summary

Replace only the two parameter-free Option-A transition shortcuts in the
accepted width-2 postactivation ResNet-20. Each new shortcut performs fixed
non-overlapping `2x2`, stride-2 average pooling, then a learned bias-free
stride-1 `1x1` projection and BatchNorm:

```text
x -> AvgPool2d(2, 2) -> Conv1x1(in, out, stride=1, bias=False) -> BN(out)
```

The accepted stride-2 `3x3` residual convolution and complete residual branch
remain byte-equivalent. The seven same-shape shortcuts remain exact identities.
No other topology, initialization, optimizer, data, CutMix, schedule, timing,
evaluation, worker, or seed change is allowed.

This is an information-preserving transition-shortcut bet, not full-path
anti-aliasing. Average pooling uses all four pixels in each `2x2` cell rather
than retaining only the top-left sample, and the learned projection transports
every input channel into every output channel rather than padding the new half
with zero shortcut signal. The residual path still uses its accepted learned
stride-2 convolution. No fallback, alternate ordering, or rescue variant is
part of EXP-017.

## Diagnosis and Experimental Context

EXP-010 remains the 94.15% frontier: width-2 postactivation ResNet-20 with
N1/M7 plus p=0.5 alpha-1 CutMix through 80%, followed by a hard weak tail. It
completed 26,898 updates, reached 89.73% at the strong/weak switch, immediately
recovered to 93.16% on the first weak checkpoint, and finished at its best with
94.15% accuracy and 0.1934 NLL.

The local evidence narrows the candidate sharply:

- Width 2 produced the largest gain (+1.25 points) despite losing updates. Its
  two Option-A transitions now copy only 32 then 64 old channels and supply no
  shortcut signal at all to the 32 then 64 newly introduced channels.
- Identity-oriented changes are a recurring failure. EXP-012 preactivation and
  EXP-015 selective zero-gamma suppressed switch fit by 2.85-3.25 points.
  EXP-017 keeps every accepted residual branch active from the first update and
  does not reorder ordinary blocks.
- EXP-014 proved that zero initial output does not control first-update scale;
  its raw-max classifier gradient was 4.10x the accepted path and collapsed to
  chance. EXP-016's full-forward BF16 width-3 candidate also hit a real-batch
  candidate-only concentration gate before timing. The learned shortcut needs
  aligned real-batch update and trajectory safety checks.
- EXP-013 found loader and host overhead negligible relative to model backward.
  The new pool, projection, and BN kernels therefore require measured H20 cost
  even though their arithmetic and parameter counts are small.
- A previously developed ordinary stride-2 `1x1` projection-plus-BN shortcut
  measured about 1.87% synchronized-step overhead. That is a useful feasibility
  prior, not a timing result for this average-pool-first candidate.

The accuracy limiter is representation/generalization under a short composite
strong phase. This candidate targets stage-boundary information transport
without weakening residual learning or increasing augmentation pressure.

## External Evidence and Transfer Limits

He et al., *Bag of Tricks for Image Classification with Convolutional Neural
Networks* (CVPR 2019), introduce the ResNet-D shortcut policy: move stride-2
downsampling out of the `1x1` projection, average-pool first, then apply a
stride-1 learned projection. Their mechanism is preservation of information at
stage transitions, and their measured gains show that shortcut downsampling is
not a neutral implementation detail.

The local Option-A shortcut is even more lossy than the projection shortcut
improved by ResNet-D: it selects one of four spatial positions and pads half the
output channels with zero. Average pooling plus projection therefore has a clear
local mechanism. However, the source experiments use ImageNet bottleneck
networks, not a CIFAR BasicBlock ResNet-20 with N1/M7 and CutMix. The published
gain is directional evidence, not a portable effect estimate.

Roshtkhari et al., *Balanced Mixture of Supernets for Learning the CNN Pooling
Architecture* (AutoML 2023), directly find that downsampling configuration
matters in CIFAR-10 ResNet-20. Their search supports the transition as an
accuracy lever but does not nominate this exact shortcut or justify proxy-weight
transfer. EXP-017 must train its exact configuration end to end.

Sources:

- [Bag of Tricks for Image Classification with Convolutional Neural Networks](https://openaccess.thecvf.com/content_CVPR_2019/html/He_Bag_of_Tricks_for_Image_Classification_with_Convolutional_Neural_Networks_CVPR_2019_paper.html)
- [Balanced Mixture of Supernets for Learning the CNN Pooling Architecture](https://proceedings.mlr.press/v224/roshtkhari23a.html)
- `experiments/017/papers/resnet-d-downsampling.md`
- `experiments/016/papers/resnet20-downsampling-search.md`

## Exact Shortcut Semantics

Add a marker subclass solely to control initialization:

```python
class ShortcutConv(nn.Conv2d):
    pass
```

For a block with `stride == 2` and `in_channels != out_channels`, construct:

```python
with torch.random.fork_rng(devices=[]):
    projection = ShortcutConv(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        bias=False,
    )
    init.kaiming_normal_(projection.weight)

self.shortcut = nn.Sequential(
    nn.AvgPool2d(
        kernel_size=2,
        stride=2,
        padding=0,
        ceil_mode=False,
        count_include_pad=False,
    ),
    projection,
    nn.BatchNorm2d(out_channels),
)
```

For every same-shape block, set `self.shortcut = nn.Identity()`. Forward becomes:

```python
out = F.relu(self.bn1(self.conv1(x)))
out = self.bn2(self.conv2(out))
out += self.shortcut(x)
return F.relu(out)
```

Pin the two exact transition inventories:

| Block | Input | Pool | Projection | Shortcut output | Residual output |
|---|---|---|---|---|---|
| `layer2[0]` | `32x32x32` | `32x16x16` | `32 -> 64`, `1x1`, s1 | `64x16x16` | accepted `64x16x16` |
| `layer3[0]` | `64x16x16` | `64x8x8` | `64 -> 128`, `1x1`, s1 | `128x8x8` | accepted `128x8x8` |

The average pool uses four disjoint samples anchored at original coordinates
`(0, 0)`, `(0, 2)`, ..., with no padding. Every input position participates in
exactly one pooled value. The projection is applied after spatial reduction and
cannot change sampling phase. BN follows projection immediately and uses
ordinary affine/running-stat defaults: gamma one, beta zero, running mean zero,
running variance one, accepted epsilon/momentum, and tracked statistics.

Do not reverse projection and pool, use a stride-2 projection, change the pool
to `3x3`, add pool padding, use max/blur pooling, omit BN, add a convolution
bias, retain zero padding after projection, or alter the residual branch. Those
are distinct candidates.

## Initialization, RNG, and State Discipline

The two new projections use the accepted Kaiming-normal Conv/Linear rule, while
their BN modules use PyTorch defaults. They do not use zero initialization,
identity-like partial copies, small BN gamma, residual/shortcut scaling, or
warmup. Standard gamma one makes all projected output channels active and
trainable from backward one; this is central to the anti-loss mechanism.

`nn.Conv2d` consumes CPU RNG in its constructor before the model-wide accepted
initializer later replaces its weight. To prevent the two added modules from
shifting shared model initialization, shuffle, worker seeds, or CutMix streams,
construct and explicitly Kaiming-initialize each `ShortcutConv` inside
`torch.random.fork_rng(devices=[])`. Then update `_weights_init` as follows:

```python
if isinstance(m, ShortcutConv):
    return
if isinstance(m, (nn.Conv2d, nn.Linear)):
    init.kaiming_normal_(m.weight)
```

The fork restores global CPU RNG after each projection, and the marker prevents
the later `self.apply` traversal from consuming a second draw for it. The two
projections are initialized from their contemporaneous seed-42 states but do not
advance global RNG. No separate seed or reseed is introduced.

Given cloned seed-42 construction states, all accepted same-named learned tensors
must be bitwise equal between candidate and control, and post-construction CPU
RNG must be byte-identical. Candidate state adds only these learned tensors and
their BN buffers:

```text
layer2.0.shortcut.1.weight
layer2.0.shortcut.2.weight / bias / running_mean / running_var / num_batches_tracked
layer3.0.shortcut.1.weight
layer3.0.shortcut.2.weight / bias / running_mean / running_var / num_batches_tracked
```

All new trainable tensors join the accepted single SGD group and receive LR 0.1,
momentum 0.9, and coupled weight decay `1e-4`, including projection weights and
BN affine parameters. Do not create a shortcut-specific LR or no-decay group.
The new BN running statistics update under strong and weak views and are never
reset or recalibrated outside ordinary training.

## Exact Parameter and Arithmetic Cost

The learned additions are:

| Component | Derivation | Parameters |
|---|---:|---:|
| Stage-2 projection conv | `64 * 32` | 2,048 |
| Stage-2 projection BN | `64 gamma + 64 beta` | 128 |
| Stage-3 projection conv | `128 * 64` | 8,192 |
| Stage-3 projection BN | `128 gamma + 128 beta` | 256 |
| **Added** | | **10,624** |
| Accepted model | | 1,073,962 |
| **Candidate total** | | **1,084,586** |

The projections add exactly 1,048,576 forward MACs per image:

```text
16*16*64*32 + 8*8*128*64
= 524,288 + 524,288
```

The two average pools compute 12,288 four-value channel windows, or 49,152
source contributions, per image:

```text
32*16*16 + 64*8*8 = 12,288 pooled outputs
12,288 * 4 source values = 49,152 source contributions
```

Projection MACs are about 0.65% of the accepted approximately 161.3M forward
MACs. Arithmetic is modest, but two pools, two tiny projections, two BNs, their
backward kernels, and new saved activations can be launch-bound. Prior Option-B
timing cannot replace paired measurement of this exact pool-first graph.

## Preserved EXP-010 Recipe

Outside the two shortcut modules and marker/init plumbing, preserve accepted
`train.py` exactly:

- width multiplier 2; all nine accepted residual branches, conv strides,
  postactivation ordering, stem, seven identity shortcuts, final average pool,
  and classifier;
- batch 128, FP32 hard/probability-target cross-entropy, ordinary SGD, momentum
  0.9, all-parameter decay `1e-4`, and no Nesterov;
- crop/flip plus N1/M7 and p=0.5 alpha-1 CutMix through 80%, then the hard weak
  crop/flip tail, including collator RNG isolation and target provenance;
- LR 0.1 through 80%, step to 0.01, cosine to `1e-4`, 64,000 max steps, seed 42,
  and accepted timer boundaries;
- eight persistent workers, explicit shutdown/rebuild, checkpoints
  `(0.2, 0.4, 0.6, 0.7)`, at most one evaluation per epoch, fixed evaluator,
  ten-minute timeout, and summary schema.

Do not combine with full-path blur, projection residual changes, ECA, Nesterov,
BF16, pooling/readout changes, zero-gamma, altered CutMix, or any fallback.

## Structural and Functional Gates

Before H20 timing, use disposable tests and require:

1. Exactly two nonidentity shortcuts at `layer2[0]` and `layer3[0]`. Each is
   exactly `AvgPool2d -> ShortcutConv -> BatchNorm2d`; every other shortcut is
   `nn.Identity`.
2. Pool kernel/stride are `(2,2)`, padding zero, `ceil_mode=False`, and
   `count_include_pad=False`; projection kernel/stride are `(1,1)`, padding zero,
   bias absent; BN features are 64 and 128 with accepted defaults.
3. The shape table above holds for seeded tensors. A coordinate-ramp test proves
   each pooled output equals the exact arithmetic mean of its disjoint `2x2`
   input cell and uses no neighboring/padded value.
4. Hook both branches and prove every residual-branch tensor, operation, and
   shape is accepted. In particular, both transition residual `conv1` modules
   remain stride 2 and no pool touches them.
5. Exact candidate parameter count 1,084,586; all new parameters appear once in
   the single SGD group; there are exactly two new BN running-stat sets and no
   stale Option-A pad path.
6. From cloned seed-42 state, all shared learned tensors are bitwise equal and
   post-construction CPU RNG is byte-identical. New projection tensors are
   finite/nonidentical and every shortcut BN begins gamma one/beta zero with
   default running state.
7. Hard `[128]` and probability `[128,10]` targets both produce finite logits,
   losses, gradients, and one-step parameters; every new conv/BN affine tensor
   receives finite data gradient and pool input receives finite nonzero gradient.
8. Compile, Ruff, formatting, pre-commit, and diff checks pass. Only `train.py`
   changes; accepted data, optimizer, schedule, timer, evaluator, worker, seed,
   and logging text outside model plumbing remains unchanged.

## Real-Batch Initialization and Update-Scale Gates

The new shortcut is active at full gamma from step one and is not output-aligned
with Option A. Safety must therefore be measured on production-distribution
inputs, following the EXP-015/016 protocol finding.

Materialize one identical N1/M7 sequence containing both hard and CutMix
probability batches. Construct aligned accepted/candidate models and ordinary
optimizers from cloned seed-42 states. Before update one, record residual,
accepted-shortcut, projected-shortcut, pre-add, and logit RMS/norm at both
transitions. Require every value finite and each projected-shortcut RMS between
0.25x and 4.0x its paired residual RMS. This is a catastrophic-scale gate, not a
claim that equal RMS is optimal.

On both a fresh hard batch and a fresh probability batch:

1. Require finite nonzero gradients in both projection convs and BN affine
   tensors. Each projection's ordinary SGD update norm must be no more than 25%
   of its pre-update parameter norm.
2. Replay the same batch after update one. Candidate loss must be no more than
   2x its own pre-update loss and no more than 2x aligned-control replay loss.
3. Candidate predictions must not put more than 95% of examples in one class
   unless the aligned control is at least as concentrated; logits and transition
   activations remain finite.
4. Both shortcut BN running variances remain positive/finite and both projection
   weights retain finite nonzero norms.

Then train aligned control and candidate separately over the same 200 distinct
materialized real strong batches. Require finite trajectories, no candidate-only
class concentration above 95%, and candidate terminal loss EMA no more than
1.5x control. The 200-batch check matches the precision-candidate horizon that
found EXP-016's failure; it remains only a broad safety gate. EXP-015 showed that
favorable short fit does not predict the complete strong phase, so no result may
be selected or tuned from this diagnostic.

If any safety gate fails, do not launch and do not change initialization, BN
gamma, pool order, shortcut LR, or projection form inside EXP-017.

## Mandatory Paired H20 Timing and Exposure Gates

On the single idle NVIDIA H20 near 97,871 MiB, benchmark the exact reviewed
candidate against accepted `7c1e7d8` in five alternating fresh-process pairs.
Use cloned aligned shared weights, separate ordinary SGD state, batch 128 pinned
host inputs, alternating hard/probability targets, 100 warmup steps, and 500
measured complete production-region steps: H2D, zero-grad, forward, CE,
backward, SGD, and terminal synchronization.

Record per-trial mean, median, p95, CV, projected steps, and peak allocation.
Require all of these:

- candidate/control median trial-mean training ratio at most **1.055**;
- `floor(26_898 * control_mean / candidate_mean) >= 25_500`, retaining at least
  94.8% of EXP-010 updates;
- candidate p95 at most 1.10x control p95 and trial-mean CV below 3% for both;
- candidate peak allocation below 700 MB and no more than 96 MB above control;
- every timed loss and parameter finite; and
- separate five-pair eval-mode inference ratio at most 1.08, CV below 3%, with
  conservative projected total runtime below 540 seconds.

The threshold allows measured projection/pool/BN launch overhead but preserves
enough exposure for a representation mechanism expected to produce a modest
gain. At 25,500 updates, the candidate completes about 65.4 epochs and 13.1
weak-tail passes, implying roughly 18-19 unique evaluations versus EXP-010's
19. More than 19 evaluations is invalid because it grants extra best-metric
opportunities. If any timing, memory, or wall gate fails, do not launch and do
not fall back to a strided projection, remove BN/pooling, change precision, or
compensate with LR.

## Testable Hypothesis

**Primary hypothesis:** average-pooling all four positions in each transition
cell and learning normalized transport into every expanded output channel will
improve stage-boundary information retention enough to raise `best_test_acc`
from 94.15% to at least **94.25%**, while retaining at least 25,500 projected
optimizer updates and preserving active accepted residual learning.

A plausible successful range is 94.25-94.50%. The lower end reflects that only
two shallow-network shortcuts change and ResNet-D evidence is indirect; the
upper end allows learned channel transport to complement width-2 capacity and
CutMix's regional class evidence. A larger prediction is not justified.

Mechanism diagnostics are:

- 80% clean checkpoint above the 87.08% underfit marker and preferably near
  EXP-010's 89.73%;
- first weak checkpoint near or above 93.16%;
- final NLL near or below 0.1934 and a nondeclining terminal trajectory;
- 25.5k-26.8k steps, 45-55% strong CutMix, one clean worker switch, and no more
  than 19 unique evaluations;
- finite shortcut BN statistics through both phases.

Only `best_test_acc >=94.25%` with all formal integrity conditions is an
improvement. Better shortcut scale, fit, NLL, or exposure cannot override a
missed primary threshold.

## Risks and Failure Interpretation

- **Loss of exact identity transport.** Option A preserves selected old channels
  without parameters. Pool/projection/BN makes both transitions learned and can
  weaken direct gradient flow even while it supplies all output channels.
- **Random full-scale shortcut.** Kaiming projection plus gamma-one BN changes
  transition sums immediately. It may dominate or destructively interfere with
  the accepted residual branch despite broad RMS/update safety gates.
- **Strong-phase underfit.** Pooling removes within-cell high frequencies, and
  randomized shortcut bases can slow fitting under N1/M7 plus CutMix. A switch
  below 87.08% identifies the recurring local failure signature.
- **CutMix area dilution.** A `2x2` average can blend donor/source features at a
  patch boundary. Small donor regions may contribute less discriminative signal
  than their area target expects.
- **BN phase shift.** Shortcut running statistics are learned mostly under
  composite strong views and have only about thirteen weak passes to adapt.
  A healthy switch followed by a poor first weak checkpoint implicates shortcut
  BN adaptation.
- **Projection bypass/overfit.** Learned `1x1` shortcuts can carry class-specific
  signal around residual processing. The 10,624 extra parameters may overfit or
  reduce the benefit of residual refinement under unchanged decay.
- **Pooling is not lossless.** It uses all four positions but retains only their
  mean, discarding within-cell arrangement. "Anti-loss" is relative to raw
  single-phase decimation, not an invertibility claim.
- **Residual path remains aliased.** This candidate is deliberately shortcut-only
  ResNet-D semantics. A result cannot establish full-network shift invariance or
  test the unrun full-path BlurPool proposal.
- **Kernel-launch overhead.** Projection MACs are small, but pool, BN, and their
  backwards add sequential kernels. Fewer fixed-budget updates can erase a
  modest representation gain.
- **Literature transfer.** ImageNet bottleneck ResNet-D and a CIFAR downsampling
  search do not directly estimate this BasicBlock operating point.
- **Single-seed threshold.** A bare 94.25-94.35 pass is valid but weak evidence.
  Never reroll, retune initialization, or run a fallback variant.

## One-Run Verification and Decision Rule

After structural, real-batch safety, paired timing, exposure, memory, and wall
gates all pass, run exactly once on the confirmed sole idle H20:

```bash
CUDA_VISIBLE_DEVICES=0 timeout 600s uv run train.py > run.log 2>&1
```

Require exit zero, finite summary fields, approximately 300 counted seconds,
total below 600 seconds, exactly 1,084,586 parameters, one augmentation switch
near 80%, eight stopped workers, correct hard/soft target provenance, unique
evaluation epochs, no more than one evaluation per epoch and no more than 19
total evaluations, seed 42, and only the reviewed `train.py` diff. Never rerun a
mechanically valid result.

The prelaunch timing gate requires at least 25,500 projected steps. If the valid
production run lands below that because of ordinary jitter, keep its accuracy
verdict but report weaker exposure attribution; do not alter LR, budget, or
evaluation cadence after observation.

- **Improvement:** accept only if `best_test_acc >=94.25%` and every formal goal
  condition passes.
- **Valid no-improvement:** revert the learned shortcuts after a correct result
  below 94.25%; use strong fit, first weak accuracy, BN state, NLL, exposure, and
  trajectory to explain the mechanism.
- **Mechanical failure:** repair only an implementation/environment defect that
  leaves exact pooling, projection, BN, initialization, and scope unchanged.
  Any alternate shortcut, initialization, optimizer, precision, or combined
  mechanism requires a newly reviewed experiment.

## Attribution

The intervention bundles the fixed average-pool downsampler, learned `1x1`
channel projection, and BN because together they define the ResNet-D shortcut.
The accepted residual path and all shared initial state/RNG are preserved, and
new parameter/cost effects are measured explicitly. The full run therefore
estimates the net value of learned anti-loss transition transport under the
fixed EXP-010 protocol.

It cannot separate pooling from projection or BN, prove shift invariance, choose
an initialization optimum, or support any fallback shortcut. Those questions
are outside EXP-017.
