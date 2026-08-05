# Proposal: Neutral Stage-3 Squeeze-and-Excitation

## Recommendation

Add one lightweight squeeze-and-excitation gate to the residual branch of each
of the two existing stage-3 `PreActBlock`s. Each gate observes the block's
128-channel `conv2` output at 8x8, compresses its globally pooled channel vector
to eight hidden units (reduction ratio 16), and rescales the residual branch
immediately before the unchanged shortcut addition. Initialize the controller
to an exactly neutral scale of one with `2 * sigmoid`, so the candidate begins
with the accepted forward function and accepted residual-branch magnitude.

Keep stage widths `[32,64,128]`, stage depths `[2,2,2]`, all convolutions,
shortcuts, final normalization/classifier, FP32 SGD, time-based LR schedule,
alpha-0.2 batch-shared mixup through 65%, hard-label tail, seed, data path, and
evaluation cadence unchanged. This is one fixed attention treatment, not a
search over stages, reduction ratios, gate offsets, or placements.

## Diagnosis And Rationale

The accepted WRN-16-2 fully fits the training data but stops at 94.07%, so raw
optimization exposure is not the apparent limiter. BF16 supplied 12.1% more
passes and regressed. In contrast, two dense stage-3 capacity probes were the
only architecture changes to move positively: 160 stage-3 channels reached
94.11%, and an extra full 128-channel 8x8 block reached 94.15%. Their 132-pass
exposure was adequate, but neither crossed 94.17%; the extra block also raised
test loss to 0.2782. A half-width post-stage-3 bottleneck then fell to 93.74%,
showing that merely appending a compressed residual transform does not retain
the dense capacity signal.

This motivates feature selection rather than another raw transform. At the
8x8 semantic stage, global channel context can modulate which already learned
residual features should alter the shortcut representation for each example.
The gate adds negligible spatial arithmetic and only 0.63% parameters, so it
should preserve essentially all accepted exposure. Limiting attention to
stage 3 follows the only positive local architecture evidence and avoids
introducing six controllers across high-resolution stages whose usefulness and
kernel overhead are unsupported.

Use both existing stage-3 blocks, not only the final block. The first block is
the 64-to-128, stride-2 transition where channels acquire their final semantic
basis; the second refines that basis. Gating both residual branches permits
selection at both operations while leaving both shortcuts untouched. Applying
the gate after the residual addition would also rescale the identity path and
would no longer preserve the residual architecture's clean information route.

## Exact Topology And Placement

For each existing block in `layer3`, retain its accepted computation through
the second convolution:

```text
preactivated = ReLU(BN1(x))
shortcut = projection(preactivated) if required else x
r = Conv1(preactivated)
r = Conv2(ReLU(BN2(r)))                  # [N,128,8,8]
s = AdaptiveAvgPool2d(1)(r).flatten(1)  # [N,128]
h = ReLU(Linear(128,8,bias=True)(s))    # reduction ratio 16
g = 2 * sigmoid(Linear(8,128,bias=True)(h))
output = r * g[:, :, None, None] + shortcut
```

The gate consumes the signed `conv2` output, before the shortcut addition and
without an extra BN or activation. The first stage-3 block's accepted
64-to-128 projection shortcut remains unchanged and ungated. The second
block's literal identity shortcut remains unchanged and ungated. There is no
gate in the stem, stage 1, stage 2, final BN, pooled representation, or
classifier, and no post-add activation is introduced.

Implement the gate as a small dedicated `Stage3SE` module. `PreActBlock` may
hold `attention = None` and apply it only when present. Attach exactly one gate
to `layer3[0]` and one to `layer3[1]`; do not generalize the scored topology to
an arbitrary per-block policy. A strict Boolean constructor switch may be used
solely so the evaluator-free preflight can instantiate exact accepted and
candidate models from the same code; reject non-Boolean values and instantiate
the scored model with the switch enabled.

## Reduction, Scale, And Initialization

Choose reduction ratio **16**, giving hidden width `128 // 16 = 8`. This is the
smallest conventional SE bottleneck that still gives every output channel a
learned function of all 128 pooled channels. Ratio 8 would double controller
parameters without evidence, while ratio 32 leaves only four hidden features
and risks an overly coarse rank constraint. Do not adapt the ratio to timing or
interim results.

Initialize each gate exactly as follows:

- first linear weight: Kaiming normal for ReLU;
- first linear bias: zero;
- second linear weight: zero;
- second linear bias: zero;
- output scale: exactly `2.0 * torch.sigmoid(logits)` with no epsilon, clamp,
  temperature, learnable scalar, or detached path.

At construction, the second projection emits zero, so every gate value is
exactly one because `2 * sigmoid(0) = 1`. The complete candidate therefore has
the accepted forward function and branch magnitude at step zero. The second
projection receives a nonzero first-step gradient through the fixed derivative
of the scaled sigmoid; the first projection correctly has zero first-step
gradient and begins learning after the second projection opens the controller.
This is deliberate staged activation, not a frozen gate.

The neutral controller is materially different from EXP-014's failed zeroed
residual endpoints. EXP-014 suppressed all six residual branches at startup.
Here every accepted convolutional branch remains fully active and identical;
only a newly added multiplier starts at identity. Do not initialize the gate
to ordinary sigmoid 0.5, because that would halve both stage-3 residual
branches and confound attention with a large residual-scale intervention.

Construct and initialize the accepted WRN first using its existing
`self.apply(self._weights_init)`. Then create/initialize the two attention
modules inside a CPU `torch.random.fork_rng(devices=[])` scope and attach them
to the two stage-3 blocks. Restoring the CPU RNG state is required: all
accepted parameters must be bitwise identical to an attention-disabled model
under the same seed, and the global RNG state after model construction must be
identical so the subsequent DataLoader and training RNG trajectory are not
shifted merely by creating the controllers. Do not alter CUDA RNG state during
CPU construction.

The four matrix weights are included in the existing `ndim >= 2` optimizer
group with `5e-4` coupled weight decay. The four biases are included in the
existing no-decay group. Add no custom optimizer group, LR multiplier, gate
regularizer, entropy loss, or clamp.

## Exact Parameter And Compute Cost

Each gate has:

| Component | Arithmetic | Parameters |
|---|---:|---:|
| `Linear 128->8` weight | `128 * 8` | 1,024 |
| `Linear 128->8` bias | `8` | 8 |
| `Linear 8->128` weight | `8 * 128` | 1,024 |
| `Linear 8->128` bias | `128` | 128 |
| **Per gate** | | **2,184** |
| **Two gates** | | **4,368** |

The accepted model has 691,674 trainable parameters. The candidate must have
exactly **696,042**, an increase of **0.6315%**. Adaptive pooling, ReLU, sigmoid,
and channel multiplication have no trainable parameters or persistent buffers.

Conventional multiply-accumulate accounting gives 2,048 FC MACs per gate and
4,096 FC MACs for both gates. Rescaling two `[128,8,8]` residual tensors adds
16,384 elementwise multiplications per image. Thus the controllers add 20,480
multiply/MAC-like operations per image before pooling/relu/sigmoid overhead.
Against the accepted model's 101,106,944 convolution/linear MACs per image,
the combined arithmetic increase is about **0.0203%**; conventional
conv/linear-only MACs rise to 101,111,040, about **0.0041%**. Global reductions
and several small CUDA kernels make measured step latency more important than
these static ratios.

## Predicted Impact

Predict `best_test_acc` in **94.15-94.32%**, centered near **94.22%**, while
retaining at least 95% of accepted matched throughput. The upside comes from
input-conditioned selection of dense 128-channel features without the
exposure and confidence cost of another full transform. The lower end admits
that the accepted representation may already encode channel importance and
that a near-neutral gate can have too little effect within 300 seconds.

The formal success threshold remains `best_test_acc >= 94.17%`: the accepted
94.07% plus the required 0.10 percentage-point margin. Run one fixed-seed
scored experiment only. Do not rerun a near miss or follow it within EXP-016
with ratio 8, final-block-only attention, ordinary sigmoid scaling, or a gate
on the post-add output.

## Evaluator-Free Semantic Preflight

Run all preflight checks in a fresh process with `prepare.Eval` replaced before
import by a fail-closed dummy. Use synthetic training-shaped inputs only and
never construct or call the test evaluator.

Require all of the following before any scored run:

- attention-disabled and enabled models contain exactly 691,674 and 696,042
  trainable parameters;
- exactly two `Stage3SE` modules exist, attached only to `layer3[0]` and
  `layer3[1]`, each with `128->8->128` biased linear projections;
- no accepted convolution, BN, shortcut, stage depth/width, final BN, or
  classifier shape changes; candidate logits are finite FP32 `[256,10]`;
- every second-projection weight and bias is exactly zero, every first bias is
  zero, first weights are finite/nonzero, and both gates emit bitwise ones for
  finite synthetic residual inputs at initialization;
- under a reset construction seed, every accepted named parameter and buffer
  is bitwise identical between disabled and enabled models, and serialized CPU
  RNG states after construction are exactly equal;
- in `eval()` mode, matched models produce bitwise-identical initial logits on
  the same input; module hooks confirm each gate sees and returns
  `[N,128,8,8]` and neither shortcut is passed through a gate;
- in `train()` mode from cloned states, one matched hard-label and one matched
  scalar-mixup forward/backward produce identical initial logits, losses, all
  accepted-parameter gradients, accepted BN buffer updates, and accepted
  parameter updates within exact equality where PyTorch is deterministic;
- for each gate, the first step gives finite nonzero gradient to the second
  projection while the first-projection gradient is exactly zero; after an SGD
  step, at least one gate value differs from one, and a second backward gives a
  finite nonzero first-projection gradient;
- a manually perturbed second-projection row changes the corresponding branch
  channel scale but never the shortcut tensor, proving the gate is operational
  and placed before addition;
- strict constructor validation rejects Boolean impostors/integers as
  appropriate, wrong stage counts, or any accidental reduction that does not
  yield hidden width eight.

The accepted-parameter first-step equality is important. It distinguishes an
identity-initialized controller from an implementation that silently halves
the residual branch, perturbs accepted initialization, consumes the later RNG
stream, gates the shortcut, or changes BN behavior.

## Matched Throughput Preflight

Benchmark exact attention-disabled and enabled production steps on the single
H20. Use cloned accepted model/optimizer states, pinned host inputs,
nonblocking transfers, the accepted LR writes, beta sampling and permutation,
mixup construction/loss, hard-label path, finite guard, backward, SGD/Nesterov
step, and final CUDA synchronization. Preserve the candidate's real gate
forward/backward; do not freeze it or replace it with identity during timing.

Warm each path for at least 25 steps. Measure three continuing 50-step windows
for both 50%-progress mixup and 80%-progress hard-label regimes in a balanced
interleaved order, restoring each path's private training RNG stream around
windows as in EXP-010/011. Define each regime center as the median window mean,
require population CV no greater than 5%, and combine medians with the
preregistered `0.65 * mixup + 0.35 * hard-label` weights.

Define retention as `accepted_aggregate_ms / candidate_aggregate_ms` and
projected passes as `141.9 * retention`. Expect at least 98% retention from the
static cost, but launch scoring only if measured retention is at least **95%**
and projected exposure is at least **134.8 passes**. Also require finite loss,
correct logits/counts, peak memory well below device capacity, one H20, and no
evaluator access. The conservative 95% gate allows small-kernel launch and
autograd overhead but prevents attention whose operational cost erases the
accepted exposure. Do not alter the ratio or placement if the gate fails.

## Full-Run Verification

After a passing preflight, remove stale `run.log` and execute exactly once:

```bash
timeout 600s uv run train.py > run.log 2>&1
```

Require exit code zero, one NVIDIA H20, exact 696,042 parameter count, complete
finite final summary, about 300 counted training seconds, total time below 600
seconds, and at most one evaluation per epoch. Confirm batch-shared alpha-0.2
mixup disables exactly once near 195 counted seconds, the final 35% uses the
unchanged hard-label path, and no attention-specific schedule or log-time
synchronization was introduced. Record best/final accuracy and loss, steps,
epochs, peak VRAM, transition time/step, realized passes, and best/final gap.

Accuracy is authoritative. A valid score below 94.17% is no-improvement. If
realized exposure falls below 134.8 passes despite a stable passing preflight,
report the operational discrepancy, but do not rerun or modify the controller.

## Risks And Interpretation

- **Attention may be redundant:** the two dense stage-3 blocks and final BN may
  already learn adequate channel selection; a neutral gate might add no useful
  boundary change within the budget.
- **Extra specialization may worsen confidence:** EXP-011 gained top-1 but
  worsened loss. Input-conditioned amplification up to two can similarly make
  already confident errors worse despite minimal parameter cost.
- **Neutral initialization can learn slowly:** zeroing the second projection
  delays first-projection learning by one step. That preserves attribution but
  leaves the controller dependent on second-layer opening dynamics.
- **Global pooling discards spatial evidence:** CIFAR features at 8x8 may need
  spatially varying modulation; this experiment tests channel attention only.
- **Mixed-image gates see mixed semantics:** during the first 65%, pooled
  representations correspond to mixup inputs. This is part of the treatment;
  no clean-image side path or detached gate target may be added.
- **Small kernels can cost disproportionate time:** eight tiny linear kernels
  across forward/backward and pooling/sigmoid launches may exceed the static
  0.02% arithmetic estimate. The matched preflight guards this.
- **The doubled sigmoid can amplify residuals:** its `[0,2]` range is required
  for an identity start but permits amplification. Clamping or residualizing
  the gate would be a different treatment.

A stable run below 94.17% with normal exposure rejects only exact two-block,
stage-3, ratio-16, neutral `2*sigmoid` SE as a sufficient standalone change. It
does not establish that all channel attention fails, but no neighboring gate
variant may be used as an adaptive rescue in this experiment.

## Falsifiable Hypothesis

If the positive EXP-010/011 signal reflects a need to select and recombine
dense low-resolution features rather than to append more raw transformation,
then exactly neutral ratio-16 SE gates on both existing stage-3 residual
branches will retain at least 95% matched throughput, project at least 134.8
passes, and raise one fixed-seed 300-second run from 94.07% to
`best_test_acc >= 94.17%`.

Failure of semantic/RNG/throughput gates rejects the implementation without a
scored fallback. A valid scored result below 94.17% falsifies this exact
attention allocation without a ratio, scale, initialization, or placement
retry.

## Evidence

- `knowledge/papers/wide-residual-networks.md`: CIFAR residual networks benefit
  from allocating capacity to width rather than extreme depth; the proposal
  preserves the accepted wide basic transforms and adds only feature selection.
- `experiments/010/04-analysis.md`: selective 160-channel stage-3 width moved
  accuracy to 94.11% at 132.16 passes, the first post-acceptance positive delta.
- `experiments/011/04-analysis.md`: one dense stage-3 block reached 94.15% but
  increased final test loss to 0.2782, motivating selection rather than more
  unconditioned transformation.
- `experiments/012/04-analysis.md`: an efficient rank-64 post-stage-3 residual
  transform retained 135.49 passes but fell to 93.74%, so low arithmetic alone
  is not enough and a different mechanism is required.
- `03-experiment-learnings.md` and `04-results.tsv`: more exposure, additive
  regularization, averaging, endpoint initialization, and coefficient
  decorrelation all failed; low-resolution dense capacity remains the clearest
  positive local architecture signal.
- `train.py`: the accepted two-block 128-channel stage 3, selective optimizer
  groups, batch-shared temporal mixup, and time budget define the exact
  integration and preflight semantics above.
