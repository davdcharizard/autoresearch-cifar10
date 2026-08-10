# Idea: Same-Width Residual Identity Initialization

## Summary

Keep the accepted EXP-010 width-2 postactivation ResNet-20 and initialize the
final BatchNorm scale (`bn2.weight`) to exactly zero only in the six ordinary
within-stage blocks: blocks 2 and 3 of each of the three stages. Leave all three
stage-entry blocks at the accepted BatchNorm scale of one.

The three retained stage-entry blocks are:

| Block | Shape transition | Shortcut | Initial `bn2.weight` |
|---|---|---|---:|
| `layer1[0]` | `32 -> 32`, stride 1 | exact identity | 1 |
| `layer2[0]` | `32 -> 64`, stride 2 | Option-A slice + 32 zero channels | 1 |
| `layer3[0]` | `64 -> 128`, stride 2 | Option-A slice + 64 zero channels | 1 |

Only the latter two change tensor shape and use padding, but all three are kept
as accepted stage-entry units to make the scope explicit and symmetric. The six
zero-gamma units all have stride one, equal input/output channels, and exact
identity shortcuts. No stem, transition, classifier, topology, forward
operation, parameter group, or training mechanic changes.

This is a compute-neutral initialization experiment. It tests whether beginning
the ordinary residual units as exact forward identities improves early
conditioning and final generalization without canonical preactivation's global
reorder or the structural dead channels created by zeroing the two padded
Option-A transition branches.

## Diagnosis

EXP-010 remains the accepted 94.15% frontier. Its width-2 postactivation model,
all-parameter `1e-4` decay, p=0.5 alpha-1 CutMix on N1/M7 strong views through
80%, and hard weak tail completed 26,898 updates. The switch checkpoint was a
healthy 89.73%, the first weak checkpoint was 93.16%, and final/best accuracy was
94.15% with 0.1934 NLL.

The local failures narrow the intervention:

- EXP-011's stronger CutMix and EXP-012's full preactivation both pushed the
  strong checkpoint below the 87.08% underfit marker. A candidate should not add
  augmentation pressure or reorder every residual path.
- EXP-012 was compute-neutral and reached 94.22%, but its complete preactivation
  package lowered strong fit by 2.85 points. Its near miss leaves a narrower
  identity-initialization mechanism untested on the accepted topology.
- EXP-013 rejected batch scaling before an accuracy run because fresh paired
  timing found only 18.91% more image throughput. More images at fewer updates
  is not a sufficiently strong current lever.
- EXP-014 showed that exact zero initial output does not ensure optimization
  continuity when a new path has a 4.10x classifier-gradient scale. Any zero
  initialization now needs first-update mechanics and a collapse gate, not only
  an initial-function proof.

The systems profile attributes 97.6% of timed step work to the existing forward
and backward graph, with backward alone at 75.46%. This candidate adds no op and
cannot improve systems throughput materially; its value must come from the
optimization trajectory while retaining essentially all accepted exposure.

## Primary Evidence and Transfer Limits

Goyal et al., *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour*,
report that zero-initializing the last BN scale in each postactivation ResNet-50
residual branch reduced ImageNet top-1 error from `23.84 +/- 0.18` to
`23.60 +/- 0.12`. Their proposed mechanism is that every residual block starts
as an identity, easing optimization at initialization. The local block has the
same critical placement: `bn2` follows the final residual convolution and feeds
the addition directly, so its scale receives gradient even when its output is
zero.

The published result is directional evidence, not a portable effect estimate.
It used a deeper bottleneck network, ImageNet, projection shortcuts at dimension
changes, a much larger batch regime, and a different schedule. In particular,
projection shortcuts give every transition output channel a live shortcut
value. The local CIFAR model uses zero-padded Option A, so blindly applying the
paper's all-block rule has different and unsafe channel semantics.

This proposal intentionally narrows the literature technique to the six blocks
where a true same-shape identity exists. A success supports scoped identity
initialization on this accepted network; a loss does not establish that
zero-gamma is universally harmful or that the ImageNet result is wrong.

Primary source: P. Goyal et al., [Accurate, Large Minibatch SGD: Training
ImageNet in 1 Hour](https://arxiv.org/pdf/1706.02677), Section 5.1 and Table 2b.

## Why Not Zero All Nine Blocks

For an ordinary postactivation block, the candidate's first forward is:

```text
residual = gamma * normalized(conv2(...)) + beta = 0
output = ReLU(x + residual) = ReLU(x) = x
```

The equality holds because every ordinary block receives the nonnegative output
of the preceding post-add ReLU. It is an exact forward identity, apart from the
already-zero coordinates whose value remains zero.

The two dimension-changing blocks do not have an identity shortcut. Option A
slices the input spatially, copies the old channels, and appends zero channels.
If their final BN scale and bias were both zero, the newly introduced high
channel half would enter the post-add ReLU at exactly zero. PyTorch's ReLU
derivative at zero is zero, so those padded channels supply no gradient to
`bn2.weight`, `bn2.bias`, or either residual convolution. With zero scale and
bias, weight decay cannot wake them. The 32 new stage-2 channels and 64 new
stage-3 channels can remain permanently dead, effectively defeating width 2.

The retired all-nine-block form therefore has a structural failure mechanism,
not merely a stronger version of this candidate. Retaining gamma one in both
padded transitions keeps every new output channel active and trainable from the
first backward pass. `layer1[0]` has no padding and could safely be zeroed in
isolation, but this proposal leaves all three stage-entry blocks accepted so it
tests one exact, predeclared family: the two ordinary blocks after each stage has
established its representation.

Do not rescue an all-block implementation by setting a positive BN bias,
changing ReLU's derivative, replacing Option A, or zeroing only selected output
channels. Those are different architectures and initialization rules.

## Exact Implementation

Add an explicit initialization flag to `BasicBlock` and pass it only to
non-entry blocks. One acceptable implementation is:

```python
class BasicBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, stride=1, zero_init_residual=False
    ):
        super().__init__()
        # Existing conv1, bn1, conv2, and bn2 construction stays unchanged.
        if zero_init_residual:
            init.zeros_(self.bn2.weight)
```

In `_make_layer`, derive the flag from both location and shape rather than a
global module-type pass:

```python
for block_index, s in enumerate(strides):
    zero_init_residual = block_index > 0 and s == 1 and ch == out_ch
    layers.append(
        BasicBlock(ch, out_ch, s, zero_init_residual=zero_init_residual)
    )
    ch = out_ch
```

This yields exactly these zero tensors:

```text
layer1.1.bn2.weight  [32]
layer1.2.bn2.weight  [32]
layer2.1.bn2.weight  [64]
layer2.2.bn2.weight  [64]
layer3.1.bn2.weight  [128]
layer3.2.bn2.weight  [128]
```

Here the numeric suffix is the zero-based `nn.Sequential` index. Exactly six
tensors and 448 scalar gamma values become zero. The three `.0.bn2.weight`
tensors remain all ones. Every `bn2.bias` remains the accepted zero default.

The existing `self.apply(self._weights_init)` touches only convolution and
linear weights, so it does not overwrite the selective BN initialization. Do
not broaden `_weights_init` to every BN, add a second model-wide pass based only
on module type, or infer the transition exception from channel number alone.

## RNG, State, and Optimizer Semantics

`init.zeros_` is deterministic and consumes no random values. No module or
parameter is added, removed, or reordered. Given cloned seed-42 CPU RNG state,
constructing accepted and candidate models must leave identical post-construction
CPU RNG states. Their state dictionaries must be bitwise identical except for
the six declared `bn2.weight` tensors, which are one in control and zero in the
candidate. CUDA RNG state must remain unaffected by CPU model construction.

The parameter count remains exactly **1,073,962**, all state-dict keys and tensor
shapes remain unchanged, and serialized checkpoints remain structurally
compatible. BatchNorm running means start at zero, running variances at one, and
`num_batches_tracked` at zero in every block. The zero-gamma branches still
execute their convolutions and BN on every forward, so their running statistics
update normally from the first strong batch even while their output is gated.

All parameters remain in the accepted single SGD group with LR 0.1, momentum
0.9, and coupled weight decay `1e-4`. Do not exempt zero gammas or residual
weights from decay. Weight decay has no first-step effect on gamma or BN bias
while they are zero, but it does act on the existing convolution and `bn1`
weights even when their first loss gradient is zero.

Because post-construction RNG is identical, the data sampler, worker seeds, and
CutMix RNG begin from the accepted seed-42 stream. Model outputs immediately
diverge by design, but no random-number realignment, new seed, or reroll is
needed or allowed.

## First-Step Gradient Mechanics

For an ordinary zero-gamma block, write the preactivation before the final ReLU
as:

```text
z = x + gamma * n + beta
```

where `n` is the normalized `conv2` output. On the first backward pass:

- `dL/dgamma = sum(dL/dz * n)` is generally finite and nonzero, so gamma learns
  immediately even though the residual output starts at zero;
- `dL/dbeta = sum(dL/dz)` is also generally nonzero;
- loss gradients to `conv2`, `bn1`, and `conv1` are multiplied by gamma and are
  exactly zero before the first optimizer step;
- BatchNorm running statistics still update because the full branch forward is
  executed;
- the shortcut carries the upstream gradient for positive input coordinates;
  at exact-zero coordinates PyTorch's post-add ReLU derivative is zero, so the
  backward map is not a strict identity everywhere even though the forward is.

During the first `optimizer.step()`, gamma and usually beta depart from zero.
The branch convolutions receive only their coupled weight-decay update because
their data gradient is zero. On the second forward/backward, nonzero gamma and
beta expose the branch, and both residual convolutions should receive finite
nonzero loss gradients. This is a one-update recruitment delay, not a dead
branch.

The stage-entry blocks remain ordinary accepted residual units: their gammas are
one and both convolutions receive data gradients on the first step. For the two
padded transitions, explicitly verify nonzero activations and gradients in the
new high channel halves. That check distinguishes this candidate from the
structurally dead all-block variant.

## Difference from Earlier Near Misses

This is not EXP-012 preactivation. Forward ordering remains
`Conv-BN-ReLU-Conv-BN-add-ReLU`; all nine post-add ReLUs, the stem BN-ReLU,
Option-A shortcut sources, and final pooling stay accepted. Zeroing `bn2` is
safe here because it is the final operation before addition, so gamma sees the
addition's gradient. In canonical full preactivation, the candidate BN would be
before a ReLU and `conv2`; zeroing it makes ReLU's derivative zero and can leave
the complete branch permanently dead.

This is also not EXP-014's failed zero-initialized max readout. EXP-014 created a
new unnormalized global-max statistic whose first classifier gradient was 4.10x
the accepted average path and whose first update overwhelmed the logits. The
present candidate adds no feature statistic or parameter. Each gate multiplies
an existing normalized residual signal, and its first-update scale is checked
directly. Exact initial identity still does not guarantee safety, which is why
the proposal includes post-update logit/loss and short-fit gates.

## Preserved Accepted Recipe

Outside the six initialization values and minimal flag plumbing, keep accepted
`train.py` unchanged:

- width multiplier 2, channels 32/64/128, three blocks per stage, all convolution
  shapes, Option-A shortcuts, postactivation ordering, adaptive average pooling,
  and the 128-to-10 classifier;
- Kaiming-normal Conv/Linear initialization and all other BatchNorm defaults;
- batch 128, hard or probability-target cross-entropy, ordinary SGD, momentum
  0.9, all-parameter weight decay `1e-4`, and no Nesterov;
- crop/flip plus N1/M7 RandAugment and alpha-1 CutMix at probability 0.5 during
  the first 80% of counted time, then the exact hard weak tail;
- the worker-side RNG fork, target counters, eight persistent workers, explicit
  shutdown, garbage collection, and loader rebuild;
- LR 0.1 through 80%, step to 0.01, cosine to `1e-4`, 64,000 max steps, seed 42,
  and the same timer boundaries;
- checkpoints `(0.2, 0.4, 0.6, 0.7)`, at most one evaluation per epoch, fixed
  evaluator, ten-minute timeout, and summary schema.

Do not combine zero-gamma with preactivation, projection shortcuts, altered
BatchNorm momentum/epsilon, stochastic depth, a gamma-specific LR, no-decay
groups, warmup, compilation, larger batches, or another regularizer.

## Structural and State Verification

Before GPU timing, use disposable tests and require:

1. Exactly nine `BasicBlock`s, six zero `bn2.weight` tensors with the exact names
   above, and three all-one stage-entry `bn2.weight` tensors.
2. Every zero-gamma block has stride one, equal input/output channels,
   `need_pad=False`, and an exact identity shortcut. Conversely, no `.0` block,
   stride-2 block, or padded block has a zero gamma.
3. Exactly 1,073,962 trainable parameters, unchanged state-dict keys/shapes, and
   unchanged membership in one SGD parameter group.
4. From cloned seed-42 state, bitwise equality for all candidate/control state
   except the six gamma tensors and byte-identical post-construction CPU RNG.
5. For seeded nonnegative inputs in train and eval modes, each zero-gamma block's
   initial output equals its input exactly. The three retained entry blocks must
   match an aligned accepted reference exactly.
6. Both hard `[128]` and CutMix probability `[128, 10]` targets produce finite
   logits and losses without changing the existing loss function.
7. The only tracked diff is the selective initialization plumbing in `train.py`;
   optimizer, schedule, data, timer, evaluator, and logging diffs are forbidden.

## Gradient and First-Update Safety Gates

Against an aligned accepted reference and identical fixed hard and probability
batches, check the candidate before any full run:

1. Before the first step, require exactly zero residual output from all six
   gated branches and finite, nonzero transition residual outputs, including the
   padded high-channel halves.
2. After the first backward and before SGD, require each of the six gated
   gamma-gradient tensors to have finite nonzero norm, exactly zero loss
   gradients for their `conv1`, `bn1`, and `conv2` parameters, and finite
   nonzero data gradients in all three retained entry blocks.
3. After the first optimizer step, require every gated gamma tensor to have
   nonzero norm, the global `max(abs(gamma)) <= 0.25`, finite parameters, and
   finite gamma-one transition parameters.
4. Replay the same batch after that update. Require candidate loss no more than
   2x its pre-update loss and no more than 2x the aligned control's replay loss;
   require finite logits and no single predicted class above 95% unless the
   aligned control is at least as concentrated.
5. On the second backward, require finite nonzero loss gradients in `conv1` and
   `conv2` of every gated block, proving all six branches recruited. Require the
   new high-channel halves at both transitions to remain active and trainable.
6. Run a disposable 64-step strong-batch fit check on an identical materialized
   batch sequence for control and candidate. Require finite losses throughout,
   candidate terminal loss EMA no more than 1.5x control, all six gamma tensors
   still finite and nonzero, and no EXP-014-style one-class collapse.

These are broad defect/collapse gates, not hyperparameter-tuning metrics. A gate
failure blocks the full run; do not weaken the threshold, change gamma's initial
value, or add warmup within EXP-015.

## Paired H20 Timing and Exposure Gates

The executed graph, tensor shapes, parameter count, optimizer membership, and
memory footprint are unchanged, so expected timing ratio is 1.000. Still run
five alternating fresh-process control/candidate pairs on the single idle H20
near 97,871 MiB. Use batch 128, cloned aligned initialization, separate SGD
state, pinned host inputs, alternating hard/probability targets, 100 warmup
steps, and 500 measured synchronized production-region steps per trial.

Record trial mean, median, p95, CV, peak allocation, and finite loss/gradient
status. Require:

- candidate/control median trial-mean step ratio at most 1.01;
- candidate p95 at most 1.02x control p95 and trial-mean CV below 2% for both;
- `floor(26_898 * control_mean / candidate_mean) >= 26_629`, retaining at least
  99.0% of EXP-010's updates;
- candidate peak allocation below 650 MB and no more than 16 MB above control;
- inference mean ratio at most 1.01 under the same batch/evaluator shapes; and
- conservative projected total runtime below 540 seconds.

Because batch size and epoch length remain 128 and 390, respectively, the
candidate should preserve about 69 epochs and 19 evaluation opportunities. The
data pipeline is byte-identical and does not need a new loader-throughput claim;
the full run must still stop eight workers exactly once and remain below the
600-second limit. If timing misses a gate, inspect environmental contention or
an accidental code change; do not alter the candidate to recover exposure.

## Testable Hypothesis

**Primary hypothesis:** initializing the six ordinary same-width residual
branches as forward identities, while retaining accepted gamma-one stage-entry
branches, will improve early conditioning and generalization enough to raise
`best_test_acc` from 94.15% to at least **94.25%**, while retaining at least 99%
of EXP-010's optimizer exposure.

A plausible success range is 94.25-94.45%. The expected mechanism is smoother
recruitment of within-stage residual refinements without weakening the feature
creation and channel expansion performed at stage entries. A larger gain is not
justified at ResNet-20 depth, and the one-step branch delay can instead reduce
short-horizon fit.

Mechanism diagnostics are:

- 80% clean checkpoint above the 87.08% underfit marker and preferably near
  EXP-010's 89.73%;
- first weak checkpoint near or above EXP-010's 93.16%;
- final NLL near or below 0.1934 and a nondeclining terminal trajectory;
- approximately 26.6k-27.2k steps, 69 epochs, 19 unique evaluations, and about
  50% realized CutMix batches.

Only `best_test_acc >= 94.25%` satisfies the goal. Fit, NLL, exposure, or a
better intermediate checkpoint cannot override a missed primary threshold.

## Risks and Failure Interpretation

- **Short-horizon underfit.** Six of nine residual branches carry no loss signal
  into their convolutions on the first update and may retain small gammas during
  the strong phase. A switch checkpoint below 87.08% supports this mechanism.
- **Too-shallow initial function.** At initialization, representation changes
  come mainly from the stem and three retained stage-entry units. A shallow
  starting map may waste early high-LR updates even though the gated branches
  wake on step two.
- **Gamma update overshoot.** Normalized residuals can still yield a large gamma
  gradient. EXP-014 proves that zero output alone is insufficient; the explicit
  first-update gamma/logit/loss gates address catastrophic scale, not accuracy.
- **Post-add ReLU is only a forward identity on nonnegative inputs.** Exact-zero
  coordinates have zero derivative in PyTorch, so the shortcut gradient is not
  mathematically identity everywhere. This may matter under sparse activations.
- **Coupled-decay first step.** Gated branch convolutions receive decay but no
  loss gradient on step one. The effect is tiny (`lr * wd = 1e-5`) but means the
  branch is not bitwise frozen between the first two forwards.
- **BN-statistics/gate mismatch.** Running statistics accumulate while gamma is
  zero. When gamma grows, those statistics reflect hidden strong-view features;
  this can stabilize recruitment or create a delayed scale mismatch.
- **Stage-entry asymmetry.** Leaving `layer1[0]` gamma one is a deliberate scope
  choice, not required by padding. The result cannot determine whether zeroing
  that one additional safe block would be better.
- **Limited depth benefit.** Literature evidence comes from deeper networks.
  Nine blocks may already optimize easily, leaving less than the required 0.10
  point or making accepted random residual perturbations beneficial.
- **Single-seed resolution.** A bare threshold pass is only ten test examples.
  It is valid under the declared protocol but weak causal evidence; do not reroll
  or tune the zeroed block set after the result.

## Verification and Decision Rule

After every structural, gradient, fit, and timing gate passes, run the candidate
exactly once:

```bash
CUDA_VISIBLE_DEVICES=0 timeout 600s uv run train.py > run.log 2>&1
```

Require exit 0, finite summary fields, about 300 counted seconds, total runtime
below 600 seconds, exactly 1,073,962 parameters, at least 26,629 steps, one 80%
augmentation switch, eight stopped workers, unique evaluation epochs, no more
than one evaluation per epoch, correct hard/soft target provenance, and only the
reviewed `train.py` diff. Preserve seed 42 and never reroll a valid run.

- **Improvement:** accept only if `best_test_acc >= 94.25%` and every integrity
  condition passes.
- **Valid no-improvement:** revert the selective initialization if the correct
  run completes below 94.25%; use gamma recruitment, switch fit, first weak
  accuracy, NLL, and trajectory to explain the mechanism.
- **Mechanical failure:** repair only an implementation or environment defect
  that leaves the candidate definition unchanged. Any different initial gamma,
  additional zeroed block, optimizer exception, warmup, or topology change
  requires a new reviewed experiment.

## Attribution

The only initial state difference is 448 gamma scalars across six existing
same-width residual branches. Runtime operations, capacity, global RNG state,
data stream, optimizer, and evaluation remain accepted. The result therefore
estimates the net effect of scoped same-width identity initialization, including
its one-step gradient delay and subsequent learned gate trajectory.

It cannot isolate first-stage-entry initialization, identify an optimal gamma,
or support all-block zero-gamma with Option-A padding. Those are explicitly
outside EXP-015.
