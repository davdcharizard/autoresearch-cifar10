# Proposal: Identity-Initialized Squeeze-Excitation in Every Residual Branch

## Summary

Add one squeeze-excitation (SE) channel gate to the residual branch of each of
the nine `BasicBlock`s in the accepted width-2 ResNet-20. For a block output
`U = bn2(conv2(...))`, globally average each channel, pass the descriptor through
a two-layer `C -> C/16 -> C` MLP with ReLU and sigmoid, and multiply `U` by the
result immediately before the existing shortcut addition. Apply no attention to
the stem, shortcut, post-addition activation, pooled classifier feature, or
classifier.

Keep the complete EXP-010 recipe unchanged: width 2, Option-A shortcuts,
all-parameter SGD decay `1e-4`, batch 128, seed 42, N1/M7 plus alpha-1 CutMix on
50% of strong batches through 80% elapsed training, the hard-label crop/flip
tail, LR schedule, timer, workers, evaluator, and summary. This isolates
input-conditioned channel recalibration as the only conceptual intervention.

Use the paper-supported reduction ratio `r=16`, giving hidden dimensions 2, 4,
and 8 for the 32-, 64-, and 128-channel stages. Initialize each gate to exactly
one with `2 * sigmoid(0)`, so the candidate begins as the accepted residual
network rather than abruptly halving every residual branch. Preserve the global
CPU RNG state while constructing and initializing SE parameters so all shared
weights and the subsequent shuffle/augmentation stream remain aligned with the
accepted seed as far as the architecture permits.

The moving baseline is EXP-010 at `94.15%`, 26,898 steps, 1,073,962 parameters,
and 598.7 MB peak VRAM. A valid improvement requires `best_test_acc >=94.25%`.
The candidate must retain at least 95% of accepted update exposure in a paired,
synchronized H20 preflight before it is allowed a full run.

## Primary Evidence

Hu, Shen, and Sun's original
[CVPR 2018 paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.pdf)
defines the SE operation as global-average squeeze followed by a two-layer
bottleneck excitation, ReLU, sigmoid, and channel-wise multiplication. For a
ResNet, Figure 3 places this transformation on the non-identity branch before it
is summed with the shortcut. The paper reports that:

- SE blocks improve several residual and non-residual ImageNet architectures,
  supporting channel recalibration as a portable representation mechanism;
- `r=16` is the authors' selected accuracy/complexity tradeoff, while their
  ablation shows accuracy is not monotonic in gate capacity;
- the added arithmetic in SE-ResNet-50 is only about 0.26% by FLOPs, but measured
  training latency rises from 190 ms to 209 ms on their system, warning that
  pooling and small fully connected kernels can be launch-bound;
- the parameter cost is concentrated in the two excitation layers, and applying
  gates across stages gives complementary benefits.

The authors'
[official repository](https://github.com/hujie-frank/SENet) likewise specifies
`r=16`, places squeeze/excitation before residual summation, and notes that their
global-pooling and scale-plus-add implementations needed optimization. These
sources justify the topology and also make an H20 wall-time gate mandatory. They
do not establish an effect size for a shallow CIFAR-10 model, CutMix, or a
300-second horizon; the expected gain here remains an experiment-specific
hypothesis.

## Why This Fits the Local Trajectory

The strongest local results point toward representation quality rather than
another scalar regularization adjustment:

- EXP-007 gained 1.25 points by doubling width despite losing 29.2% of width-1
  updates. The accepted model has enough channel capacity for selective routing
  to be meaningful.
- EXP-008 and EXP-009 bracketed the accepted all-parameter `1e-4` decay from
  opposite directions. Changing norm pressure did not improve top-1.
- EXP-010 gained 0.60 points from p=0.5 CutMix while retaining 99.10% of EXP-007
  updates. Its strong checkpoint remained healthy and its weak tail ended at the
  best score, so the full data and schedule recipe should be preserved.
- EXP-011 showed that simply increasing CutMix probability to 0.75 compounds
  strong-phase underfit and loses 0.15 points. SE changes how the established
  model represents those views instead of increasing augmentation pressure.

CutMix makes the proposed mechanism especially plausible but not guaranteed.
Each mixed image contains two spatially separated class-bearing regions. A
global descriptor can condition channel emphasis on their combined content,
allowing the residual branch to retain features useful for both target classes.
RandAugment also changes which response families are reliable per sample. The
same gates can then adapt to ordinary single-image views after the 80% switch.
This is a higher-ceiling hypothesis than another probability interpolation, but
the phase distribution shift is also a central failure risk.

## Exact Architecture

Add one constant:

```python
SE_REDUCTION = 16
```

Use two biased linear layers. For input `x` with shape `[B, C, H, W]`:

```python
descriptor = x.mean(dim=(2, 3))
hidden = F.relu(self.reduce(descriptor))
scale = 2.0 * torch.sigmoid(self.expand(hidden))
return x * scale[:, :, None, None]
```

The factor of two is a deliberate fixed-horizon initialization control, not a
claim that it is part of the canonical paper. The original sigmoid constrains
gates to `(0, 1)` and a zero logit produces 0.5, which would change the scale of
every accepted residual branch before attention has learned anything. The
centered form retains independent sigmoid gates, allows both attenuation and
amplification in `(0, 2)`, and makes a zero excitation output an exact identity
gate. Do not tune this range after timing or accuracy is observed.

In `BasicBlock.__init__`, construct exactly one gate using `out_channels`. In
`BasicBlock.forward`, change only the residual path between `bn2` and addition:

```python
out = self.bn2(self.conv2(out))
out = self.se(out)
# existing Option-A shortcut logic remains byte-identical
out += shortcut
return F.relu(out)
```

This placement matches the proposed SE-ResNet integration: squeeze and excitation
operate on the complete transformed branch before aggregation. Do not gate after
addition, because that would modulate the identity path and the paper's placement
study found post-aggregation SE weaker. Do not gate after `conv1`, only the two
stride-2 blocks, only the last stage, or the classifier; those are distinct scope
and capacity experiments.

## Reduction Ratio and Parameter Count

Use `hidden = out_channels // 16` without rounding alternatives; all accepted
stage widths divide exactly. With biases enabled, each gate adds
`2*C*hidden + hidden + C` parameters:

| Stage | Blocks | C | Hidden | Parameters per block | Stage total |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 3 | 32 | 2 | 162 | 486 |
| 2 | 3 | 64 | 4 | 580 | 1,740 |
| 3 | 3 | 128 | 8 | 2,184 | 6,552 |
| **Total** | **9** | | | | **8,778** |

The exact candidate count must therefore be `1,082,740`, an increase of 0.817%
over 1,073,962. The excitation matrix work is 8,192 multiply-accumulates per
image across all blocks. Squeeze and channel scaling each touch 172,032 feature
values per image. Arithmetic is small relative to the convolutions, but nine
reductions, eighteen tiny linears, nine sigmoids, and nine multiplications add
sequential kernel launches; parameter/FLOP counts cannot substitute for timing.

## Initialization and RNG Alignment

Pin the SE initialization as follows:

- `reduce.weight`: Kaiming normal with ReLU nonlinearity;
- `reduce.bias`: zero;
- `expand.weight`: zero;
- `expand.bias`: zero.

Thus every initial excitation logit is zero and every gate is exactly one.
`expand.weight` and `expand.bias` receive gradients on the first backward pass;
`reduce.weight` begins receiving gradients after the expansion moves away from
zero. A one-update delay for the reduction layer is accepted and must not be
"fixed" by choosing an unreviewed nonzero expansion initialization.

Adding modules normally consumes CPU RNG twice: once in `nn.Linear`'s default
constructor initialization and again in the model's current `self.apply` pass.
That would shift later shared convolution/classifier initialization and, because
the DataLoader uses the process RNG, can also shift shuffle and worker seeds.
Prevent this confound:

1. Use a distinct `SELinear(nn.Linear)` marker type.
2. Construct and explicitly initialize both SE linears inside
   `torch.random.fork_rng(devices=[])`, so their draws do not advance the global
   CPU RNG.
3. Make `ResNet._weights_init` return immediately for `SELinear` before its
   existing generic `nn.Linear` branch. All shared Conv/Linear draws then occur
   in the same order as EXP-010.

Do not add a new seed or a separate fixed SE seed. The forked scope derives each
gate's reduction initialization from the contemporaneous seed-42 state while
restoring that state afterward. The global RNG state after model construction
must be byte-identical to an accepted-model reference, and all same-named shared
parameters must be bitwise identical in a disposable construction test.

All new parameters remain in `model.parameters()` and the existing single SGD
group, including both biases. They therefore receive the accepted momentum and
coupled `1e-4` weight-decay semantics. Do not create a no-decay gate group; the
local history rejected changing all-parameter decay, and a new optimizer group
would confound the architecture test.

## Exact Scope and Confound Controls

Starting from accepted commit `7c1e7d8`, modify only `train.py` to add:

- `SE_REDUCTION = 16`;
- the `SELinear` marker and SE module;
- one SE module in every `BasicBlock`;
- the one residual-branch call after `bn2`;
- the minimal `_weights_init` exclusion needed for the pinned SE/RNG policy.

Keep unchanged:

- width, block count, channel widths, convolutions, BatchNorm, Option-A shortcut,
  stem, classifier, post-addition ReLU, and FP32 precision;
- all optimizer values and the single all-parameter group;
- transforms, CutMix alpha/probability/order, target formats, worker-side
  collator, batch size, loader options, shutdown lifecycle, and RNG fork in the
  existing CutMix collator;
- the 80/20 elapsed-time LR and augmentation boundary, maximum steps, seed,
  timer boundaries, per-step synchronization, evaluation cadence, evaluator,
  and summary fields.

Do not combine SE with projection shortcuts, preactivation, zero-initialized
`bn2`, BF16, compilation, channels-last, fused SGD, larger batches, altered
CutMix timing, or attention diagnostics in the timed path. Each changes a second
mechanism or compute path.

## Mandatory Functional Preflight

Use disposable `/tmp` scripts; do not edit tracked files or run a partial
training experiment. Construct an accepted reference and candidate from cloned
seed-42 RNG states and require:

- all shared state-dict tensors are bitwise equal and the final global CPU RNG
  states are byte-identical;
- exactly nine gates with stage counts 3/3/3 and hidden widths 2/4/8;
- exact candidate parameter count `1,082,740`, with all parameters in the one SGD
  group and `weight_decay=1e-4`;
- for seeded tensors at 32x32, 16x16, and 8x8, every initial gate is exactly 1,
  candidate logits match the accepted reference, and all outputs are finite;
- after one backward pass, shared gradients match the accepted reference within
  `rtol=1e-6, atol=1e-7`, every expansion layer has a finite nonzero gradient,
  and every reduction-layer gradient is exactly zero as declared;
- after two optimizer steps, at least one reduction-layer gradient is finite and
  nonzero, proving the gate is trainable rather than permanently inert;
- forward/backward succeeds with both hard `[128]` labels and CutMix probability
  targets `[128, 10]`.

Also compile `train.py`, run Ruff/pre-commit, and statically confirm that the
evaluator remains reachable no more than once per epoch. Any RNG-alignment,
identity, gradient, count, optimizer-membership, or target-format failure is a
planning defect and a full-run no-go.

## Mandatory Synchronized H20 Timing Gates

Confirm the only visible accelerator is an idle NVIDIA H20 with approximately
98 GB VRAM. Benchmark accepted and candidate models in fresh processes on the
same device, PyTorch/cuDNN version, and default precision settings. Use identical
cloned shared weights, separate fresh SGD state, a reusable pinned CPU image
batch `[128, 3, 32, 32]`, and deterministic alternating hard and probability
targets to represent the accepted 50% CutMix mixture.

For each model:

1. Run 100 untimed full training steps to warm cuDNN and allocators.
2. Run 500 timed steps using the exact accepted `t0`/`dt` region: pinned
   nonblocking H2D copies, `zero_grad`, forward, cross-entropy, backward, SGD
   step, and `torch.cuda.synchronize()`.
3. Repeat five paired trials, alternating candidate/control order. Recreate model
   and optimizer state for every trial.
4. Record each trial's aggregate mean, median, p95, images/s, and peak allocation.
   Use the median of the five trial means for the exposure projection.

All training gates must pass:

- coefficient of variation of trial means below 3% for each model;
- `candidate_mean / accepted_mean <= 1.0526`;
- `projected_steps = floor(26_898 * accepted_mean / candidate_mean) >= 25_553`,
  retaining at least 95.0% of EXP-010 updates;
- candidate p95 no more than 1.10x accepted p95;
- candidate peak allocation below 700 MB and no more than 64 MB above the paired
  control.

Benchmark inference separately with `model.eval()`, `torch.inference_mode()`, 100
warmups, and 500 synchronized forwards per trial. Require candidate/control mean
ratio at most 1.10 and trial CV below 3%. Conservatively project total runtime as
accepted 330.7 seconds plus the measured incremental cost over its 19 evaluator
passes; require the result below 540 seconds. This leaves one minute before the
hard 600-second timeout for loader and process jitter.

The 95% exposure floor projects at least about 20,442 strong-phase and 5,111
weak-tail steps because both phases are time-based. It is a feasibility gate,
not an accuracy acceptance condition. If it fails, do not remove late-stage
gates, change `r`, fuse operations, compile, or add mixed precision to rescue the
candidate; those would define another experiment.

## Hypothesis and Expected Impact

**Hypothesis:** identity-initialized SE gates at `r=16` will learn useful
per-sample channel dependencies across the width-2 residual hierarchy, retain at
least 25,553 optimizer steps on the H20, and improve `best_test_acc` from 94.15%
to at least **94.25%** under the unchanged EXP-010 training and evaluation
protocol.

The plausible successful range is 94.25-94.55%. A gain near the lower end would
mean channel selection provides a small benefit after CutMix and width have
already captured most easy headroom. A larger gain is possible if the current
wide model wastes channels on augmentation-specific responses and the hard tail
learns to reuse the gates on clean views. This range is deliberately much smaller
than the original paper's deeper ImageNet improvements.

Mechanism-supporting, non-veto diagnostics are: a final strong checkpoint above
the predeclared 87.08% compounded-underfit marker, a first weak checkpoint near
or above EXP-010's 93.16%, a terminal NLL no worse than 0.1934, and a rising or
stable late trajectory. The formal verdict remains the primary 94.25% threshold;
these diagnostics explain an outcome but cannot override it.

## Failure Mechanisms

- **Small-kernel launch overhead.** Nine reductions and many tiny elementwise/FC
  kernels may cost far more wall time than their FLOP count suggests, shortening
  both representation learning and the hard tail.
- **Composite-descriptor mismatch.** During mixed batches, global pooling blends
  donor and source regions into one descriptor. The gate may emphasize channels
  for the dominant area or learn composite-specific correlations that disappear
  abruptly in the weak tail.
- **Compounded strong-phase suppression.** Even centered gates can learn values
  below one. Channel suppression on top of N1/M7 and CutMix may recreate
  EXP-011's underfit mechanism despite unchanged probability.
- **Too-narrow early bottlenecks.** With `r=16`, stage 1 uses only two hidden
  values. This is paper-supported and cheap, but may underfit useful early
  channel relationships. Changing to `r=8` after seeing the result is not allowed.
- **Identity initialization learns slowly.** Zero expansion preserves the
  accepted starting network but blocks reduction gradients on the first update
  and may keep gates near one for too much of the short horizon.
- **BatchNorm interaction.** SE scales an already normalized residual branch
  without a following BN before addition. Input-conditioned branch magnitude can
  change the shortcut/residual balance and downstream running statistics.
- **Overfitting channel correlations.** The paper's reduction-ratio ablation was
  non-monotonic and explicitly attributes excessive gate capacity as a possible
  source of overfit. CIFAR-10 is much smaller than ImageNet.
- **Weak evidence transfer.** The primary evidence comes mainly from deeper,
  longer ImageNet models. It does not guarantee a gain for width-2 ResNet-20,
  CutMix, one seed, or 300 counted seconds.
- **Margin and seed noise.** The acceptance margin is ten CIFAR-10 examples. One
  valid seed-42 run is decisive for this protocol; no rerun may be used to select
  a favorable result.

## Verification and Decision Rules

If selected, pass every functional and paired H20 gate, then run exactly once as
required:

```bash
uv run train.py > run.log 2>&1
```

Require exit zero, one complete finite summary, approximately 300 counted
training seconds, total below 600 seconds, peak VRAM below the preflight bound,
and exactly 1,082,740 parameters. Require one augmentation/CutMix-to-base switch
near 80%, all eight old workers stopped, no soft target in the weak tail, unique
evaluation epochs, and no more than one evaluation per epoch. Record actual
steps, epochs, strong/tail exposure, switch checkpoint, first weak checkpoint,
final NLL, best/final gap, and accuracy trajectory against EXP-010.

- **Accept:** `best_test_acc >=94.25%`, all protocol checks pass, and the full run
  retains at least 25,553 steps. SE becomes part of the moving recipe.
- **Accuracy failure with valid exposure:** reject SE at this exact all-block,
  `r=16`, identity-initialized operating point. Do not tune placement, ratio, or
  initialization inside EXP-012.
- **Accuracy pass below the exposure gate:** the metric formally clears the goal
  threshold but the predeclared mechanism comparison is confounded; review timing
  integrity before integration and do not describe SE as low-cost.
- **Underfit trajectory:** if the strong checkpoint crosses below 87.08% and
  accuracy fails, attribute the result first to compounded branch suppression,
  not insufficient CutMix probability or decay.
- **Runtime, RNG, target, initialization, count, optimizer, lifecycle, or
  evaluator failure:** invalid. Fix only the protocol defect and rerun the same
  declared candidate; do not alter the seed or combine a rescue optimization.

Remove `run.log` after analysis and revert to accepted commit `7c1e7d8` on any
valid no-improvement.

## Evidence Paths

- `.autoresearch/goals/maximize-cifar10-best-test-accuracy/01-definition.md`
- `.autoresearch/goals/maximize-cifar10-best-test-accuracy/03-experiment-learnings.md`
- `.autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/010/04-analysis.md`
- `.autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/011/04-analysis.md`
- `train.py` at accepted commit `7c1e7d8`
- [Hu, Shen, and Sun, Squeeze-and-Excitation Networks, CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/papers/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.pdf)
- [Official SENet implementation repository](https://github.com/hujie-frank/SENet)
