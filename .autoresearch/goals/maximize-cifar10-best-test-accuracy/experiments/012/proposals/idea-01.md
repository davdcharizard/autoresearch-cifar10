# Idea: Option-B Projection Shortcuts at the Two Stage Transitions

## Summary

Replace only the two parameter-free Option-A shortcuts that downsample and double channels with learned `1x1`, stride-2, bias-free convolutions followed by BatchNorm. Keep the other seven residual-block shortcuts as exact identities. Preserve the complete accepted EXP-010 recipe: width-2 post-activation ResNet-20, alpha-1 CutMix on 50% of N1/M7 plateau batches, the 80% switch to a hard-label weak tail, all-parameter coupled `1e-4` weight decay, batch 128, elapsed-time LR schedule, seed, evaluator, and worker lifecycle.

This is the original ResNet paper's Option B at dimension-changing shortcuts, adapted to the accepted width-2 model. It is not a general replacement of identity shortcuts and not an anti-aliasing intervention. A `1x1`, stride-2 convolution samples the same even spatial phase as the current `x[:, :, ::2, ::2]`; its additional mechanism is learned channel mixing and a nonzero shortcut contribution to every output channel.

The change adds exactly 10,624 trainable parameters, taking the model from 1,073,962 to 1,084,586 parameters, and adds 1,048,576 forward multiply-accumulates per image. A local paired H20 timing diagnostic measured a 1.87% synchronized-step cost and projects 26,404 fixed-budget steps versus EXP-010's 26,898. It is therefore a small representation-quality bet rather than a meaningful capacity-throughput trade.

## Diagnosis

The accepted frontier is EXP-010 at 94.15%:

- Width 2 produced the largest architectural gain so far: +1.25 points despite 29.2% fewer updates than width 1. This supports improving representation quality even when the intervention is not compute-free.
- Plateau-only p=0.5 CutMix then added 0.60 points with 99.10% of width-2 exposure, reaching 94.15%, 0.1934 final NLL, 26,898 steps, and 598.7 MB peak VRAM.
- EXP-011's p=0.75 result and EXP-008/009's decay results bracket the accepted regularization operating point. The data geometry, phase boundary, and optimizer should remain fixed.
- The current two stage-transition shortcuts are unusually asymmetric. At stage 2, 32 input channels are subsampled and copied while 32 output channels receive zero shortcut signal. At stage 3, 64 are copied and 64 are zero-filled. The residual branch alone must create the new-channel half at both transitions.

The relevant untested question is whether the accepted wider model benefits from learned basis alignment exactly where spatial resolution and channel dimensionality change. Projection shortcuts give every one of the 64 and 128 transition outputs a learned skip-path mixture of the preceding features. This may improve transport of regional features created by CutMix and RandAugment into later stages without changing residual-branch depth or the training objective.

The diagnosis must not be overstated. Option A does not delete the entire skipped representation: it passes every input channel exactly at one spatial phase, and the stride-2 3x3 residual branch sees local neighborhoods. Option B also samples only one spatial phase, so it does not recover the other three positions of each `2x2` cell or solve aliasing. The experiment isolates learned channel projection and normalization, not information-preserving spatial downsampling.

## Primary Evidence and Transfer Limits

He et al., [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385), define Option A as identity downsampling plus zero padding and Option B as a `1x1` projection only when dimensions increase, with stride 2 across spatial-size changes. On ImageNet ResNet-34, their Option B reduced top-1 error from 25.03% to 24.52% relative to Option A. They attributed the small advantage to the zero-filled dimensions in Option A lacking a shortcut contribution. Option C, which projected every shortcut, added much more complexity for only a marginal further gain, supporting this proposal's two-transition scope.

That result is directional evidence, not a portable effect estimate. The paper used Option A for its CIFAR experiments, did not report an A/B CIFAR ablation, and its ImageNet ResNet-34 differs in dataset, width, training duration, and augmentation. The local model is a shallow width-2 CIFAR ResNet trained for only 300 counted seconds with composite augmentation.

Han et al., [Deep Pyramidal Residual Networks](https://openaccess.thecvf.com/content_cvpr_2017/papers/Han_Deep_Pyramidal_Residual_CVPR_2017_paper.pdf), provide relevant counterevidence: they favor zero-padded identity transport over learned projections for generalization and optimization, particularly in very deep networks. The accepted model has only nine residual blocks and would add only two projections, which weakens but does not eliminate that concern. Losing exact identity gradients at the two transitions is the main architectural downside.

## Exact Architecture Semantics

Only `BasicBlock` shortcut construction and use should change. The residual branch remains byte-equivalent:

```text
conv3x3(stride) -> BN -> ReLU -> conv3x3 -> BN
```

The shortcut becomes:

```python
if stride == 2:
    self.shortcut = nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=2,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
    )
else:
    self.shortcut = nn.Identity()
```

Forward remains post-activation:

```python
out = F.relu(self.bn1(self.conv1(x)))
out = self.bn2(self.conv2(out))
out += self.shortcut(x)
return F.relu(out)
```

The exact block inventory is:

| Location | Input -> output | Shortcut |
|---|---|---|
| Stage 1, blocks 1-3 | `32x32x32 -> 32x32x32` | identity |
| Stage 2, block 1 | `32x32x32 -> 64x16x16` | `1x1`, stride 2, `32 -> 64`, BN |
| Stage 2, blocks 2-3 | `64x16x16 -> 64x16x16` | identity |
| Stage 3, block 1 | `64x16x16 -> 128x8x8` | `1x1`, stride 2, `64 -> 128`, BN |
| Stage 3, blocks 2-3 | `128x8x8 -> 128x8x8` | identity |

Static inspection must find exactly two projection convolutions, exactly two projection BNs, and exactly seven `nn.Identity` shortcut modules. Do not project the same-shape blocks. Do not add a projection to the stem, classifier, or residual branches. Do not use a `3x3` shortcut, average pooling, blur pooling, or symmetric channel padding.

The `1x1` convolutions use no padding and therefore select input coordinates `(0, 2, 4, ...)`, matching the spatial phase of the current slice. This makes the architectural contrast sharper: fixed per-channel copy plus zero fill versus learned channel mixtures at the same sampled coordinates.

## Initialization and Optimization Semantics

The existing `self.apply(self._weights_init)` visits the new projection convolutions and initializes them with the same `init.kaiming_normal_` rule as every other convolution. Projection BNs retain PyTorch defaults: affine scale one, affine bias zero, running mean zero, and running variance one. Projection convolutions stay bias-free because the following BN makes a convolution bias redundant.

Do not zero-initialize projection BN scales, use identity-like partial copies, scale either branch, or zero-initialize residual `bn2`. Those alternatives answer different questions and could make the transition initially identity-like or residual-dominant. The proposal tests ordinary learned conv+BN projections.

The accepted optimizer continues to receive `model.parameters()` as one group. New projection convolution weights and BN affine parameters therefore use ordinary SGD momentum 0.9 and the accepted all-parameter coupled `1e-4` weight decay. EXP-008/009 specifically warn against changing this decay policy at width 2.

Adding randomly initialized convolution weights consumes extra CPU RNG during module construction and initialization before the first training iterator is created. Consequently, even with seed 42, later shuffle, worker-seed, augmentation, and CutMix draws can differ from EXP-010. The run estimates the net fixed-seed value of the projection architecture, not a paired sample-by-sample causal effect. Do not add RNG realignment machinery, reseed after model construction, or reroll a valid run; any such change would alter the accepted stochastic protocol and complicate the isolated architecture diff.

## Exact Parameter and Compute Cost

All counts below include trainable BN affine parameters and exclude non-trainable running buffers:

| New component | Derivation | Parameters |
|---|---:|---:|
| Stage-2 projection convolution | `64 * 32 * 1 * 1` | 2,048 |
| Stage-2 projection BN | `64 weight + 64 bias` | 128 |
| Stage-3 projection convolution | `128 * 64 * 1 * 1` | 8,192 |
| Stage-3 projection BN | `128 weight + 128 bias` | 256 |
| **Added** | | **10,624** |
| Accepted EXP-010 model | | 1,073,962 |
| **Candidate total** | | **1,084,586** |

The increase is 0.989% of accepted trainable parameters. The forward convolution MAC increment per image is:

```text
16 * 16 * 64 * 32 + 8 * 8 * 128 * 64
= 524,288 + 524,288
= 1,048,576 MACs
```

The accepted convolution/classifier path is approximately 161.3M forward MACs per image, so the theoretical increment is about 0.65%, before BN and kernel-launch effects. The extra forward activations across both projections total 24,576 values per image. They are short-lived stage-transition tensors, and measured allocator peak remained effectively unchanged.

## Measured H20 Feasibility

During proposal development, one idle NVIDIA H20 with 97,871 MiB was confirmed. An in-memory candidate implementation was compared with accepted `train.ResNet` in three alternating paired trials. Each trial used a fresh model and optimizer, batch 128 pinned host inputs, nonblocking H2D, FP32 forward, hard-label cross-entropy, backward, SGD, and terminal CUDA synchronization. Thirty steps were warmed and 150 were measured per model per trial.

| Model | Trial mean range | Median trial mean | Trial-mean CV | Peak allocation | Parameters |
|---|---:|---:|---:|---:|---:|
| Accepted Option A | 10.835-10.878 ms | 10.852 ms | 0.20% | 598.7 MB | 1,073,962 |
| Two Option-B projections | 11.020-11.101 ms | 11.055 ms | 0.37% | 598.8 MB | 1,084,586 |

The candidate/control ratio was 1.01870. Calibrating to EXP-010's real 26,898 steps gives:

```text
26,898 * 10.8522 / 11.0551 = 26,404 projected steps
```

That is 98.17% exposure retention, roughly 67.7 epochs at 390 batches per epoch, about 21,123 strong steps and 5,281 weak-tail steps, or approximately 13.5 weak epochs. The measured cost is larger than the 0.65% MAC estimate because the two conv+BN paths add small-kernel launch and normalization overhead, but it is still operationally modest.

This diagnostic did not import an implemented candidate from `train.py`, used a repeated synthetic batch, and cannot predict accuracy. Planning must rerun the paired benchmark against the actual reviewed implementation. Its value here is to establish a credible H20 prior and rule out a large hidden throughput penalty.

## Preflight and Launch Gates

Before a full experiment, the implemented candidate must pass all of the following:

1. Confirm exactly one idle H20 near 97,871 MiB, the moving baseline 94.15% at `7c1e7d8`, and no stale `run.log` variant.
2. Compile and lint `train.py`; require output shape `(2, 10)` and exactly 1,084,586 trainable parameters.
3. Hook every block and assert the shape inventory above, two conv+BN projection paths, seven identity paths, no Option-A padding path, and no projection on a stride-1 block.
4. Check projection convs are `1x1`, stride 2, padding 0, bias-free; check projection BN feature counts are 64 and 128 and their initial affine/running state is standard.
5. Diff against accepted EXP-010 and require the model width, residual branches, CutMix collator, transforms, loaders, optimizer, LR/phase schedule, timer, evaluator, seed, and summary schema to remain unchanged.
6. On the idle H20, use three paired fresh-state trials with alternating order, at least 50 warmup and 200 synchronized steps, and the exact accepted H2D/forward/loss/backward/SGD timed region. Record mean, median, p95, trial CV, finite loss/gradients, peak allocation, ratio, and calibrated steps.
7. Treat timing gates as broad operational gates rather than accuracy-derived point estimates: candidate/control mean ratio must be at most 1.05, calibrated exposure at least 25,500 steps, trial-mean CV below 5%, and peak allocation below 750 MB. The measured 1.0187 ratio, 26,404 projection, sub-0.4% CV, and 598.8 MB pass with margin.

The 25,500-step hard floor preserves about 94.8% of EXP-010 exposure and more than 13 projected weak epochs. Expected exposure remains at least 26,000 steps; landing between 25,500 and 26,000 is feasible but weakens the throughput premise and must be highlighted rather than used to tune the design. Do not relax a failed gate, remove BN, or change shortcut form within EXP-012. A different shortcut requires a new reviewed experiment.

## Preserved Accepted Recipe

Outside the shortcut implementation, keep accepted `train.py` unchanged:

- width multiplier 2, channels `32/64/128`, three blocks per stage, post-activation ordering, 3x3 residual convolutions, adaptive average pool, and classifier;
- Kaiming-normal convolution/linear initialization and BatchNorm defaults;
- crop/flip plus N1/M7 RandAugment and alpha-1 CutMix at probability 0.5 during the first 80% of counted time;
- exact worker-side RNG isolation in `cutmix_collate`, target-format counters, and no CutMix on the weak loader;
- one explicit strong-loader break, evaluation, shutdown of eight persistent workers, garbage collection, and weak crop/flip loader rebuild at the 80% boundary;
- batch 128, hard or CutMix probability-target cross-entropy as already selected by target shape;
- ordinary SGD, momentum 0.9, all-parameter weight decay `1e-4`, and no Nesterov;
- `lr=0.1` through 80%, then the accepted step to `0.01` and elapsed-time cosine to `1e-4`;
- checkpoints `(0.2, 0.4, 0.6, 0.7)`, dense weak-tail evaluation at most once per epoch, fixed evaluator, seed 42, 300-second counted budget, 600-second wall timeout, and existing summary keys.

Do not add projection-specific learning rates, exclude projection BN from decay, alter CutMix probability, reset new BN statistics at the phase switch, or change batch size to recover the small measured overhead.

## Testable Hypothesis

**Primary hypothesis:** replacing only the two stride-2 Option-A shortcuts with ordinary `1x1` stride-2 conv+BN projections on the accepted width-2 p=0.5 CutMix recipe will improve learned channel transport at both stage boundaries enough to raise `best_test_acc` from 94.15% to at least **94.25%**, while retaining at least 97% of EXP-010's optimizer exposure in expectation.

A plausible success range is 94.25-94.50% (+0.10 to +0.35 points). The lower end reflects the original ResNet paper's characterization of Option B as a small advantage and the strength of the current baseline. A larger prediction is not justified because CIFAR-specific A/B evidence is absent, the current model is shallow, and learned projections sacrifice exact transition identities.

Secondary predictions and diagnostics:

- `num_params` is exactly 1,084,586 and peak allocation remains below 750 MB;
- actual steps land near 26,000-26,700, preserving approximately 13-14 weak-tail epochs;
- the augmentation switch occurs once near 80.0%, stops eight workers, and reports approximately 50% mixed strong batches;
- the 80% clean checkpoint should be compared with EXP-010's 89.73%, the first weak checkpoint with 93.16%, and final NLL with 0.1934;
- projection BN running statistics remain finite and receive updates in both phases; no BN reset or recalibration pass occurs;
- total wall time remains near the accepted low-330-second range and below 600 seconds.

Only `best_test_acc >=94.25%` satisfies the goal. Lower NLL, better strong-phase accuracy, or preserved exposure cannot override a missed primary threshold.

## Attribution

The intervention is structurally isolated: two fixed slice-and-pad operators become two trainable conv+BN paths. Same-shape identities, residual branches, total depth, channel widths, classifier, data, loss, optimizer policy, phase schedule, timing, and evaluation remain fixed. Parameter count and step exposure are measured explicitly, so analysis can distinguish a major throughput failure from the intended representation bet.

Two effects cannot be separated in one fixed-time run:

1. Projection convolution and projection BN are bundled by the requested architecture; success cannot be attributed specifically to channel mixing versus normalization.
2. Added random parameters shift subsequent model/data RNG consumption. The accuracy result is the net fixed-seed candidate trajectory, not an equal-minibatch paired comparison with EXP-010.

The fixed-time objective also couples representation to a projected 1.8% update loss. This is the declared candidate, not a confound to compensate post hoc. No rerun, seed change, or throughput-recovery modification is allowed after a valid result.

## Risks and Failure Modes

- **Exact-identity loss:** Option A provides an unparameterized gradient path for retained channels. A random learned projection plus BN may make transition optimization less stable or generalize worse, especially when the gain sought is only 0.10 points.
- **Projection BN distribution shift:** the two new running-statistic states are trained mostly on N1/M7/CutMix data and receive only about 13.5 weak epochs to adapt. A comparable strong checkpoint followed by a worse first weak checkpoint would implicate transition-BN adaptation.
- **No spatial preservation gain:** `1x1`, stride 2 still samples only even locations. If spatial aliasing is the real transition problem, this mechanism should have little benefit.
- **Already sufficient residual branch:** the stride-2 3x3 residual convolutions may already learn the necessary new-channel basis, making Option A's zeros harmless in a nine-block network.
- **Small signal and one run:** the formal +0.10-point threshold is only ten CIFAR-10 examples. A bare pass is valid under the declared protocol but weak evidence for a precise causal effect size; do not overclaim or replicate by rerolling.
- **Mild fixed-time exposure loss:** measured retention is 98.17%, but clocks or an implementation discrepancy could cost more. Report actual steps and tail epochs against EXP-010.
- **Overfitting:** 10,624 extra trainable parameters are small, but the learned shortcut can bypass residual processing and fit shortcuts in the statistical sense. Final train loss, NLL, and trajectory should be compared without changing decay.
- **Implementation overreach:** projecting all nine blocks, using conv without BN, changing initialization, or resetting BN creates a different experiment and invalidates this proposal's evidence and parameter count.

## Verification and Decision Rule

After all preflight gates pass, run exactly once:

```bash
CUDA_VISIBLE_DEVICES=0 timeout 600s uv run train.py > run.log 2>&1
```

Require exit 0, ten finite summary fields, approximately 300 counted training seconds, total time below 600 seconds, one idle H20, one augmentation switch, eight clean worker exits, unique evaluation epochs, no more than one evaluation per epoch, correct CutMix target provenance, and exact candidate parameter count. Preserve seed 42 and do not retry a mechanically valid run.

- **Improvement:** accept only if `best_test_acc >=94.25%` and every integrity condition passes.
- **Valid no-improvement:** revert the projections if the run completes correctly below 94.25%; retain EXP-010 and use checkpoint/BN/throughput diagnostics to classify the failure mechanism.
- **Timing degradation:** a valid run below 25,500 steps remains a primary-metric result, but report unexpected candidate overhead as a major causal risk; do not rerun for favorable clocks.
- **Invalid:** wrong shortcut inventory or parameter count, altered accepted recipe, duplicate evaluation, worker leak, non-finite state, crash, or timeout. Fix only the protocol defect and rerun the identical reviewed candidate.
- **No adaptive rescue:** do not remove BN, initialize projections differently, project all shortcuts, change p=0.5, or adjust decay after observing the result. Each is a separate experiment.

## Evidence

- `.autoresearch/goals/maximize-cifar10-best-test-accuracy/01-definition.md`: primary metric, fixed one-H20 protocol, +0.10-point moving-baseline rule, scope, and no-seed-hacking rule.
- `.autoresearch/goals/maximize-cifar10-best-test-accuracy/04-results.tsv`: accepted 94.15% baseline at `7c1e7d8`, making 94.25% the current threshold.
- `.autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/007/04-analysis.md`: width-2 capacity result and validated H20 paired-timing methodology.
- `.autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/010/04-analysis.md`: accepted 94.15% CutMix frontier, exposure, checkpoints, NLL, runtime, VRAM, and parameter count.
- `.autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/011/04-analysis.md`: evidence to preserve p=0.5 and avoid further regularization interpolation.
- [He et al., Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385): Option-A/Option-B semantics and the original projection comparison.
- [Han et al., Deep Pyramidal Residual Networks](https://openaccess.thecvf.com/content_cvpr_2017/papers/Han_Deep_Pyramidal_Residual_CVPR_2017_paper.pdf): counterevidence favoring zero-padded identities over projections in deeper networks.
