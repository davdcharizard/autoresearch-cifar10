# Proposal: Reallocate PreAct WRN Depth from 2-2-2 to 1-2-3

## Summary

Fork from EXP-002 and change only the six-block PreAct WRN-16-4 stage-depth
allocation: one residual block at 64 channels, two at 128, and three at 256,
instead of two blocks in every stage. Keep widths `[64, 128, 256]`, total block
count six, twelve 3x3 convolutions, three projection shortcuts, and the complete
EXP-002 training recipe unchanged.

This moves one same-width block from 64 channels at 32x32 to 256 channels at
8x8. The two blocks have exactly equal convolution MACs because
`64^2 * 32^2 = 256^2 * 8^2`, so total Conv/Linear work remains exactly
392,612,352 MACs per image. Parameters rise from 2,748,890 to 3,855,578 because
weights, unlike activation compute, do not scale with spatial area.

The intervention spends the same major-convolution path on later semantic
features, increases approximate final receptive field from 53 to 65 pixels,
and introduces no auxiliary kernels. A same-harness GPU-0 latency preflight is
still mandatory because equal MACs do not guarantee equal H20 kernel speed.

## Motivation

EXP-002 is the intended parent at 95.23%. It contributes the validated
front-loaded CutMix gain while avoiding EXP-004's late two-pass SAM, whose dose
would vary if architecture throughput changed. EXP-009 already showed that a
nominally lightweight channel mechanism can fail this benchmark through extra
launches: four SE paths added 20.7% median step latency. The new candidate keeps
the existing dense-convolution count and control flow instead of adding a side
path.

The parent's first stage allocates two expensive-resolution residual units to
low-level 64-channel features and only two units to 256-channel semantic
features. Moving one unit late has three coupled effects:

- it adds 1.107M parameters where the representation is most class-specific;
- its two 3x3 convolutions add 16 pixels to final receptive-field diameter,
  versus only 4 pixels for the removed early block;
- it reduces BN/ReLU activation work because a 256x8x8 tensor has one quarter
  as many elements as a 64x32x32 tensor.

The PyramidNet paper establishes channel allocation as a first-class CIFAR
architecture variable rather than a fixed stage convention. Gradually Updated
Networks similarly motivates reorganizing existing convolutional computation
instead of adding nominally cheap auxiliary paths. ResNeXt supports comparing
representation dimensions at matched complexity, though this proposal avoids
grouped convolution because its small-resolution throughput is uncertain.
None of these papers directly validates 1-2-3 WRN staging; they support the
allocation principle, while the exact configuration remains exploratory.

## Exact Architecture

Keep `PreActWideBlock` unchanged. Replace the hard-coded 2-2-2 block list with:

```python
STAGE_BLOCKS = (1, 2, 3)

block_specs = [
    (16, 64, 1),
    (64, 128, 2),
    (128, 128, 1),
    (128, 256, 2),
    (256, 256, 1),
    (256, 256, 1),
]
```

Equivalently, generate this list from stage widths and `STAGE_BLOCKS`, but the
resolved sequence must match exactly. The first block of stages 2 and 3 keeps
stride 2; all others use stride 1. Learned 1x1 projections remain only on the
three shape-changing blocks: 16-to-64, 64-to-128, and 128-to-256.

Retain the six linearly increasing base drop probabilities by global block
index: `MAX_DROP_PATH * (index + 1) / 6`, followed by the existing time-based
late annealing. This assigns the three stage-3 blocks the deepest rates 0.0533,
0.0667, and 0.08. Do not preserve stage-specific rates from 2-2-2; depth-based
stochastic survival is part of the existing recipe.

Keep stem Conv(3,16,3), final BN(256), ReLU, global average pool, and
Linear(256,10). Keep pre-activation ordering, Kaiming initialization, identity
addition, and no post-addition ReLU. Log
`architecture=PreActWideResNet stage_blocks=1-2-3 widths=64-128-256`.

No optimizer or training-loop change is proposed. In particular, this EXP-002
fork does **not** add SAM.

## Exact Complexity

Counts include convolution/linear weights, BatchNorm affine parameters, and
classifier bias. MACs include Conv/Linear operations for one 32x32 image.

| Quantity | EXP-002 2-2-2 | Candidate 1-2-3 | Delta |
|---|---:|---:|---:|
| Residual blocks | 6 | 6 | 0 |
| 3x3 convolutions | 12 | 12 | 0 |
| Projection 1x1 convolutions | 3 | 3 | 0 |
| BN layers in blocks | 12 | 12 | 0 |
| Drop-path draws | 6 | 6 | 0 |
| Parameters | 2,748,890 | 3,855,578 | +1,106,688 (+40.3%) |
| Conv/Linear MACs per image | 392,612,352 | 392,612,352 | exactly 0 |

The removed 64-channel same-width block has 73,984 parameters; the added
256-channel block has 1,180,672. Each contributes 75,497,472 convolution MACs
per image. The candidate reduces per-image BN/ReLU tensor elements by roughly
98,304 because two normalization/activation points move from 65,536-element
feature maps to 16,384-element maps.

Parameter, gradient, and Nesterov-momentum storage increase by roughly 13 MiB
combined in FP32. Conversely, saved early-stage activations are larger than the
added late-stage activations. Peak VRAM should stay near or below the parent's
1,178.9 MiB and is nowhere near the H20 limit.

## Training Recipe Preservation

Keep every EXP-002 setting:

- batch 256, BF16 autocast, channels-last model and inputs;
- SGD with momentum 0.9, Nesterov, weight decay `1e-4`;
- peak LR 0.2, 5% time warmup, cosine decay to ratio 0.01;
- random crop/flip and evaluator-compatible normalization;
- CutMix probability 0.5, alpha 1.0, cutoff progress 0.75;
- dedicated seed-42 CutMix CPU/CUDA generators;
- global seed 42 and one evaluation per natural epoch;
- 300-second charged training and 600-second outer timeout.

All forward/backward work remains between `t0` and CUDA synchronization. The
time-indexed schedule remains valid even if the candidate completes a slightly
different number of steps. The architecture uses the same six stochastic-depth
draw sites and no new randomness. Model initialization necessarily consumes a
different number of CPU random values because tensor shapes differ; this is a
normal consequence of the genuine architecture change, not seed selection.

## Expected H20 Behavior

The candidate should be within a few percent of parent latency and may be
faster. Both execute the same number of dense convolutions and identical MACs,
but the moved block changes kernel shape from 64-channel 32x32 to 256-channel
8x8. Larger channel dimensions may use BF16 tensor hardware better, while the
smaller spatial grid reduces normalization/activation traffic. Conversely,
8x8 kernels may expose less parallel spatial work, and 40% more weights increase
cache traffic. Parameter count alone cannot resolve this.

At parent throughput, 300 seconds yields 27,950 steps and 144 epochs. A 5%
latency penalty projects about 26,620 steps and 136 epochs, still substantial
but potentially enough lost augmentation exposure to offset the capacity gain.
The preflight therefore bounds latency before any test-set metric run.

## Parent-Relative GPU-0 Preflight

Run the committed EXP-002 parent and candidate in the same standalone harness
on physical GPU 0. Use batch 256, BF16 autocast, channels-last tensors, the
production forward/loss/backward/Nesterov update, and synchronization. Do not
compile either arm. Use fixed synthetic inputs/targets so data loading and
CutMix do not obscure model latency; those costs are identical in the full run.

1. Warm each arm for at least 50 iterations.
2. Measure at least five alternating, randomized-order rounds of 200 iterations
   per arm to control thermal/order drift.
3. Report per-round median and p90 step latency, overall paired candidate/parent
   ratios, peak allocated VRAM, finite loss, and parameter count.
4. Separately time at least 200 evaluation forwards per arm at batch 256.

Proceed only when all parent-relative criteria pass:

- candidate median training latency is at most `1.05 * parent median`;
- candidate p90 latency is at most `1.08 * parent p90`;
- `27,950 / median_ratio >= 26,500` projected optimizer steps;
- candidate evaluation latency projects total runtime below 600 seconds;
- no OOM, nonfinite loss/gradient, unsupported kernel, or parameter-count error.

These thresholds are ratios to the measured parent in the same harness. There
is no absolute images/second floor, avoiding the EXP-008 error where a gate
could reject the measured parent itself. If preflight fails, reject 1-2-3
without testing another depth allocation against accuracy in this experiment.

## Expected Effect and Falsification

Formal improvement over parent EXP-002 requires at least 95.33%; exceeding the
global best requires more than 95.40%. The stronger hypothesis is:

> Reallocating one equal-MAC block from stage 1 to stage 3 will reach
> `best_test_acc` of 95.53-95.80% (+0.30 to +0.57 over EXP-002), retain at least
> 26,500 steps, and remain below 1.3 GiB peak VRAM.

The expected effect comes from greater late semantic capacity and receptive
field at unchanged major-convolution work, not from extra training compute.
Run one seed-42 metric experiment after preflight. Below 95.33% is a tree
no-improvement. A result of 95.33-95.52 formally improves the parent but
falsifies the preregistered detectable >=0.30-point architecture hypothesis.
Do not choose 1-1-4, 2-1-3, or a width adjustment after seeing this test result.

## Risks

- Removing the second high-resolution block may weaken local edge/texture
  composition before the first downsample; later capacity cannot recover lost
  spatial detail.
- The 40% parameter increase can overfit despite equal MACs. Parent CutMix and
  drop path mitigate this, but their strengths were tuned implicitly for 2-2-2.
- The deepest new block has a high drop rate, reducing its effective training
  dose during the first 75% and potentially wasting added capacity.
- Equal MACs can hide slower 8x8 kernels, higher weight traffic, or different
  cuDNN algorithm selection. The paired preflight is mandatory.
- A larger receptive field is not automatically useful on 32x32 images; 65
  pixels exceeds the image extent and may overemphasize global context.
- Shared-parameter initialization cannot be bit-identical because block shapes
  and RNG consumption change. Seed 42 remains fixed and no reroll is allowed.
- Protocol noise is 0.14-0.29 points. A marginal single-run delta should not be
  attributed confidently even if it clears the 0.10 formal gate.

## Verification

1. Unit-check resolved block specs, stage output shapes, three projection
   strides, six global drop rates, and `num_params == 3_855_578`.
2. Instrument Conv2d modules or use static formulas to verify twelve 3x3 and
   three projection calls and exactly `392_612_352` Conv/Linear MACs per image
   for both parent and candidate.
3. Run BF16/channels-last forward/backward smokes with finite logits, loss, and
   non-None finite gradients for every trainable parameter.
4. Pass the parent-relative GPU-0 latency/evaluation preflight above.
5. Launch once with
   `timeout 600s env CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`.
6. Confirm physical GPU 0 is the 97,871 MiB H20, charged time is approximately
   300 seconds, total time is below 600 seconds, CutMix exposure remains near
   0.5 before progress 0.75, validation occurs at most once per epoch, the
   summary is complete, and metric/step hypotheses are evaluated separately.
7. Verify only `train.py` changed, `prepare.py` and evaluator remained untouched,
   no dependency or seed change occurred, and transient logs are removed.

