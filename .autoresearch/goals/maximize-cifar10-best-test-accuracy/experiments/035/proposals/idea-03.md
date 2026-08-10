# Proposal: Reflection-Padded Strong and Weak Random Crops

## Summary

Change the accepted CIFAR-10 random crop in both training phases from implicit
constant-zero padding to reflection padding:

```python
transforms.RandomCrop(32, padding=4, padding_mode="reflect")
```

Keep the crop size and four-pixel width literal, and preserve the accepted
horizontal flip, strong-phase RandAugment N1/M7, 50% alpha-1 CutMix, weak-tail
switch at 80%, width-2 postactivation ResNet-20, initialization, SGD, learning
rate schedule, batch size, loader lifecycle, seed, and evaluator exactly. The
only production diff should be adding `padding_mode="reflect"` to the two
existing `RandomCrop` constructors in `train.py`.

The hypothesis is that zero padding makes crop displacement partially
identifiable from an artificial black band. Reflection padding replaces that
band with continued local image texture, so translated views carry less of a
crop-position shortcut and more boundary content. This could improve
translation invariance and reduce the train/test boundary mismatch without
discarding pixels or changing targets. The formal prediction is seed-42
`best_test_acc >= 94.25%` versus the 94.15% frontier at `7c1e7d8`; the point
prediction is only 94.27%. That narrow prediction is deliberate: the mechanism
is plausible, but direct literature support and expected effect size are weak.

## Diagnosis and Border-Distribution Mechanism

The accepted EXP010 recipe already has strong generalization components. It
holds N1/M7 RandAugment and 50% CutMix through 80% of counted training, reaches
89.73% at the strong-to-weak switch, then uses hard crop/flip views under a
quenched cosine tail to reach 94.15%. Later experiments repeatedly found that
more target mixing, unlabeled deletion, early weakening of regularization, or
more weak-tail learning-rate amplitude either deepened strong-phase underfit or
worsened final generalization. Reflection padding is therefore proposed as a
change in the *boundary prior* of the existing crop, not as more augmentation
strength and not as another optimization-path intervention.

With `padding=4`, torchvision first forms a 40x40 image and uniformly chooses a
32x32 window whose top and left offsets are integers from 0 through 8. Only the
central `(4, 4)` offset contains no padded pixels, so 80 of the 81 possible crop
positions touch padding. In one dimension the number of padded columns is
`abs(offset - 4)`, with expectation `20/9 = 2.22`. Across two independent crop
coordinates, the expected fraction of the output occupied by padding is
approximately:

```text
1 - ((32 - 20/9)^2 / 32^2) = 13.41%
```

Thus this is a compact two-keyword code change but not a tiny pixel
intervention. On a typical crop it changes roughly one eighth of the pixels.
Under constant padding those pixels are RGB zero before tensor conversion; after
the accepted channel-mean subtraction and unit standard deviation they become a
fixed negative-color border. Border width also encodes crop displacement. A
model can use that high-contrast, spatially simple signal to distinguish
translation directions or to suppress features near the artificial frame.

Torchvision reflection padding mirrors pixels without repeating the edge pixel.
It keeps local color/texture energy near the boundary and removes the fixed
black-band cue. The intended generalization mechanism is therefore:

1. crop position becomes less trivially recoverable from padding value;
2. edge-touching object evidence is retained as reflected texture rather than
   replaced with a constant;
3. the weak hard-label tail trains on views whose low-level statistics are
   closer to unpadded evaluation images than constant black frames are; and
4. CutMix donor regions that happen to include a crop boundary carry image
   texture rather than an easy synthetic zero strip.

The fourth point is secondary. CutMix still pastes the same sampled rectangle
and uses the same area-derived target. Reflection does not change class-bearing
area semantics, and must not be paired with a CutMix probability/alpha change.
RandAugment remains after the crop, so geometric RandAugment operations can
still introduce their own default-fill pixels. This proposal removes only the
padding artifact created by `RandomCrop`; it does not promise artifact-free
strong views.

## Distinction From Failed Data-Policy Changes

This proposal does not retry the mechanisms already rejected locally:

- EXP006 and EXP033 removed source information with unlabeled constant/mean-fill
  occlusion. Reflection padding preserves all original image pixels visible in
  the crop and fills missing context with mirrored source texture; it neither
  deletes an interior region nor introduces label-agnostic holes.
- EXP011 increased the fraction of soft CutMix targets and over-regularized the
  short strong phase. This proposal retains exactly 50% CutMix probability,
  alpha 1, and every target.
- EXP026 replaced half of accepted CutMix events with Mixup and worsened switch
  fit. Reflection does not interpolate images or labels.
- EXP005 and EXP027 changed when strong components were removed. This proposal
  preserves the simultaneous augmentation/objective/LR transition at 80%.

The distinction is mechanistic, not an assurance of success. Reflection changes
about 13.4% of crop pixels on average in *both* phases, so it can still change
task difficulty materially. Mirroring can duplicate a partial object, create
unnatural bilateral structure, or turn a semantic fragment at an image edge
into stronger repeated evidence. Zero padding may itself be useful
regularization by explicitly representing missing context. Because CIFAR-10
objects are small and often touch the source boundary, these risks are not
academic. A regression would retire this exact two-phase reflection policy, not
justify tuning edge/symmetric modes or applying reflection only to one phase
after seeing the score.

## Evidence Boundary

The original CIFAR ResNet recipe reports padding each side by four pixels and
sampling a random 32x32 crop, which supports retaining the crop geometry but
does not establish that reflection is better than constant padding:
<https://arxiv.org/abs/1512.03385>. The repository README likewise treats
four-pixel padding plus crop/flip as the baseline, without evidence for a
padding-mode substitution.

The official torchvision `RandomCrop` documentation defines constant padding as
the default and reflection as mirroring without repeating the boundary value:
<https://docs.pytorch.org/vision/0.24/generated/torchvision.transforms.RandomCrop.html>.
That is implementation evidence only, not an accuracy result. Inspection of the
installed torchvision 0.24.1 source confirms that the PIL constant path uses
`ImageOps.expand`, while nonconstant PIL modes convert through NumPy and call
`np.pad`. It also confirms that `padding_mode` changes deterministic padding but
not `RandomCrop.get_params` or the number of crop RNG draws.

There is no direct, high-signal result in the accumulated goal knowledge base
showing that constant-to-reflection padding clears even 0.10 points for a
width-2 CIFAR ResNet-20 already trained with N1/M7 and CutMix under a 300-second
horizon. Reflection is common as an image-boundary convention, but broad usage
is not causal evidence for this recipe. The strongest support is the local
distribution argument above, and that argument cuts both ways: removing a
shortcut may improve invariance, or it may replace useful regularization with
duplicated border content. This should be scored as a low-evidence,
low-implementation-risk experiment with a likely effect near the ten-image
acceptance threshold.

## Exact Proposed Change and Frozen Scope

Modify exactly these two accepted transforms:

```python
weak_train_tf = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)

strong_train_tf = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=1, magnitude=7),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)
```

Do not change transform order. In particular, do not move crop after
RandAugment, add a fill value, switch to `symmetric` or `edge`, convert the
pipeline to tensor/v2 transforms, or apply reflection to only the strong or
weak loader. Preserve:

- crop size 32, padding width 4, horizontal-flip probability and normalization;
- N1/M7 RandAugment, its operation space/interpolation/fill, and its strong-only
  lifetime;
- CutMix alpha 1, probability 0.5, collate implementation, and strong-only
  lifetime;
- width-2 ResNet-20, all weights and initialization, global-average classifier,
  and FP32 execution;
- batch 128, shuffle/drop-last/pinning, eight persistent forkserver workers,
  strong-loader shutdown and weak-loader construction;
- LR 0.1 through 80%, weak-tail start 0.01, cosine floor, ordinary momentum,
  all-parameter `1e-4` decay, and 300 counted seconds;
- seed 42, checkpoint cadence, dense-tail evaluation, and the untouched fixed
  `Eval.evaluate()` implementation.

Only tracked `train.py` may change. Do not alter `prepare.py`, dependency files,
or evaluator/test transforms.

## RNG Semantics and Exact-Corpus Limitations

At the torchvision transform level this candidate should be RNG-neutral.
Padding itself is deterministic, the padded image remains 40x40 in both arms,
and `RandomCrop.get_params` draws the same top/left coordinates. With the same
incoming CPU RNG state, constant and reflection arms should leave the same
outgoing state; subsequent horizontal-flip and RandAugment parameter draws
should therefore align. CutMix consumes the corresponding worker-side state and
should retain the same apply decision, permutation, rectangle, and soft targets
when replayed from an identical state. Production still uses one fixed seed 42,
not a favorable crop sequence.

That reasoning is not permission to compare two freshly spawned DataLoaders and
call them an exact pair. EXP019/021/026 established that fresh forkserver
processes do not reliably reproduce post-transform batches from seed alone.
Worker scheduling, iterator construction, and seed assignment can make a
same-seed loader comparison non-replayable. Moreover, a data-policy candidate
cannot use one identical image tensor for both arms: doing so would erase the
intervention being tested.

Before any trajectory assertion, build and persist a paired real-CIFAR corpus in
one controlled process:

1. preregister training indices/order and a finite sequence of per-sample RNG
   states; do not select images or offsets after observing model behavior;
2. for every source image, apply the accepted and reflection transforms from
   the same saved incoming RNG state and assert their outgoing RNG states are
   identical;
3. apply the same saved CutMix RNG state to corresponding accepted/candidate
   batches, assert target tensors and outgoing RNG states are bitwise equal,
   and preserve both resulting input tensors rather than regenerating them;
4. record hashes, hard/soft batch counts, crop-offset counts, CutMix counts, and
   the fraction and magnitude of changed pixels before training; and
5. train same-initialization control/candidate models on their corresponding
   serialized streams.

The corpus controls sample order and stochastic decisions, but the two input
tensors must differ at reflected boundaries. Therefore loss/logit/gradient
parity is neither expected nor a validity criterion. The corpus also cannot
prove that the production forkserver trajectory will match it; it is a bounded
mechanism/safety screen. Do not reuse EXP022/028 post-transform corpora as the
sole evidence because those tensors already contain constant-padded crops and
cannot be transformed into the counterfactual reflection views.

## Mandatory Semantic and Trajectory Preflight

Run these checks before throughput or production:

### 1. Constructor and differential audit

- Verify both production `RandomCrop` objects report size 32, padding 4, and
  `padding_mode == "reflect"`; all other transform reprs remain accepted.
- On preregistered CIFAR training images, exhaust all 81 crop offsets without
  flip/RandAugment/CutMix. The center `(4,4)` outputs must be bitwise equal.
  Every noncenter crop may differ only in pixels sourced from padding before
  later transforms. Confirm output shape, dtype, normalization, and labels.
- Snapshot and compare CPU RNG state around matched constant/reflection weak and
  strong transforms. Require identical outgoing states and aligned crop, flip,
  RandAugment, CutMix, and target metadata.
- Confirm the empirical mean changed-pixel area is consistent with the
  preregistered crop plan and report it. Do not impose 13.41% exactly on a finite
  random corpus; that value is the uniform-offset expectation.

### 2. Exact paired trajectory

Use at least 200 persisted production-distribution strong batches and 64 weak
hard-label batches, corresponding to the established exact-corpus scale. Start
both models from bitwise-identical parameters and optimizer state under the
accepted backend flags. Run the accepted LR/momentum/decay recurrence on the
paired streams and record at each preregistered checkpoint:

- finite loss, logits, parameters, gradients, optimizer buffers, and BN state;
- predicted-class shares for candidate and control;
- whole-model and per-stage logit, gradient, update, and update/parameter norms;
- strong and weak terminal loss EMA ratios; and
- BN running-variance minima and `num_batches_tracked` parity.

Veto production for candidate-only `>95%` one-class concentration, nonfinite
state, BN counter mismatch, or a gross candidate/control update/logit excursion
using the same conservative ratio framework as recent exact-corpus gates. Do
not require close loss or norm parity merely because this is a two-line diff;
the candidate intentionally changes inputs. Conversely, a lower preflight loss
cannot override a concentration or update-spike veto. A veto retires this exact
policy without phase-only, padding-width, or mode rescue inside EXP035.

## CPU Loader Throughput and Paired Timing Gates

Reflection has no GPU model cost, but it is not guaranteed to be free. In the
installed PIL path, constant padding calls `ImageOps.expand`; reflection converts
the image to a NumPy array, executes `np.pad`, and reconstructs a PIL image for
every sample. That extra allocation/copy happens in DataLoader workers. Loader
work is mostly hidden by prefetch in the accepted run (system understanding
measures 0.145 ms median and 0.171 ms p95 iterator wait), but a slowdown can
empty the queue, extend uncounted wall time, and threaten the 600-second limit.
The weak loader is especially important because it lacks RandAugment, so padding
can be a larger fraction of host transform cost even though it runs for only the
last 20% of counted time.

Use alternating fresh-process control/candidate pairs with the exact production
DataLoader settings. Benchmark strong and weak pipelines separately. Warm the
workers and caches, separate first-batch and iterator-rollover latency, then
measure at least three full epochs per arm. Record sustained batches/s,
images/s, median wait, non-rollover p95 wait, first-batch latency, worker health,
and epoch variance. Do not use the absolute rollover-inclusive p95 gate rejected
by EXP033.

Proceed only if all are true:

- each candidate pipeline sustains at least 95% of its paired control batch
  rate;
- each candidate sustains at least 1.25 times the contemporaneous production
  demand inferred from control counted-step throughput, preserving queue margin;
- candidate non-rollover p95 wait is no more than 1.5 times paired control and
  shows no repeated starvation bursts;
- all workers finish cleanly with the expected batch count; and
- a conservative strong/weak-weighted projection remains below 540 seconds
  total, leaving 60 seconds before the hard timeout.

Then run alternating fresh-process short full-step timing with the accepted
model and optimizer, using the corresponding paired serialized batches. Require
candidate/control median synchronized GPU-step time no greater than 1.01 and
projected optimizer exposure at least 99% of control. The tensors have identical
shape/dtype and GPU code is unchanged, so a larger counted-step penalty is
evidence of an uncontrolled measurement/environment issue. Report wall-per-step
and loader wait separately; do not hide a host regression behind identical CUDA
time. Any failed timing gate blocks production. Do not raise worker count,
change multiprocessing mode, reduce evaluation, or move transforms to GPU as a
same-experiment rescue.

## Production Verification and Decision Rule

After all preflights pass:

1. Query the moving baseline and require it to remain 94.15% at `7c1e7d8` (or
   recompute the literal +0.10 gate if the integration frontier changed before
   launch).
2. Confirm one idle NVIDIA H20 with approximately 98 GB VRAM, no stale completed
   log, a clean tracked scope except the two `train.py` keyword additions, and
   passing compile/lint/format checks.
3. Launch exactly one fixed-seed production run as
   `uv run train.py > run.log 2>&1`; do not tee or reroll.
4. Monitor without streaming the full log, terminate at 600 seconds, and verify
   approximately 300 counted seconds, normal numeric summary, once-per-epoch
   maximum evaluation, unchanged evaluator cadence, and finite metrics.
5. Record best/final accuracy, final NLL, switch accuracy/loss, first weak-tail
   response, steps, epochs, training/total/startup seconds, peak VRAM, strong
   hard/soft counts, and all preflight throughput evidence.

Verdicts:

- **Improvement:** valid completion with `best_test_acc >= 94.25%` and every
  integrity constraint satisfied.
- **No improvement:** valid completion below 94.25%; restore constant padding
  and do not retry reflection in only one phase or with another seed.
- **Invalid:** semantic, exact-corpus, loader, or timing veto before production.
- **Crash:** runtime failure, nonfinite/missing summary, wrong hardware, timeout,
  evaluator/scope violation, or training-budget violation.

Switch accuracy and NLL are explanatory only. They cannot promote a subthreshold
run or reject a valid 94.25% result. CIFAR-10 test accuracy is quantized in 0.01
points, so the formal +0.10 gain is only ten examples. A bare threshold pass is
protocol-valid but weak causal evidence under one fixed seed and CUDA
nondeterminism; it must not trigger a confirmation reroll.

## Risk Assessment and Expected Value

Implementation effort is low, while evidence engineering is medium because the
change lives inside a multiprocessing stochastic pipeline. GPU memory and model
compute should be unchanged. Host cost may rise measurably due to NumPy-backed
PIL reflection, but the accepted loader has enough observed headroom that a
clean pass is plausible.

Accuracy upside is modest. The candidate preserves every validated component
and removes a conspicuous synthetic cue from most crops, so a 0.10-0.20 point
gain is possible. Against that, modern crop/RandAugment/CutMix already exposes
many boundary conditions, global-average pooling reduces spatial shortcut
pressure, and reflection creates its own unnatural duplicated texture. There is
no direct literature result establishing this substitution at the current
frontier. The most likely outcomes are statistical flatness or a small
regression; the proposal is worthwhile primarily because it tests a clean,
orthogonal boundary-distribution hypothesis with essentially no recurring GPU
cost.

The preregistered testable claim is: two-phase reflection padding will preserve
at least 99% of accepted optimizer exposure, avoid candidate-only early geometry
failure, maintain healthy strong-to-weak adaptation, and raise seed-42
`best_test_acc` from 94.15% to at least 94.25%. No adjacent padding mode, width,
phase scope, or seed may be selected after observing the result.
