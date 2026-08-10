# Proposal: Fixed 288-channel final-stage taper from EXP-004

## Summary

Widen only EXP-004's final 8x8 stage from 256 to 288 channels, changing the
six-block stage widths from `64/128/256` to `64/128/288`. The first four
residual blocks, spatial transitions, block count, and all training mechanisms
remain unchanged. This is one fixed `64/128/288` architecture, not a sweep and
not a conditional fallback to another width.

The intervention adds 32 channels to the two low-resolution blocks, final
BatchNorm, and affine classifier. It increases trainable parameters from
`2,748,890` to exactly `3,260,442` (`+511,552`, `+18.61%`) and Conv/Linear MACs
per image from `392,612,352` to exactly `425,315,136` (`+32,702,784`,
`1.0832954x`). It adds no module, forward, loss, stochastic decision, or kernel
launch. All new convolutional arithmetic occurs at 8x8.

Everything else remains EXP-004: batch 256, crop/flip, front-loaded CutMix,
six-block drop path, time-based cosine LR, Nesterov SGD, late clean-tail
period-two SAM at rho 0.05, BF16/channels-last, fixed seed 42, the 300-second
charged budget, and one live-model evaluation per epoch. There is no EMA on
this parent.

The formal hypothesis is that modest late semantic capacity can offset its
roughly 7-10% expected throughput cost and raise `best_test_acc` from 95.40% to
at least 95.50%. Stronger support would be `>=95.60%`, final accuracy close to
the best, and at least 23,200 optimizer steps. The plausible useful range is
about 95.50-95.75%; a result much above 95.8% would be surprising for this
modest package, while a valid result below 95.50% closes this exact taper.

## Why 288, and why this is not an EXP-014 retry

EXP-014 fixed its own stage-3 width at 320 and correctly stopped when the first
complete preflight measured a `1.160975x` weighted latency ratio, just above
its `1.15x` gate. It never queried accuracy. Its terminal report explicitly
left a separately preregistered 288-channel taper open. Consequently, selecting
288 here uses systems evidence rather than test feedback: no width-320 metric
exists to tune against, and EXP-021 commits to 288 before its own timing or
accuracy is observed.

Width 288 is a 12.5% increase in the final representation dimension, is a
multiple of 32, and retains dense tensor-friendly shapes on H20. Relative to
320, it removes 566,848 parameters and 36,241,728 MACs/image. Relative to the
parent, its 8.33% MAC increase is less than half EXP-014's 17.56% increase.
Linearly interpolating the *measured* EXP-014 latency increment by incremental
MACs predicts

```text
1 + (1.160975 - 1) * (0.0832954 / 0.1756045) = 1.07636x.
```

That is only a prior. H20 latency depends on tensor shapes, activation traffic,
workspace selection, and SAM's two-pass mix; the first fixed paired preflight
must measure it. A reasonable expectation is `1.07-1.10x`, corresponding to
roughly 23,200-23,900 updates from EXP-004's 25,560-update dose.

The architecture rationale is narrower than “more parameters help.” EXP-010
moved an equal-MAC block from the 32x32 stage to 8x8, ran 9.3% more updates,
and still scored 95.04%. That rejects deleting early local processing in favor
of late depth, but it leaves intact EXP-010's explicit avenue of preserving the
2-2-2 allocation while modestly widening only the final stage. The PyramidalNet
note supports channel allocation as a CIFAR representation variable, while not
directly validating this coarse final-stage taper. EXP-014 showed that the 320
implementation was operationally sound and memory-cheap; it rejected only
that width's predeclared cost envelope, not the capacity hypothesis.

## Exact implementation

Keep `PreActWideBlock` unchanged. Change only these architecture shapes in
`PreActWideResNet`:

```python
block_specs = [
    (16, 64, 1),
    (64, 64, 1),
    (64, 128, 2),
    (128, 128, 1),
    (128, 288, 2),
    (288, 288, 1),
]
...
self.bn = nn.BatchNorm2d(288)
self.fc = nn.Linear(288, num_classes)
```

Update only truthful architecture metadata, for example
`stage_widths=64,128,288` and `PreAct WRN-16-[4,4,4.5]`; the computed parameter
count remains authoritative. Do not add a width argument, runtime switch,
alternate 272/304/320 path, automatic fallback, or benchmark-only branch to
production. Optional terminal train-loss and evaluation-progress diagnostics
may reuse already computed values outside the charged per-step interval, but
they must never select or stop the run.

Preserve exactly:

- data set, batch size, shuffle, worker, crop/flip, normalization, pinning, and
  dropped-last semantics;
- seed 42, dedicated CutMix CPU/CUDA generators, probability, box geometry,
  mixed-target weighting, and 75% cutoff;
- LR 0.2, warmup/minimum ratios, charged-time cosine progress, momentum 0.9,
  Nesterov, and weight decay `1e-4`;
- six depth-indexed drop probabilities, maximum 0.08, and final-quarter decay;
- SAM start 0.75, period 2, rho 0.05, global FP32 norm, RNG replay, second-pass
  BatchNorm suppression, exact restore, and one optimizer update;
- BF16 autocast, channels-last layout, timer boundaries, evaluator, epoch
  cadence, metric definition, and 600-second outer limit.

No LR, decay, drop-path, CutMix, or SAM compensation is allowed after width
changes. Such compensation would turn a controlled architecture test into a
multi-factor recipe search.

## Parameter and MAC accounting

Counts include convolution/linear weights, classifier bias, and BatchNorm
affine parameters.

| Component | EXP-004 | Width 288 |
|---|---:|---:|
| Stem | 432 | 432 |
| Blocks 1-4 | 646,432 | 646,432 |
| Block 5, 128 -> final width | 918,272 | 1,115,968 |
| Block 6, final width -> final width | 1,180,672 | 1,494,144 |
| Final BN + classifier | 3,082 | 3,466 |
| **Total parameters** | **2,748,890** | **3,260,442** |

The candidate's width-dependent parameter count can be independently checked
as `647,120 + 27*c^2 + 1,298*c + 10` at `c=288`.

The MAC convention counts one Conv/Linear multiply-accumulate as one MAC and
excludes BN, activations, pooling, and augmentation.

| Component | Width-288 MACs/image |
|---|---:|
| Stem | 442,368 |
| Block 1 | 48,234,496 |
| Block 2 | 75,497,472 |
| Block 3 | 58,720,256 |
| Block 4 | 75,497,472 |
| Block 5 | 71,368,704 |
| Block 6 | 95,551,488 |
| Classifier | 2,880 |
| **Total** | **425,315,136** |

The candidate adds about 24.3% parameters within the changed final-stage/tail
region but only 18.6% model-wide parameters and 8.33% MACs. Live weights,
gradients, momentum, and SAM snapshots add several copies of the extra
511,552 elements, but their raw storage is small relative to EXP-004's
1,190.5 MiB peak. Memory is not the expected constraint; kernel latency and
lost data exposure are.

## Initialization and RNG package effects

Retain the existing constructors and `_weights_init` exactly. Kaiming fan-in
initialization remains appropriate for each wider convolution; BN affine state
remains one/zero and classifier bias zero. Two candidate constructions after a
complete seed reset must be bitwise identical.

Do **not** require parent/candidate overlapping tensors or their
post-construction CPU RNG states to match. PyTorch module constructors consume
shape-dependent random draws, and the model-wide `apply(_weights_init)` consumes
another shape-dependent sequence. Enlarging blocks 5-6 therefore changes the
fixed-seed realization of the entire package, including common-shaped tensors
initialized later in traversal. Because DataLoader iteration begins only after
model construction, the shifted global CPU RNG state can also change sampler
and worker base seeds, hence crop/flip realizations and image order.

This is an unavoidable package-level confound under the frozen program, not a
seed reroll. Do not burn random draws, copy parent submatrices, introduce
per-layer seeds, preserve a parent data stream artificially, or run another
seed. Any of those would create a second initialization design. Dedicated
CutMix generators remain isolated and deterministic, and the architecture adds
no forward RNG calls: there are still six drop-path draws in the same order.
Cross-arm numerical equality is not expected, while candidate self-
determinism and within-candidate SAM replay are mandatory.

A narrow 95.50-95.59 pass must therefore be described as evidence for this
fixed width-288 seed-42 package, not clean causal proof that 32 extra channels
alone add that many points.

## Interaction with EXP-004's SAM tail

All wider parameters participate in EXP-004's existing global SAM gradient
norm and have matching preallocated restore snapshots. On an eligible even
late step, the candidate must still:

1. execute the ordinary forward/backward and update BN exactly once;
2. compute a finite positive global FP32 gradient norm;
3. copy every trainable parameter, perturb with total Euclidean norm 0.05;
4. restore CUDA RNG, disable BN running-stat tracking, and execute the second
   clean forward/backward with the same stochastic-depth masks;
5. restore all flags and parameters exactly, then apply one Nesterov update.

Width changes how the fixed-radius perturbation is distributed across tensors
and will likely reduce `||epsilon||/||w||` modestly. EXP-014 measured
`0.0006286761` for the width-256 parent and `0.0005873253` at width 320; width
288 might lie near `0.00060`, but that is diagnostic only. Record parent and
candidate values in the paired smoke. Do not rescale rho, use layerwise SAM,
or gate on the relative value. Exact perturbation norm, finite gradients, one
BN update, RNG replay, and exact restoration are integrity gates.

The candidate keeps the wall-clock 75% boundary, so slower steps reduce counts
without shortening the clean-tail *time*. Under the interpolation prior,
EXP-004's 20,662 early batches project to about 19,196 and its 4,898 late SAM-
eligible batches to about 4,551, with roughly 2,275 SAM pulses. At the hard
`1.10x` latency ceiling the corresponding arithmetic is about 18,784 early,
4,453 late eligible, and 2,226 pulses. Real counts depend on path-specific
latency and boundary overshoot; audit rather than force them.

## Deterministic correctness checks

Before any gate vector or metric result:

1. Materialize exact parent commit `1a8d0de` under `/tmp`, hash-check it, and
   import parent/candidate without invoking `main`.
2. Reconcile the six block tuples, strides `1,1,2,1,2,1`, three projection
   shortcuts, 16 convolutions, unchanged module count/order, tail width 288,
   `(256,10)` output, `3,260,442` parameters, and `425,315,136` MACs/image.
   Shape changes must be confined to blocks 5-6, final BN, and classifier.
3. Reset seed and prove candidate self-determinism. Record cross-arm common-
   tensor and post-construction RNG comparisons without treating mismatch as a
   failure.
4. Run CPU FP32 and candidate-only physical-GPU-0 BF16/channels-last forward,
   backward, and Nesterov smokes. Require finite logits/loss/BN state and finite
   nonzero gradients for every trainable tensor.
5. Exercise deterministic CutMix and require six active-training drop-path
   draws and zero eval/terminal-drop draws.
6. Exercise one ordinary and one production-faithful SAM update, requiring
   perturbation norm 0.05 within numerical tolerance, identical replayed CUDA
   RNG masks, one BN update, exact weight restore before the optimizer step,
   and a real finite update afterward.
7. Require complete snapshot inventory, no aliases, preserved optimizer
   identities, and no allocation growth after repeated steady-state SAM calls.
8. Run syntax/diff checks and require `train.py` as the only tracked change;
   evaluator, `prepare.py`, dependencies, and all protocol files are immutable.

Straightforward harness or implementation exceptions may be corrected only
before a complete numeric gate vector exists. There is at most one production
code repair after the preflight script is frozen and before it emits any gate
number. A correctness failure after that bounded repair is a pre-metric failed
leaf, not permission to redesign the taper.

## Fixed, accuracy-blind preflight

Confirm physical GPU 0 is the approximately 97,871 MiB NVIDIA H20 and expose
exactly one visible device. Monkeypatch both imported evaluators to raise on
any call. Use parent and candidate in one process, five warmed alternating-
order rounds, fixed synthetic tensors, and scripted stochastic decisions.

Each round must cover at least 100 early ordinary, 100 early CutMix, 40 late
ordinary, and 40 late SAM steps per arm. Include input movement, autocast,
loss, backward, global SAM norm/snapshot/restore, Nesterov, and synchronization.
Measure evaluation-mode forward latency separately without iterating a test
loader. Weight charged paths by EXP-004's observed production counts:

```text
early ordinary: 10,409 / 25,560
early CutMix:   10,253 / 25,560
late ordinary:   2,449 / 25,560
late SAM:        2,449 / 25,560
```

An optional shared-batch 200-step real-CIFAR conditioning trace may record
loss, activation norms, logit norms, and stagewise gradient norms at fixed
steps. It is report-only except for nonfinite/collapse/integrity failure and
cannot select the width, tune the recipe, or authorize a repeat.

The first complete valid preflight is decisive. Proceed to the one metric run
only if all fixed gates pass:

- parent weighted-round drift `(max-min)/median <=0.03`;
- paired-ratio `MAD/median <=0.01` and every ratio finite;
- median candidate/parent weighted charged-latency ratio `<=1.10`;
- maximum paired ratio `<=1.13`;
- projected steps `25,560 / median_ratio >=23,200` and projected complete
  natural epochs `floor(projected_steps/195) >=118`;
- conservative end-to-end projection using measured evaluation latency
  `<550s`, comfortably inside the hard 600-second timeout;
- candidate-only peak allocation `<2,048 MiB` and zero steady-state allocation
  growth beyond a fixed small diagnostic tolerance;
- exact source/shape/count/RNG/SAM/BN/restore/optimizer checks and no evaluator
  call, nonfinite value, or collapse.

These gates are fixed with no fallback. A ratio of 1.101, an unstable parent
vector, or any other valid numeric failure ends EXP-021's width-288 candidate
before accuracy. Do not rerun timing, reduce to 272, increase to 304/320,
change batch size, remove SAM, alter a threshold, or inspect test accuracy.
Only a Python/shell exception, missing file, or demonstrably malformed
assertion before any numeric gate vector is emitted can be repaired.

## One metric run and diagnostics

After all checks pass, run exactly once:

```bash
timeout 600s env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 uv run train.py > run.log 2>&1
```

Do not stop on finite loss, intermediate test accuracy, or apparent dose. Abort
only on process failure, CUDA/OOM, a nonfinite/integrity error, 120 seconds with
no process/GPU/log progress, or the outer timeout. Never rerun the seed or try a
neighbor width after seeing accuracy.

Require exit 0, charged time `[299.5,301.0]s`, total time `<600s`, exactly one
evaluation per completed epoch, complete summary, parameter count `3,260,442`,
no error signature, and only `train.py` changed. Record:

- best/final accuracy, final loss, best epoch, evaluation count, and final-16
  live-model accuracy mean/range when available;
- total steps, epochs, realized early ordinary/CutMix and late ordinary/SAM
  counts, first SAM step/progress, and applied/eligible ratios;
- terminal debiased train loss and stagewise activation/gradient diagnostics
  already preregistered outside the decision path;
- peak VRAM, charged/total/startup seconds, and realized versus preflight
  latency/dose projection;
- candidate SAM perturbation norm, `||epsilon||/||w||`, complete snapshot
  coverage, RNG replay, BN suppression, and restore-failure count.

Mechanism support requires at least 23,200 steps, at least 4,400 late eligible
batches, SAM application ratio within the exact period-two arithmetic, finite
state throughout, `best_test_acc >=95.60%`, and a non-spiky endpoint (final no
more than 0.15 points below best). Dose or stability shortfall weakens the
interpretation but does not override the formal metric verdict.

## Decision rule, ceiling, and falsification

- `best_test_acc >=95.50%` with all hard constraints satisfied is a formal
  improvement over EXP-004.
- `95.50-95.59%` is a valid tree improvement but below the stronger capacity
  hypothesis; report it as narrow package-level evidence given RNG divergence.
- `>=95.60%`, adequate dose, and a stable endpoint support the claim that the
  extra low-resolution capacity improves the deployed representation.
- A valid complete result below 95.50% is `no-improvement` and forbids another
  width, LR, decay, rho, cadence, or seed in this experiment.
- A preflight failure, crash, timeout, extra evaluation, wrong device/scope,
  nonfinite state, SAM/BN/RNG/restore failure, or incomplete summary is encoded
  according to the loop's integrity-first rules and never converted into an
  accuracy claim.

The realistic ceiling is constrained. EXP-004 already reaches 95.40 with
final equal to best; EXP-011's orthogonal EMA package lifts the same width-256
lineage to 95.61. Width 288 could plausibly supply a 0.10-0.35-point stable
lift, yielding roughly 95.50-95.75 and leaving a later clean composition with
EMA available. There is little evidence for expecting `>=96%`: EXP-010 showed
that late capacity redistribution alone is not reliably beneficial, the
candidate loses roughly 1,600-2,300 updates, and no paper establishes this
exact taper under CutMix plus SAM. The added capacity may instead increase
overfitting or dilute the fixed global SAM radius.

The principal causal risks are therefore:

- fewer images, updates, epochs, and max-selection checkpoints outweigh the
  representation gain;
- the baseline is data/objective-limited rather than final-stage-capacity-
  limited;
- fixed per-weight decay and global rho change their package-level effect as
  parameter count grows;
- width-specific H20 kernels cost more than MAC interpolation predicts;
- shape-dependent initialization and later worker/sampler RNG create a
  different deterministic training realization;
- improved peak accuracy is checkpoint noise rather than a stable endpoint.

This is still a clean architecture experiment: it preserves both early blocks
that EXP-010 removed, changes only existing low-resolution tensor widths, has a
strongly bounded systems cost from EXP-014, and leaves the validated CutMix/SAM
recipe untouched. Its implementation effort is low, memory risk low, latency
risk low-to-medium, and accuracy risk medium-high. A successful stable result
would justify a separate later composition with EXP-011's EMA; EMA must not be
added here because doing so would obscure whether width itself earned the gain.
