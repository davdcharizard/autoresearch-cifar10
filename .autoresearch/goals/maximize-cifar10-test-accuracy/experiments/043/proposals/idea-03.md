# Proposal: Restrained Early 8x8 Cutout After Mixup

## Recommendation

Advance only as a low-confidence, one-score closure experiment if the other
EXP043 proposals do not offer a better-supported mechanism. Retain accepted
commit `a7c42dc` in full: the `(2,2,3)` WRN, bias-free scale-0.1 pooled
`128 -> 64 -> 128` residual MLP, uniform GAP, ordinary affine classifier,
sole refined-path cross-entropy, batch-shared alpha-0.2 mixup, worker-safe
N1/M5 RandAugment, optimizer, global counted-time cosine, seed 42, and clean
tail. Only during the existing mixup branch, erase one independently located
`8 x 8` square from every already-mixed image and fill it with normalized
zero. Stop Cutout on the same `use_mixup` decision at 65% counted time.

This proposal was first developed in EXP041 and remains unscored. Its prior is
weak rather than newly strengthened. EXP003 replaced mixup with shared CutMix
rectangles averaging 31% pasted area and scored 93.72% at normal exposure.
EXP006's broad early residual dropout lost 0.55 points, and EXP030's isolated
p=0.05 drop-path on the added stage-3 block lost 0.41 points with worse loss.
The exact Cutout treatment is materially different because it is
label-preserving, input-space, per-example, only 6.25% area, additive to rather
than a replacement for accepted mixup, and absent from the hard-label tail.
Those distinctions justify at most one restrained test; they do not overturn
the local evidence that additive masking is more likely to harm than help.

EXP041 and EXP042 further lower confidence in broad refinement narratives:
direct-path auxiliary CE scored 94.26% and exact-neutral content attention
pooling scored 93.80%, both at at least 127 passes with worse loss. They do not
test occlusion, but they show that plausible low-cost representation controls
are not moving this 94.48% frontier. Cutout must therefore stand on the narrow
claim that small missing-evidence invariance is orthogonal to the accepted
global augmentation and interpolation stack, not on an assumption that any
additional regularizer is useful.

## Exact Fixed Treatment

Use exactly one square hole of side 8 on every early sample. The hole removes
64 of 1,024 pixels, or 6.25%, much less information than EXP003's mean 31%
donor area and less than the common 16x16 CIFAR Cutout treatment. The mild
setting is fixed because Cutout is stacked with both accepted RandAugment and
mixup. There is no application probability, sampled scale, aspect-ratio
distribution, boundary clipping, or magnitude sweep.

For an input tensor `x` of shape `B x C x 32 x 32`, independently draw

```text
top_i  ~ Uniform{0, ..., 24}
left_i ~ Uniform{0, ..., 24}
```

and set `x[i, :, top_i:top_i+8, left_i:left_i+8]` to floating-point zero.
Coordinates are independent across examples and shared across channels.
Sampling top-left coordinates guarantees exactly 64 erased pixels at corners
and in the interior; it avoids clipped-center severity changing by position.
Because accepted normalization subtracts the CIFAR mean and divides by unit
standard deviation, zero is the configured mean-color fill.

Apply the hole after the accepted `mixup_batch` returns and immediately before
`model(mixed_inputs)`. Thus each final mixed input contains one complete hole,
accepted interpolated targets and their weights remain exact, and no donor is
mutated. Before-mixup masking is not equivalent: two independently masked
donors can leave two partially filled artifacts after interpolation. Cutout
must run only inside `if use_mixup`; at `progress >= 0.65`, the entire accepted
hard-label path is exactly restored, with no coordinate draws, allocation, or
branch-specific state update. The accepted epoch-exhausted RandAugment switch
remains unchanged, including its short possible RandAugment-only interval.

Implement a pure vectorized helper that creates broadcast row and column
coordinates plus a `B x 1 x 32 x 32` Boolean erase mask and returns
`inputs.masked_fill(mask, 0.0)`. Do not loop over samples, mutate inputs in
place, cache masks, or use a batch-shared rectangle. Create one device-local
private `torch.Generator` after accepted model, optimizer, and mixup
distribution construction, seed it exactly once with `torch.initial_seed()`
(42), and pass it explicitly to the two coordinate draws. Reading the initial
seed and constructing/seeding the private generator must not consume or alter
global CPU or CUDA RNG. This is isolation from accepted mixup coefficients and
permutations, not a seed search.

## Mechanism and Falsifiable Hypothesis

The accepted model has enough compute exposure and low-resolution nonlinear
capacity but still ends at 94.45% accuracy and 0.2456 loss. Crop/flip and
RandAugment teach global geometric or photometric invariance, while mixup
teaches dense between-example interpolation. A small local hole asks for a
different property: class evidence should remain usable when one contiguous
region is missing. Placing the hole after mixup preserves the proven target
softness while preventing either constituent's strongest local patch from
dominating the mixed prediction. The final 35% clean tail then calibrates the
same inference graph on complete hard-label images.

The primary hypothesis is deliberately strict: per-example post-mixup 8x8
Cutout through exactly 65% counted time will retain at least 127 data passes
and increase fixed-seed `best_test_acc` from 94.48% to at least 94.58%. Final
accuracy at least 94.45% and final loss at most 0.2456 are corroboration only;
they cannot rescue a primary miss.

The better-supported counter-hypothesis is that the accepted early stack is
already calibrated and further information removal weakens representation
learning. Under that account, even a 6.25% hole compounds mixup and
RandAugment, reproducing the worse top-1 and loss seen with CutMix and feature
masking despite normal exposure. One normal-exposure miss closes additive
early Cutout/local erasing for this accepted stack. It does not establish that
occlusion augmentation is universally ineffective.

## Expected Benefit and Risks

Potential upside is a distinct input invariance with no model parameters,
unchanged labels, no inference cost, and only one small GPU-side mask/write
during the early window. It is cleaner than CutMix for this question because
erased area is not translated into potentially mismatched class mass and every
sample receives its own location.

The risks dominate the prior:

- a third simultaneous early regularizer may remove useful signal from an
  interaction already calibrated by EXP027;
- normalized-zero squares can become a synthetic mean-color cue rather than
  realistic occlusion;
- per-example locations add stochastic trajectory variance even though the
  accepted global RNG streams are isolated;
- 8x8 may be too mild to deliver +0.10, while increasing it after a miss would
  be unjustified adjacency tuning;
- Boolean-mask construction and a full input write occur inside counted time,
  where only 2.5356% overhead can be tolerated before accepted 130.304 passes
  project below 127;
- the sole fixed-seed score cannot estimate a seed-averaged effect.

## Production Scope

Only `train.py` may change. The diff is limited to:

1. add `CUTOUT_SIZE = 8` beside the augmentation constants;
2. add one vectorized `cutout_batch(inputs, size, generator)` helper;
3. construct and seed the private device generator after accepted
   model/optimizer/mixup-distribution setup without changing global RNG;
4. call the helper once on `mixed_inputs` after `mixup_batch` and before model
   forward in the existing `use_mixup` branch.

Do not change `EarlyRandAugment`, transforms, worker state, model graph,
parameter initialization or groups, loss weights, learning-rate curve, mixup
strength/duration, evaluation cadence, summary schema, `prepare.py`, or any
dependency. The accepted parameter count must remain 1,003,482.

## Semantic Preflight

Use an ignored evaluator-free harness with an independent
`git show a7c42dc:train.py` oracle. Print diagnostics before assertions. The
candidate may proceed only if all of the following hold:

- only the four production scopes above differ and `train.py` is the sole
  modified production file;
- initial model state keys, values, bytes, parameter count, buffers, optimizer
  groups/options/state, schedule, transforms, loader options, cutoff constants,
  evaluator cadence, and post-construction global CPU/CUDA RNG states match
  accepted exactly;
- all-one fixtures have exactly 64 erased spatial positions per example across
  all channels, no changed value outside the rectangle, coordinates in
  `[0,24]`, channel-shared geometry, and at least two locations in a fixed
  multi-example fixture;
- an independently coded per-example slicing oracle equals production bitwise
  on CPU and CUDA, including top-left `(0,0)`, bottom-right `(24,24)`, batch
  size one, noncontiguous input, dtype/device preservation, no source mutation,
  finite output, and finite gradients;
- saving and restoring the private generator reproduces coordinates and pixels
  bitwise, helper calls change only that generator, and global CPU/CUDA states
  remain exact;
- accepted and candidate mixup inputs, coefficient, permutation, targets, and
  post-mixup global RNG states are exact before Cutout; a call-order probe shows
  the sole mask occurs after mixing;
- below 65%, logits, loss, and at least one update differ nontrivially while
  remaining finite; at and above 65%, complete accepted/candidate hard-label
  steps, RNG, logits, loss, parameter updates, and optimizer state match
  bitwise from restored fixtures;
- worker-side RandAugment isolation and the exhausted-iterator transition stay
  accepted, with no clean-tail prefetch leakage;
- the fixed 300-second counted budget, no-more-than-once-per-epoch evaluation,
  and final-summary contract are unchanged.

Verifier-only defects may be corrected before timing. A semantic production
failure ends the candidate; it does not permit changing size, fill, ordering,
probability, location law, generator seed, cutoff, or kernel implementation.

## Timing and Exposure Gate

On one idle H20, compare accepted and candidate complete
production-equivalent steps in both early and hard regimes. Each body includes
H2D transfer, LR calculation and writes, zeroing, accepted mixup sampling and
permutation when active, candidate Cutout when active, full forward, exact
loss, finite guard, backward, coupled Nesterov update, and final CUDA
synchronization. Use at least 20 warmups and four counterbalanced windows of at
least 50 steps per arm and regime, with fresh deterministic fixtures per
replicate. Print all raw windows before assertions and require population CV
at most 5% for each arm/regime.

Compute

```text
retention =
    (0.65 / candidate_early_ms + 0.35 / candidate_hard_ms) /
    (0.65 / accepted_early_ms  + 0.35 / accepted_hard_ms)
projected_passes = 130.304 * retention
```

Proceed only if retention is at least `127 / 130.304 = 0.9746439096`,
projected passes are at least 127, and the candidate/accepted hard-step ratio
is within 1%, because Cutout must have an exact hard-tail bypass. Run a short
real-loader marker check for the unchanged worker-safe RandAugment transition;
the CPU pipeline itself is source-identical.

A stable timing miss closes this exact proposal without a scored run. Do not
rerun timing, lower the floor, make locations batch-shared, reduce the hole,
add a probability, move masking into workers, mutate in place, or cache masks
to rescue exposure.

## Sole Scored Run and Decision Contract

After both gates pass, reconfirm baseline 94.48% at `a7c42dc`, an idle single
NVIDIA H20, local CIFAR-10, frozen `prepare.py`, no stale `run.log`, and the
exact reviewed production diff. Run exactly once at seed 42:

```bash
timeout 600s uv run train.py > run.log 2>&1
```

Require exit zero, one finite complete summary, 300.0-300.1 counted seconds,
wall time under 600 seconds, 1,003,482 parameters, correct cutoff ordering,
unique every-fifth-epoch evaluations plus the final partial epoch, and no
traceback, OOM, worker, evaluator, or non-finite error. Record realized passes
as `num_steps * 256 / 50000`.

Primary success is only `best_test_acc >= 94.58%`. Record final accuracy and
loss against 94.45% and 0.2456, but neither is a substitute for primary
success. A valid score below 127 realized passes counts and cannot be rerun,
though it weakens attribution to the intended normal-exposure mechanism. A
timeout, malformed summary, wrong source/graph/state/transition, or repeated
evaluation epoch is a failure rather than a weak score.

## Closure Contract

- **Normal-exposure success:** supports only exact 8x8 per-example post-mixup
  erasure as a complement to this fixed RandAugment/mixup learner. It does not
  authorize stronger, longer, stochastic-scale, or always-on erasing.
- **Normal-exposure miss:** close additive early Cutout and Random Erasing on
  this accepted stack. Do not rescue with 4x4/12x12/16x16 holes, application
  probability, random scale/aspect, clipped centers, noise fill, before-mixup
  order, worker placement, a different seed, or a different cutoff.
- **Low exposure:** report the valid score but make no clean accuracy-mechanism
  claim. Reimplementation after observing the score is not allowed.
- **Invalid execution:** fix only an independently verified infrastructure or
  harness defect; never replace a valid result with a second score.

## Final Evaluation

**Evidence and reasoning: 2/5.** The intervention is causally clean and
distinct from EXP003's label-changing large rectangles and EXP006/EXP030's
feature masks. However, all three local masking results are negative, the
accepted stack already uses two early regularizers, and EXP041-EXP042 reinforce
that plausible low-cost additions are not automatically frontier-positive.

**Potential impact: 2/5.** The implementation can likely retain the protected
127-pass regime and localized missing-evidence invariance could move a narrow
generalization error mode. More likely, the hole is either too mild to clear
+0.10 or compounds information removal and worsens loss.

**Overall:** testable, isolated, and cheap, but not recommended over a stronger
non-masking finalist. If selected, treat it as the final one-score closure of
this family rather than the start of a Cutout/Random Erasing search.
