# Proposal: Early 8x8 Cutout After Mixup

## Recommendation

Advance only as a low-confidence closure experiment if the competing EXP041
ideas lack a cleaner mechanism. On accepted `a7c42dc`, retain the complete
`(2,2,3)` WRN, scale-0.1 pooled residual MLP head, worker-safe N1/M5 early
RandAugment, batch-shared alpha-0.2 mixup, optimizer, schedule, seed, and clean
tail. During the existing mixup branch only, erase one independently located
`8 x 8` square from each already-mixed input and fill it with normalized zero
(the configured channel mean because accepted normalization uses unit standard
deviation). Disable Cutout exactly when mixup disables at 65% counted time.

Do not replace mixup, alter its targets, remove RandAugment, extend erasing into
the hard-label tail, tune hole size/probability/fill, or combine Cutout with a
new loss. Retaining both accepted regularizers is mechanistically defensible:
RandAugment supplies global photometric/geometric invariance, mixup supplies
between-example linearity, and Cutout tests localized missing-evidence
robustness. Replacing either proven component would mostly retest EXP003's
failed substitution logic rather than isolate localized erasure.

The evidence prior is nevertheless poor. EXP003 replaced mixup with shared
CutMix rectangles averaging 31% pasted area and scored 93.72% at normal
exposure. EXP006 and EXP030 show that additive early feature masks lose 0.55
and 0.41 points, respectively. EXP026's early RandAugment was only +0.05 alone,
and its exact interaction with added low-resolution capacity became valuable
in EXP027; that does not establish that a third early regularizer will help.
This proposal is distinct from those failures because the mask is input-space,
label-preserving, per-example, only 6.25% of the image, and removed for the
full hard tail, but those distinctions justify at most one restrained test.

## Fixed Treatment

Use one square hole of side 8 on every early sample. An `8 x 8` hole is the
conservative CIFAR-scale Cutout setting chosen before seeing any EXP041 result:
it removes exactly 64 of 1,024 pixels (6.25%), materially less information than
the canonical large-hole CIFAR Cutout recipe and far less than EXP003's mean
31% CutMix replacement. The smaller fixed treatment is necessary because it is
being stacked with accepted RandAugment and mixup rather than used as the sole
regularizer. There is no Bernoulli application probability, scale distribution,
aspect-ratio distribution, boundary clipping, or magnitude sweep.

For a tensor `x` of shape `B x C x 32 x 32`, independently sample integer
top-left coordinates

```text
top_i  ~ Uniform{0, ..., 24}
left_i ~ Uniform{0, ..., 24}
```

and set all channels in
`x[i, :, top_i:top_i+8, left_i:left_i+8]` to zero. Sampling top-left positions
rather than clipped centers guarantees exactly 64 erased pixels for every
sample and avoids edge-dependent severity. Locations must be independent by
example but shared across channels. The operation occurs after
`mixup_batch(inputs, targets, distribution)` and before `model(mixed_inputs)`:
there is one hole in the final mixed input, rather than two donor holes blended
together, and accepted target interpolation remains exact.

A vectorized implementation should construct broadcast row/column coordinates
and a `B x 1 x 32 x 32` Boolean keep mask, then use `masked_fill` or
multiplication. Do not loop over examples or mutate a donor/accepted input
alias in place. This adds no parameter and leaves the accepted parameter count
at 1,003,482.

Create one private device-local `torch.Generator` after model and optimizer
construction, seed it from the already fixed `torch.initial_seed()` (42), and
pass it explicitly to both coordinate draws. This is not a searched seed.
The private stream prevents Cutout from perturbing accepted CUDA mixup
coefficients/permutations or any global CPU/CUDA RNG trajectory. Constructing
the generator must not change global RNG state. Because the helper is called
only inside `if use_mixup`, the complete hard-label path performs no Cutout
allocation, draw, branch, or state update.

## Mechanism and Falsifiable Hypothesis

The pooled-head model remains generalization-limited: it nearly interpolates
the hard tail but finishes with 0.2456 test loss. A small local hole can force
the early representation to distribute evidence across spatial locations,
which is not identical to RandAugment's global transforms or mixup's dense
convex interpolation. Applying the hole after mixup preserves exact mixed
targets while preventing a single local region from dominating both source
signals. The 35% clean tail then trains on unmasked, un-mixed images and can
restore full-evidence calibration.

The falsifiable claim is narrow: exact per-example post-mixup 8x8 Cutout through
65% will retain at least 127 data passes and raise fixed-seed
`best_test_acc` from 94.48% to at least 94.58%. Final accuracy at least 94.45%
and final loss at most 0.2456 are non-decisive corroboration that any best-epoch
gain reflects boundary quality rather than a transient checkpoint.

The counter-hypothesis is stronger on current evidence: even 6.25% additional
input removal compounds an already calibrated early regularization stack,
weakens learning of the useful depth/RandAugment composition, and reproduces
the normal-exposure degradation seen for CutMix and residual masking. A null or
loss does not imply that all occlusion augmentation is universally ineffective;
it closes this exact additive Cutout family for this fixed-budget learner.

## Expected Benefit and Risks

Potential upside is a near-zero-spatial-compute invariance that acts before the
accepted convolutions and does not alter labels, parameters, or the hard tail.
Unlike shared-rectangle CutMix, every sample receives its own small location
and no erased area is translated into potentially incorrect class mass.

Risks are substantial:

- three simultaneous early regularizers may remove too much usable signal;
- zeroing after mixup creates a mean-color square that can itself become a
  recognizable synthetic cue;
- per-example locations destroy batch-level spatial coherence and introduce a
  new stochastic trajectory even though accepted global RNG streams are kept;
- the 8x8 choice may be too weak for a 0.10-point gain, while the better-known
  16x16 CIFAR hole is implausibly severe in this accepted stack;
- mask construction and an extra full-input write occur inside counted time;
  only about 2.54% overhead can be tolerated before accepted 130.304 passes
  fall below 127;
- one fixed seed cannot estimate a seed-averaged treatment effect.

## Exact Production Scope

Only `train.py` may change. The intended diff is limited to:

1. add `CUTOUT_SIZE = 8` near the other fixed constants;
2. add a pure vectorized `cutout_batch(inputs, size, generator)` helper;
3. construct the private device generator after accepted model/optimizer and
   mixup-distribution construction without consuming global RNG;
4. call the helper once on `mixed_inputs` inside the existing `use_mixup`
   branch, after `mixup_batch` and before model forward.

Do not modify `EarlyRandAugment`, transforms, shared worker state, model
construction/forward, parameter groups, learning rate, cutoff constants,
evaluation cadence, summary output, `prepare.py`, or any dependency. In
particular, preserve accepted RandAugment's epoch-exhausted removal, which may
lag the exact mixup/Cutout cutoff by less than one iterator. That short
RandAugment-only interval is part of accepted behavior.

## Semantic Preflight

Use an ignored, evaluator-free harness with an independent
`git show a7c42dc:train.py` oracle. Print all diagnostics before assertions and
require:

- the production diff is confined to the four scopes above and only
  `train.py` differs;
- accepted/candidate initial state-dict keys, tensor shapes and bytes,
  parameters (1,003,482), buffers, optimizer groups/options/state, schedule,
  transforms, loader options, temporal constants, evaluator cadence, and
  post-construction global CPU/CUDA RNG states are exact;
- fixed all-one inputs receive exactly 64 zero pixels per example in all three
  channels, no zeros outside their rectangles, per-example coordinates are in
  `[0,24]`, and at least two locations differ in a deterministic batch fixture;
- a separately coded per-example slicing oracle is bitwise equal to the
  vectorized production helper on CPU and CUDA fixtures;
- channel sharing, boundary positions `(0,0)` and `(24,24)`, batch size one,
  noncontiguous inputs, dtype/device preservation, absence of source mutation,
  finite gradients, and deterministic private-generator replay all pass;
- helper calls change only the private generator state and leave global
  CPU/CUDA RNG states exact; restoring private state reproduces coordinates and
  pixels bitwise;
- accepted and candidate `mixup_batch` outputs, coefficients, permutations,
  targets, and post-mixup global RNG states are exact before Cutout; source
  inspection and a call-order probe prove Cutout occurs after mixing;
- at `progress < 0.65`, candidate logits/loss/updates differ nontrivially but
  remain finite; at `progress == 0.65` and above, accepted/candidate complete
  hard-label steps, RNG states, model/optimizer updates, and outputs are
  bitwise equal from restored fixtures;
- worker-side EarlyRandAugment private RNG behavior and the exhausted-iterator
  state transition remain accepted, including no clean-tail prefetch leakage;
- evaluation is never more frequent than once per epoch and the 300-second
  budget/final summary contract is unchanged.

The harness may be corrected after verifier-only mistakes, but production must
not be adjusted in response to a semantic failure. A failure does not permit a
different fill, clipped center, probability, hole size, mask order, RNG seed,
or in-place kernel; those are different treatments.

## Timing and Exposure Gate

Measure accepted and candidate complete production-equivalent steps on one
idle H20 in both early and hard regimes. Include H2D, LR calculation/group
writes, zeroing, accepted mixup draws and permutation when active, Cutout when
active, full forward, cross-entropy, finite guard, backward, coupled Nesterov
step, and final synchronization. Use at least 20 warmups and four
counterbalanced windows of at least 50 steps per arm/regime with fresh,
deterministic fixtures per replicate.

Print raw windows before assertions. Require finite results and population CV
no greater than 5% for every arm/regime. Compute

```text
retention =
    (0.65 / candidate_early_ms + 0.35 / candidate_hard_ms) /
    (0.65 / accepted_early_ms  + 0.35 / accepted_hard_ms)
projected_passes = 130.304 * retention
```

Proceed only if retention is at least
`127 / 130.304 = 0.9746439096`, projected passes are at least 127, and the
candidate/accepted hard-step ratio is within 1% because Cutout must have an
exact hard-tail bypass. Also run a short real-loader marker check to ensure the
existing RandAugment transition remains worker-safe; no extended loader timing
is needed because the CPU pipeline is source-identical.

A stable timing miss ends the proposal. Do not rerun timing, lower the floor,
reuse a batch-shared hole, reduce size/probability, move masking to workers,
use in-place mutation, or cache masks to rescue exposure.

## Sole Scored Run and Decision Contract

After both gates pass, reconfirm baseline 94.48% at `a7c42dc`, one idle NVIDIA
H20, local CIFAR-10, frozen `prepare.py`, no stale log, and the exact production
diff. Run exactly once at seed 42:

```bash
timeout 600s uv run train.py > run.log 2>&1
```

Require exit zero, one finite complete summary, 300.0-300.1 counted seconds,
wall time below 600 seconds, 1,003,482 parameters, accepted transition ordering,
unique every-fifth-epoch evaluations plus the final partial epoch, and no
traceback, OOM, worker, evaluator, or non-finite error. Record realized passes
as `num_steps * 256 / 50000`.

Primary success is only `best_test_acc >= 94.58%`. Record final accuracy and
loss against the 94.45% / 0.2456 corroboration thresholds, but neither rescues
a primary miss. A valid score below 127 realized passes counts and cannot be
rerun, though it is operationally inconclusive for the intended low-cost
mechanism. A timeout, malformed summary, wrong source/graph/state/transition,
or repeated evaluation epoch is a failure rather than a weak score.

## Restrained Closure

- **Normal-exposure success:** supports only the exact claim that small
  per-example post-mixup missingness complements accepted early RandAugment and
  mixup on this fixed seed. It does not justify a larger hole, longer window,
  RandomErasing scale/aspect distributions, or always-on erasure.
- **Normal-exposure miss:** close additive early Cutout/random-erasing on this
  accepted stack. Do not rescue with 4x4/12x12/16x16 holes, probability,
  clipping, fill noise, before-mixup ordering, worker placement, another seed,
  or cutoff changes. EXP003, EXP006, and EXP030 already make such adjacency
  search especially weak.
- **Low exposure:** report the valid score but do not infer the accuracy
  mechanism. Reimplementation for speed is not allowed after seeing it.
- **Invalid execution:** fix only an independently verified infrastructure or
  harness defect. Never use a second score to replace a valid result.

## Final Evaluation

**Evidence and reasoning: 2/5.** The localized-occlusion mechanism is distinct
and the exact 8x8 treatment addresses EXP003's large shared rectangles, but
all relevant local additive-masking evidence is negative. Stable prior
knowledge supports Cutout on CIFAR generally, not this already regularized
deeper-plus-RandAugment-plus-mixup learner.

**Potential impact: 2/5.** Compute and implementation cost are small enough to
preserve the 127-pass regime, and localized invariance could in principle move
the remaining boundary error. More likely, the intervention is too weak to
clear +0.10 or compounds early information removal and regresses.

**Overall:** technically testable and causally isolated, but not recommended
over a cleaner non-masking mechanism. If selected, it should be understood as
one final closure score with no result-conditioned variants.
