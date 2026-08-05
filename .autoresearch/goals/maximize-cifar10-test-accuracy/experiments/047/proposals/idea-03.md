# Proposal: Early-Only Epsilon-0.05 Uniform Label Smoothing

## Recommendation

Retain as a low-confidence, GPU-local closure candidate, not a preferred lead
by default. If it wins EXP047 review, add PyTorch-uniform
`label_smoothing=0.05` to both integer-target cross-entropies in the accepted
batch-shared alpha-0.2 mixup branch while the existing strict predicate
`progress < 0.65` is true. Keep the literal accepted hard-label CE for the
final 35% and preserve every other accepted `a7c42dc` behavior.

```python
LABEL_SMOOTHING = 0.05

if use_mixup:
    mixed_inputs, targets_a, targets_b, mix = mixup_batch(
        inputs, targets, mixup_distribution
    )
    outputs = model(mixed_inputs)
    loss = mix * F.cross_entropy(
        outputs, targets_a, label_smoothing=LABEL_SMOOTHING
    ) + (1.0 - mix) * F.cross_entropy(
        outputs, targets_b, label_smoothing=LABEL_SMOOTHING
    )
else:
    outputs = model(inputs)
    loss = F.cross_entropy(outputs, targets)
```

This is intentionally a weakly diagnosed test. Mixup already supplies
example-aware target softness; uniform smoothing compounds it rather than
repairing a demonstrated calibration defect. Near-zero late training loss
does not establish harmful early confidence, and epsilon 0.05 is a fixed
literature-shaped convention rather than a locally measured optimum.
EXP004/020 protect the 65% duration, EXP005/035 protect alpha 0.2, and EXP041
shows that an additional CE-derived constraint can weaken the pooled-head
frontier. EXP042-045 close other normal-exposure representation, gradient, and
anti-aliasing treatments but do not causally support smoothing. EXP046 failed
before scoring on a prospective loader-service gate; it motivates avoiding
worker-side experiments, not confidence in this objective change.

## Mechanism and Fixed Timing

The only plausible nonredundant signal is class support. Accepted mixup places
target mass on one or two observed classes; smoothing distributes a fixed five
percent across all ten classes. During the saved early critical period, that
class-wide prior might discourage overly separated representations not
addressed by pairwise interpolation. Removing it at exactly 65% preserves the
validated ordinary-example, exact-label boundary-refinement phase.

Only this temporal treatment is admissible:

| Placement | 0-65% counted time | 65-100% counted time | Decision |
|---|---|---|---|
| Whole-run | mixup plus smoothing | smoothed hard CE | Reject: perturbs the validated exact-label tail. |
| **Early-only** | **mixup plus smoothing** | **accepted hard CE** | **Candidate.** |
| Hard-tail-only | accepted mixup | smoothed hard CE | Reject: regularizes boundary refinement. |

Use the existing `use_mixup = progress < MIXUP_END_FRACTION` Boolean as the
sole gate. Add no epoch approximation, second flag, ramp, bridge, warmup, or
worker transition. Do not alter alpha, Beta sampling, permutation, input
mixing, RandAugment's exhausted-iterator cutoff, LR, or the hard branch.
Smoothing is active immediately below 0.65 and absent exactly at and above it.

## Exact PyTorch Dense-Target Semantics

Let `K=10`, `epsilon=0.05`, `u_k=1/K`, batch-shared coefficient `lambda`, and
one-hot labels `e_a`, `e_b`. Accepted mixup uses

```text
y_mix = lambda * e_a + (1 - lambda) * e_b.
```

PyTorch integer-target CE with label smoothing uses uniform mass over all
classes. Linearity of CE in the target gives

```text
q = (1 - epsilon) * y_mix + epsilon * u

L_candidate
  = lambda * CE_LS(logits, a, epsilon)
    + (1 - lambda) * CE_LS(logits, b, epsilon)
  = CE(logits, q).
```

This is not the alternative `epsilon/(K-1)` convention. For a different-class
pair,

```text
q_a = 0.95 * lambda + 0.005
q_b = 0.95 * (1 - lambda) + 0.005
q_k = 0.005 for k not in {a,b}.
```

For a same-class pair, the coincident components combine to `q_a=0.955`, all
nine other classes receive `0.005`, and lambda cancels. Every target row sums
to one. Production must retain the two integer-target calls; explicit dense
`q` is an independent semantic oracle, not an alternate implementation.

For one unreduced example with softmax probabilities `p`,

```text
dL_candidate / dlogits
  = p - q
  = (p - y_mix) + epsilon * (y_mix - u).
```

With default batch-mean reduction, each row contributes `(p-q)/B`. The added
term weakens the drive toward the pairwise target and adds the uniform-prior
correction. Do not detach, clamp, renormalize, symmetrize, change reduction,
reverse lambda on one CE, or smooth only one target component.

## Redundancy and Counter-Hypothesis

The accepted early target is already soft, and alpha 0.2 often draws lambda
near an endpoint; epsilon 0.05 therefore modifies many nearly hard early
targets as well as balanced pairs. The strongest counter-hypothesis is that
accepted pairwise entropy is sufficient and class-uniform mass merely blunts
useful inter-class motion during the most influential training period. The
saved label-smoothing note itself warns against stacking soft-target methods
without a calibration rationale.

No local logit-calibration, entropy, or class-error measurement demonstrates
the proposed failure mode. The test is justified only as a fixed, nearly free
closure point after multiple orthogonal mechanisms failed, not because those
failures raise its probability. A success belongs to the complete
mixup-plus-smoothing trajectory; it does not prove early overconfidence was
causal. A miss closes the exact treatment and immediate family below.

## State, RNG, and Hard-Tail Semantics

The source change is one scalar and two loss keywords. It adds no parameter,
buffer, module, optimizer state, forward, backward, activation, data transform,
evaluation, or inference path. The model remains exactly `1,003,482`
parameters and 52 trainable tensors; construction bytes, pooled-head seed
36036, parameter groups, momentum allocation, data loader, and all accepted
controls remain unchanged.

Label smoothing must consume no CPU or CUDA RNG. From identical incoming
state, accepted and candidate early steps draw byte-identical Beta samples and
permutations, construct identical mixed inputs, produce identical logits and
BN updates, and leave RNG byte-identical; only loss, data gradients, and
updates differ. This is a step-aligned claim. A small timing difference can
change total steps or epoch boundaries in separate 300-second runs without
changing the RNG law or corresponding-step draw order.

The hard-tail source operator remains accepted. From identical incoming model,
optimizer, batch, and RNG states, candidate and accepted hard steps must have
byte-identical logits, loss, gradients, BN updates, optimizer updates, and RNG.
The real candidate enters that tail with intentionally changed parameters and
momentum accumulated under smoothed early gradients, so its full late
trajectory is not accepted-identical. That inherited difference is the
intended durable mechanism; do not reset, realign, or modify state at 65%.

## Fail-Closed Semantic Qualification

Before timing or scoring, run an ignored evaluator-blocked preflight that
prints measurements before assertions. Require:

1. The production diff from `git show a7c42dc:train.py` contains only
   `LABEL_SMOOTHING = 0.05` and the two early CE keywords. `prepare.py`, model,
   optimizer, data, transitions, schedule, evaluation, and summary are exact.
2. Cloned accepted/candidate construction yields byte-identical named
   parameters/buffers, optimizer groups/state, logits, post-construction
   CPU/CUDA RNG, 52 trainable tensors, and `1,003,482` parameters.
3. Fixed same-class and different-class early fixtures with fixed lambda and
   permutation match both explicit paired smoothed integer CEs and independent
   `-(q * log_softmax).sum(-1).mean()` FP64/FP32 oracles. Mixed inputs, logits,
   BN updates, and RNG must equal accepted.
4. Dense rows have the exact masses above, minimum `0.005`, and unit sums.
   Autograd logit gradients match `(p-q)/B`; complete finite nonzero parameter
   gradients and fresh/preseeded-momentum Nesterov updates match an independent
   oracle with accepted coupled decay ordering.
5. From restored identical incoming states, representative hard steps have
   byte-identical loss, gradients, BN buffers, Nesterov updates, and RNG.
   Boundary probes prove smoothing iff the accepted strict mixup predicate is
   true, including immediately below and exactly at 0.65.
6. Restored replay is deterministic; label smoothing uses no RNG; Beta and
   permutation draws, worker-safe RandAugment semantics, LR samples, one
   backward/step, finite guard, and every-fifth-plus-final cadence remain exact.

Any semantic failure closes before timing. Only an independently demonstrated
verifier defect may be repaired; epsilon, target convention, timing,
implementation form, and accepted controls stay frozen.

## Timing and Exposure Gate

Accepted early loss already invokes two component CEs. Smoothing adds the
uniform log-probability term inside each call but no model pass, so the
convolution-dominated backward should make overhead small. This is only a
prior: eager loss kernels and reductions can change cost, and the accepted
130.304-pass run has just 2.536% retention headroom before the protected
127-pass floor.

On one idle H20, compare complete production-equivalent accepted and candidate
steps in early-mixup and hard regimes. Include pinned H2D, LR writes, zeroing,
mixing where active, forward, exact loss, finite check, backward, coupled
Nesterov step, and synchronization. Use at least 20 warmups and two
counterbalanced `A/C/C/A` cycles, producing four retained windows of at least
50 steps per arm/regime from independently restored model, optimizer, fixture,
and RNG state.

For every paired early/hard combination compute

```text
retention_i =
  (0.65 / candidate_early_i + 0.35 / candidate_hard_i) /
  (0.65 / accepted_early_i  + 0.35 / accepted_hard_i)

projected_passes_i = 130.304 * retention_i.
```

Require each window CV `<=5%`, early/hard paired-ratio population CV `<=1%`,
candidate peak allocation `<2,048 MiB`, every retention
`>=127/130.304 = 0.9746439096`, and median projected passes `>=127`. Interleave
regimes or assess all early-by-hard combinations so drift cannot be hidden by
index pairing. A stable timing miss closes this exact implementation before
scoring; do not replace it with a dense-target or fused-loss rescue.

## Sole Score and Decision Contract

After qualification, source-audit rather than rerun the accepted baseline.
Remove stale `run.log`, confirm one idle NVIDIA H20 and frozen evaluator, and
launch exactly one fixed-seed candidate score:

```bash
timeout 600s uv run train.py > run.log 2>&1
```

Require exit zero, one finite summary, `300.0-300.1` counted seconds, total
wall below 600 seconds, `1,003,482` parameters, the correct one-way mixup and
exhausted-iterator RandAugment transitions, unique every-fifth plus final
evaluations with no more than one per epoch, no error signature, and

```text
num_steps * 256 / 50000 >= 127.0 passes.
```

The decision is conjunctive:

- **Improvement:** `best_test_acc >=94.58%` and realized exposure `>=127`.
- **No improvement:** a valid normal-exposure score below 94.58%; final
  accuracy/loss cannot rescue it.
- **Mechanism unsupported at protected exposure:** any valid run below 127
  passes, regardless of score. Record it and do not rerun.
- **Invalid:** repair only an independently proven infrastructure/verifier
  defect while keeping production semantics byte-for-byte fixed. Never repeat
  a valid completed score.

## Strict No-Rescue Closure

A valid `>=127`-pass miss closes this exact early-only epsilon-0.05
PyTorch-uniform treatment and immediate smoothing work on the accepted
learner. Do not try epsilon 0.01/0.02/0.1, a sweep, ramp, whole-run or
hard-tail smoothing, a bridge, `K-1` semantics, confidence penalty,
class-dependent priors, mixup replacement, alpha/cutoff changes, one-component
smoothing, dense targets, fusion, RNG realignment, another seed, or rerun.

A success supports only the full fixed treatment. It does not establish that
calibration improved, uniform mass caused the gain, epsilon 0.05 is optimal,
or smoothing generally helps. A stable timing failure closes implementation
feasibility without an accuracy claim. Any future revisit requires a new
independent training-only calibration diagnosis, not a variant inferred from
this result.

## Falsifiable Hypothesis and Sources

If accepted pairwise mixup still permits harmful early class overconfidence
that a small class-uniform prior can reduce, then PyTorch-uniform epsilon-0.05
smoothing on both mixup component CEs through the existing strict 65% boundary,
followed by exact accepted hard CE, will retain at least 127 passes and raise
fixed-seed `best_test_acc` from 94.48% to at least 94.58%. The honest prior is
low because softness is redundant, calibration is unmeasured, and epsilon is
not locally bracketed.

Evidence is offline/local only: the current goal definition, system
understanding, learnings/results, accepted `a7c42dc` `train.py`, saved
label-smoothing/mixup/time-dependent-regularization notes, EXP004/005/020/035
mixup reports, EXP041's auxiliary-CE miss, EXP042-045 normal-exposure misses,
and EXP046's pre-score loader-gate report. No test examples, evaluator-derived
design, network retrieval, seed search, or result-conditioned parameter choice
informs this proposal.
