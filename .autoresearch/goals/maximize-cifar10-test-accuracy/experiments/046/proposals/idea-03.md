# Proposal: Early-Only Epsilon-0.05 Uniform Label Smoothing

## Recommendation

If this candidate survives comparison with stronger EXP046 ideas, permit exactly
one low-confidence closure score. Add PyTorch-uniform `label_smoothing=0.05` to
both integer-target cross-entropies in the accepted batch-shared alpha-0.2
mixup branch while the existing strict predicate `progress < 0.65` is true.
Retain the literal accepted hard-label cross-entropy afterward and preserve
every other byte of accepted `a7c42dc` behavior.

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

The diagnosis is weak and must remain explicit. Mixup already supplies
example-aware soft targets; this proposal adds uniform mass on top rather than
addressing an established calibration failure. The accepted learner nearly
interpolates its hard tail, but that fact does not show that early logits are
overconfident. Epsilon 0.05 is a conservative prospective convention, not a
value identified by local data. EXP004/020 protect the 65% mixup boundary,
EXP005/035 protect alpha 0.2, and EXP041 shows that an additional CE-derived
constraint can weaken the pooled-head frontier. EXP045's normal-exposure
shortcut miss closes another architecture route but supplies no positive
causal evidence for smoothing. This proposal is a cheap falsifiable target
intervention, not permission for a smoothing sweep.

## Mechanism and Temporal Placement

The saved label-smoothing note supports a possible reduction in overconfident
class separation, while the saved time-dependent regularization note supports
removing regularization before final convergence. The narrow hypothesis is
that uniform mass supplies an early class-wide prior not present in a single
pairwise mixup target, and that the complete final 35% exact-label phase can
then refine ordinary class boundaries without continued target entropy.

Only this placement is admissible:

| Placement | 0-65% counted time | 65-100% counted time | Decision |
|---|---|---|---|
| Whole-run | mixup plus smoothing | smoothed ordinary CE | Reject: disturbs the validated exact-label tail. |
| **Early-only** | **mixup plus smoothing** | **accepted hard CE** | **Candidate.** |
| Hard-tail-only | accepted mixup | smoothed ordinary CE | Reject: regularizes the boundary-refinement phase. |

Use the existing `use_mixup = progress < MIXUP_END_FRACTION` Boolean as the
sole gate. Do not add a second cutoff, epoch approximation, ramp, bridge,
warmup, or worker-side flag. Do not smooth ordinary inputs, change alpha,
resample lambda, alter the permutation, or move the exhausted-iterator
RandAugment transition. Smoothing stops on the exact first step on which the
accepted strict mixup predicate is false.

## Exact PyTorch Target Semantics

Let `K=10`, `epsilon=0.05`, `u_k=1/K`, batch-shared mix coefficient `lambda`,
and one-hot labels `e_a` and `e_b`. The accepted early target is

```text
y_mix = lambda * e_a + (1 - lambda) * e_b.
```

PyTorch integer-target cross-entropy with `label_smoothing=epsilon` uses the
uniform-over-all-classes convention. By linearity of cross-entropy in its
target, the required paired implementation is mathematically

```text
q = (1 - epsilon) * y_mix + epsilon * u

L_candidate
  = lambda * CE_LS(logits, a, epsilon)
    + (1 - lambda) * CE_LS(logits, b, epsilon)
  = CE(logits, q).
```

It is not the alternative `epsilon/(K-1)` convention. For a different-class
pair the exact row masses are

```text
q_a = 0.95 * lambda + 0.005
q_b = 0.95 * (1 - lambda) + 0.005
q_k = 0.005 for k not in {a,b}.
```

For a same-class pair, coincident terms combine to `q_a=0.955`, all nine other
classes receive `0.005`, and lambda cancels. Every dense row sums to one.
Production must keep the two integer-target calls rather than materializing
`q`; the dense expression is an independent oracle, not an alternate
implementation.

For one unreduced example with softmax probabilities `p`,

```text
dL_candidate / dlogits
  = p - q
  = (p - y_mix) + epsilon * (y_mix - u).
```

With PyTorch's default batch mean, each row contributes `(p-q)/B`. The second
term is the complete intervention: it weakens motion toward the pairwise mixed
target and adds a class-uniform correction. Do not detach, clamp, renormalize,
symmetrize, change reduction, reuse a different lambda, or apply smoothing to
only one component CE.

## Redundancy and Counter-Hypothesis

Mixup and label smoothing are not algebraically identical: mixup puts nearly
all target mass on one or two observed classes, whereas smoothing distributes
five percent across all ten. That distinction is the only plausible
nonredundant signal. It does not establish usefulness. The accepted mixup
strength and duration are already locally bracketed, and adding uniform mass
may simply reduce useful class-to-class motion during the critical early
period. Because batch-shared alpha 0.2 frequently yields lambdas near an
endpoint, smoothing also changes many nearly hard early targets even though
mixup's image interpolation remains unchanged.

The strongest counter-hypothesis is therefore that the accepted early target
entropy is already sufficient and that uniform mass blunts formation of the
pooled-head decision boundary. A gain would support only the complete fixed
mixup-plus-uniform-smoothing treatment. It would not prove overconfidence was
the cause, that epsilon 0.05 is optimal, or that smoothing is superior to
mixup. A miss would reject the complete treatment under the closure below.

## State, RNG, and Hard-Tail Controls

The production change adds one Python scalar and two loss keyword arguments.
It adds no parameter, buffer, module, optimizer state, activation path,
forward pass, backward pass, input transform, evaluation, or inference
operation. Parameter count remains exactly `1,003,482`; all named model and
optimizer state, initialization bytes, construction RNG, pooled-head seed,
data-loader configuration, and parameter-group ordering remain accepted.

Label smoothing performs deterministic tensor arithmetic and must consume no
CPU or CUDA RNG. From identical incoming state, the accepted and candidate
must draw the same Beta sample and permutation, produce identical mixed
inputs/logits/BN updates, and leave RNG byte-identical; only loss and gradients
differ. This is a step-aligned claim, not an overclaim that two wall-time runs
must complete the same number of steps. A measurable loss-cost difference can
change exposure and therefore later epoch boundaries while preserving the RNG
law and draw order for corresponding steps.

The hard-tail source branch is literally accepted. From an identical incoming
model, optimizer, input, and RNG state, accepted and candidate hard steps must
have byte-identical loss, gradients, BN state, updates, and RNG. The actual
candidate enters the hard tail with intentionally different parameters and
momentum due to its early smoothed gradients, so its full hard-tail trajectory
is not expected to equal accepted. That historical state difference is the
intended lasting effect; no reset or realignment is allowed at 65%.

## Fail-Closed Semantic Qualification

Before timing or scoring, use an ignored evaluator-blocked harness. Print all
measurements before assertions and require:

1. The production diff from `git show a7c42dc:train.py` contains only the
   `LABEL_SMOOTHING = 0.05` constant and the two early CE keyword arguments.
   `prepare.py`, model, data, schedules, optimizer, transitions, evaluation,
   seeds, and summary code remain unchanged.
2. Accepted and candidate construction from cloned CPU/CUDA states yields
   byte-identical named parameters and buffers, optimizer groups/state,
   post-construction RNG, logits, and exactly `1,003,482` parameters.
3. On fixed different-class and same-class fixtures with a fixed lambda and
   permutation, the production paired loss matches both two explicit smoothed
   integer CEs and an independently constructed
   `-(q * log_softmax).sum(-1).mean()` oracle within preregistered FP32/FP64
   bounds. Forward logits, mixed inputs, and BN updates equal accepted.
4. Dense rows have the exact masses above, minimum `0.005`, and unit sums.
   Autograd logit gradients match mean-reduced `p-q`; full finite nonzero
   parameter gradients and fresh plus preseeded-momentum Nesterov updates match
   an independent oracle with accepted coupled decay ordering.
5. From restored identical incoming states, representative hard steps give
   byte-identical accepted/candidate loss, gradients, BN buffers, optimizer
   updates, and RNG. Boundary probes prove smoothing is active immediately
   below 0.65 and absent exactly at and above 0.65.
6. Restored replay is deterministic; smoothing consumes no CPU/CUDA RNG; the
   Beta draw, permutation, worker-safe RandAugment semantics, one backward and
   step, finite guard, LR samples, and evaluation cadence remain accepted.

Any production-semantic failure closes before timing. Repair is allowed only
for an independently demonstrated harness defect; do not change epsilon,
target convention, cutoff, reduction, or accepted controls.

## Timing and Exposure Gate

The accepted early loss already evaluates two component cross-entropies. The
candidate adds the uniform log-probability contribution inside each call but
does not add a model forward or backward, so convolutional backpropagation
should dominate. This remains a prior: loss kernels and reduction behavior can
change eager H20 cost, and only 2.536% exposure loss is available between
accepted 130.304 passes and the protected 127-pass floor.

On one idle H20, time complete production-equivalent accepted and candidate
steps in both early-mixup and hard-label regimes. Include pinned H2D, LR group
writes, zeroing, accepted mixing where applicable, full forward, exact loss,
finite check, backward, coupled Nesterov step, and final synchronization. Use
at least 20 warmups and two counterbalanced `A/C/C/A` cycles, yielding four
retained windows of at least 50 steps per arm and regime from independently
restored model, optimizer, fixture, and RNG states.

For each paired early/hard timing combination compute

```text
retention_i =
  (0.65 / candidate_early_i + 0.35 / candidate_hard_i) /
  (0.65 / accepted_early_i  + 0.35 / accepted_hard_i)

projected_passes_i = 130.304 * retention_i.
```

Require every window CV `<=5%`, early and hard paired-ratio population CV
`<=1%`, candidate peak allocation `<2,048 MiB`, every retention
`>=127/130.304 = 0.9746439096`, and median projected passes `>=127.0`.
Interleave regimes or evaluate all early-by-hard combinations so drift cannot
be hidden by favorable index pairing. A stable timing miss closes systems
viability without an accuracy score or a cheaper-loss rescue.

## Sole Score and Decision Contract

After semantic and timing gates pass, source-audit the accepted baseline rather
than rerunning it. Remove stale `run.log`, confirm one idle NVIDIA H20 and the
frozen evaluator, then run exactly one fixed-seed candidate score:

```bash
timeout 600s uv run train.py > run.log 2>&1
```

Require exit zero, one finite final summary, `300.0-300.1` counted seconds,
wall time below 600 seconds, exactly `1,003,482` parameters, one strict mixup
transition, the correct exhausted-iterator RandAugment transition, unique
every-fifth plus final evaluations with no more than one per epoch, no error
signature, and realized exposure

```text
num_steps * 256 / 50000 >= 127.0 passes.
```

The preregistered decision is conjunctive:

- **Improvement:** `best_test_acc >=94.58%` and realized exposure `>=127`.
- **No improvement:** a valid run at normal exposure with
  `best_test_acc <94.58%`; final accuracy or loss cannot rescue it.
- **Mechanism unsupported at protected exposure:** a valid run below 127
  passes, regardless of score. Record the sole result and do not rerun.
- **Invalid:** only an independently demonstrated infrastructure or verifier
  defect may be repaired while the production treatment remains byte-for-byte
  fixed. A valid completed score is never repeated.

## No-Rescue Closure

A valid normal-exposure miss closes this exact early-only epsilon-0.05
PyTorch-uniform treatment and the immediate smoothing neighborhood on the
accepted learner. Do not follow it with epsilon 0.01/0.02/0.1, a sweep, ramp,
whole-run or hard-tail smoothing, a bridge window, `K-1` semantics, confidence
penalty, class-dependent prior, mixup replacement, alpha or cutoff changes,
one-component smoothing, dense-target implementation rescue, RNG realignment,
another seed, or another score.

A success supports only the complete fixed treatment and may be accepted
without tuning. It does not prove calibration improved, uniform mass was
causal, epsilon 0.05 is locally optimal, or label smoothing generally helps.
A stable timing failure closes the exact implementation's feasibility without
an accuracy claim. Reconsidering smoothing after any closure requires a new
independent training-only calibration diagnosis, not a variant of this result.

## Falsifiable Hypothesis and Evidence

If the accepted pairwise mixup targets still permit harmful early class
overconfidence that a small class-uniform prior can reduce, then applying
PyTorch-uniform epsilon-0.05 smoothing to both mixup component CEs through the
existing strict 65% boundary, followed by the exact accepted hard CE, will
retain at least 127 passes and raise fixed-seed `best_test_acc` from 94.48% to
at least 94.58%. The honest prior is low because target softness is redundant,
the calibration failure is unmeasured, and epsilon is not locally bracketed.

Evidence is offline/local only: `01-definition.md`, `02-system-understanding.md`,
`03-experiment-learnings.md`, `04-results.tsv`, accepted `a7c42dc` `train.py`,
the saved label-smoothing, mixup, and time-dependent-regularization notes,
EXP004/005/020/035 mixup reports, EXP041's auxiliary-CE report, and EXP045's
normal-exposure shortcut report. No test examples, evaluator-derived design,
network retrieval, seed search, or result-conditioned parameter choice informs
this proposal.
