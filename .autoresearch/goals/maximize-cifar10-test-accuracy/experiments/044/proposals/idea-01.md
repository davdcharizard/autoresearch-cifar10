# Proposal: Early-Only Epsilon-0.05 Label Smoothing on Accepted Mixup

## Recommendation

If label smoothing must receive one score, test exactly one conservative
formulation: keep the accepted batch-shared alpha-0.2 mixup through the strict
65% counted-time boundary, but use PyTorch-style `epsilon = 0.05` label
smoothing in each of its two cross-entropies. At 65%, return to the exact
accepted hard-label cross-entropy for the full final 35%:

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

Do not smooth the hard tail, alter alpha or its RNG, change the 65% cutoff, or
replace mixed inputs with ordinary inputs. Preserve every other byte of the
accepted `a7c42dc` training recipe.

This is a **low-confidence, low-ranked candidate**, not a leading
recommendation. It is executable and nearly free, but local evidence is weak:
the accepted learner already has soft, example-aware mixup targets; no local
calibration or confidence measurement diagnoses missing uniform-target
entropy; and epsilon 0.05 has no local bracket. EXP035 and EXP038 previously
ranked this idea below better-attributed alternatives for those reasons. The
recent EXP041-043 misses further counsel against loosely motivated loss,
readout, or gradient constraints. A single preregistered closure score is the
most that the evidence supports.

## Prospective Timing Comparison

There are three coherent placements, all considered before any score:

| Formulation | Early 0-65% | Final 65-100% | Prospective verdict |
|---|---|---|---|
| Whole-run | alpha-0.2 mixup plus smoothing | smoothed hard-label CE | Reject. It both stacks softness early and eliminates the validated clean tail. |
| Early-only | alpha-0.2 mixup plus smoothing | exact hard-label CE | Select, narrowly. It preserves the accepted temporal boundary and all clean refinement. |
| Hard-tail-only | exact accepted mixup | smoothed hard-label CE | Reject. It spends the intervention exactly where EXP002/004/020 and the time-regularization note say softness should be removed. |

The clean tail is not unused budget. EXP002's gain used 35% hard-label
refinement; moving the mixup cutoff to 50% or 75% both lost accuracy, while
late prefix freezing (EXP028) showed that even low-level parameters need to
keep adapting there. Its purpose is to recover class-specific confidence and
refine actual decision boundaries after early vicinal and image
regularization. Whole-run and hard-tail-only smoothing replace those exact
one-hot targets with a uniform prior throughout some or all of this protected
phase. Early-only smoothing is therefore the only timing choice that does not
contradict the strongest local temporal evidence, although it remains
potentially redundant with mixup.

Epsilon 0.05 is fixed as a deliberately mild one-shot point. For ten classes,
it assigns only 0.005 uniform mass to each class and retains 0.955 on a hard
label. Epsilon 0.1 would double an already redundant prior, while a smaller
value would make a one-run top-1 test even less informative. This is a
prospective convention, not a locally estimated optimum; no sweep or
result-conditioned adjustment is permitted.

## Exact Target and Gradient Semantics

Let `K = 10`, `u_k = 1/K`, batch-shared `lambda = mix`, and one-hot vectors
`e_a`, `e_b`. Accepted mixup uses

```text
y_mix = lambda * e_a + (1 - lambda) * e_b
L_accepted = CE(logits, y_mix).
```

The selected loss is exactly

```text
epsilon = 0.05
q = (1 - epsilon) * y_mix + epsilon * u
L_candidate = CE(logits, q)
            = lambda * CE_LS(logits, a, epsilon)
              + (1 - lambda) * CE_LS(logits, b, epsilon).
```

This uses PyTorch's uniform-over-all-classes convention, not the alternative
`epsilon/(K-1)` convention. If `a != b`, target masses are

```text
q_a = 0.95 * lambda + 0.005
q_b = 0.95 * (1 - lambda) + 0.005
q_k = 0.005 for k not in {a, b}.
```

If the permutation pairs an example with the same class, the coincident masses
combine: that class receives `0.955`, every other class `0.005`, independently
of lambda. Do not smooth the two labels and then interpolate inputs with a
different coefficient, clamp/symmetrize lambda, create dense targets in
production, or detach any term.

For softmax probabilities `p`, the per-example logit gradient is

```text
dL_candidate/dlogits = p - q
                      = (p - y_mix) + epsilon * (y_mix - u).
```

Thus smoothing leaves the forward pass unchanged but reduces the drive toward
the mixed target and adds a uniform-prior correction. For a highly confident
correct prediction, it reverses part of the accepted push toward still larger
class separation. That can reduce overconfidence, but it can also blunt the
useful class-boundary motion supplied by mixup. All parameter gradients change
through this logit signal; weight decay, momentum, Nesterov ordering, and
optimizer state remain accepted.

## Scope and Throughput

The only production changes are one scalar constant and the two keyword
arguments in the existing early mixup loss. Model parameters, forward graph,
ordinary hard-tail loss, data tensors, Beta sample, permutation, RNG stream,
optimizer groups, continuous `5e-4` matrix decay, LR curve, augmentation
transitions, evaluator, and seed 42 remain unchanged. In particular, do not use
the dense `q` expression in production; the two built-in smoothed CEs preserve
the accepted paired-loss structure.

There is no extra model forward or backward, parameter, activation, random
draw, or data transform. PyTorch's smoothed CE adds a uniform log-probability
reduction inside the already present two early CE calls, so expected overhead
is well below spatial convolution cost and likely below 1%. That expectation
is not a timing verdict. The accepted baseline delivered 25,450 steps and
130.304 passes; the candidate must prospectively retain at least 127 passes.

## Fail-Closed Qualification

Use an ignored, evaluator-free local preflight and do not construct or inspect
the CIFAR-10 test set. Print diagnostics before assertions. Before scoring,
require all of the following:

1. Diff against `git show a7c42dc:train.py`; require changes only to the fixed
   constant and the two early `cross_entropy` calls. Hash `prepare.py` and prove
   all accepted constants, model/state, optimizer, data, RNG, timing,
   transitions, and evaluation cadence are unchanged.
2. Instantiate accepted and candidate models from cloned CPU/CUDA RNG states;
   require identical named parameter/buffer bytes, optimizer state/groups,
   setup RNG, default logits, parameter count `1,003,482`, and no added state.
3. On deterministic early fixtures, reuse one fixed scalar lambda and one fixed
   permutation. Require candidate input/logits/BN updates/RNG to equal accepted;
   require the production paired loss to match both an independent dense-target
   `-(q * log_softmax).sum(...).mean()` oracle and the explicit two-smoothed-CE
   formula within preregistered FP32 tolerance.
4. Include same-class and different-class pairs. Require target rows sum to one,
   minimum mass 0.005, and exact masses above. Compare autograd logit gradients
   to `p-q`, and parameter gradients/one fresh plus one preseeded-momentum
   Nesterov update to an independent oracle. Require finite, nonzero treatment.
5. On hard-tail fixtures, require candidate loss, gradients, updates, BN state,
   and RNG to be byte-identical to accepted hard CE. Statically require
   smoothing only under `progress < MIXUP_END_FRACTION`; prove the strict 0.65
   boundary, exhausted-iterator RandAugment cutoff, LR samples, finite guard,
   sole backward/step, and every-fifth-plus-final evaluation remain accepted.
6. Replay early and hard fixtures from restored state and require deterministic
   equality. Prove label smoothing consumes no CPU or CUDA RNG and the accepted
   Beta draw/permutation sequence is unchanged.

A semantic failure closes before scoring. Repair only a demonstrable verifier
or implementation defect while preserving epsilon, timing, loss convention,
and all accepted semantics.

On one idle H20, time complete production-equivalent accepted and candidate
steps for both early and hard regimes. Use at least 20 warmups and two
counterbalanced `A/C/C/A` cycles, giving four retained windows per arm and
regime of at least 50 synchronized steps. Include H2D, LR writes, zeroing,
mixup, forward, exact loss/finite guard, backward, coupled Nesterov step, and
final synchronization. Restore identical model, optimizer, fixture, and RNG
before every window. Require every window CV <=5%, paired-ratio population CV
<=1% in each regime, candidate peak allocation <2,048 MiB, and every paired
whole-run retention to satisfy

```text
retention_i =
    (0.65 / candidate_early_i + 0.35 / candidate_hard_i) /
    (0.65 / accepted_early_i  + 0.35 / accepted_hard_i)

retention_i >= 127 / 130.304 = 0.9746439096
median_projected_passes = 130.304 * median(retention_i) >= 127.0.
```

A stable timing miss closes without a score and is not an accuracy result.

## Sole Score and Closure

After all gates pass, run exactly one fixed-seed score on one H20 with the
required redirection and timeout:

```bash
timeout 600s uv run train.py > run.log 2>&1
```

Require exit zero, one finite summary, 300.0-300.1 counted seconds, wall time
under 600 seconds, correct single transitions and unique evaluation cadence,
unchanged parameter count, no error signature, and realized exposure
`num_steps * 256 / 50000 >= 127.0`.

The sole success threshold is `best_test_acc >= 94.58%`, exactly 0.10 points
above the accepted 94.48%. Final accuracy and loss are descriptive only and
cannot rescue a top-1 miss. Both `>=94.58%` and `>=127` realized passes are
required to support this proposal's normal-exposure hypothesis. A valid run
below 127 remains the sole recorded goal result but does not support the
mechanism and must not be rerun.

A valid normal-exposure miss is strong closure for **this exact** early-only,
PyTorch-uniform, epsilon-0.05 addition. Do not rescue it with epsilon 0.01,
0.02, 0.1, a ramp, whole-run smoothing, hard-tail smoothing, a bridge window,
mixup replacement, alpha/cutoff changes, class-dependent smoothing, confidence
penalties, alternate `K-1` semantics, RNG realignment, another seed, or a
second score. The result would not prove label smoothing universally useless,
but absent a new independent calibration diagnosis it should close immediate
label-smoothing work on this accepted learner.

## Risks and Evidence

- **Redundant regularization:** mixup already supplies soft, example-aware
  targets; uniform mass may weaken useful signal rather than add information.
- **No calibration diagnosis:** near-zero train loss and 0.2456 test loss show a
  generalization gap, not specifically harmful overconfidence.
- **Weak epsilon basis:** 0.05 is a conservative literature-shaped convention,
  not a local optimum. The strict no-rescue rule is needed to prevent a sweep.
- **Single-seed resolution:** a 0.10-point hurdle is only ten CIFAR-10 test
  examples; the fixed protocol can establish benchmark improvement, not a
  seed-averaged effect.
- **Recent negative neighborhood:** EXP041's extra CE, EXP042's adaptive
  pooling, and EXP043's gradient projection all retained >=127 passes yet
  worsened accuracy/loss. They do not directly falsify smoothing, but they make
  another generic constraint a low-priority bet.

Evidence is offline/local only: accepted `train.py` at `a7c42dc`;
`01-definition.md`, `02-system-understanding.md`, `03-experiment-learnings.md`,
and `04-results.tsv`; `knowledge/papers/{label-smoothing,mixup,time-matters-regularization}.md`;
EXP002/004/020/028/035/038 and the EXP041-043 reports. No test data, network
source, result-conditioned sweep, or external state was used to choose the
formula.
