# Proposal: One-Shot Early-Only Epsilon-0.05 Label Smoothing

## Recommendation

Advance exactly one low-confidence closure score: add PyTorch-uniform
`label_smoothing=0.05` to both component cross-entropies of the accepted
batch-shared alpha-0.2 mixup loss while `progress < 0.65`, then retain the exact
accepted hard-label cross-entropy for the final 35% counted time. Preserve every
other part of accepted `a7c42dc` `train.py`.

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

This proposal remains weakly diagnosed. Mixup already supplies example-aware
soft targets, near-zero late training loss does not establish harmful early
overconfidence, and epsilon 0.05 is a conservative convention rather than a
locally fitted value. EXP005/035 bracket alpha around 0.2, EXP004/020 bracket
the 65% cutoff, and EXP041 shows that an additional CE-derived constraint can
hurt the accepted pooled-head learner. EXP044's valid 93.95% dispersion miss
does not improve the positive evidence for smoothing; it only closes another
readout route and makes a nearly compute-free, orthogonal target intervention
reasonable as a final one-shot test. This is not permission for a label
smoothing sweep.

## Hypothesis and Timing Choice

During the established early critical period, a small uniform prior may reduce
excessive class separation not addressed by example-pair interpolation, while
the complete 35% hard-label tail restores exact class targets and terminal
boundary refinement. The testable hypothesis is that this fixed treatment
reaches `best_test_acc >= 94.58%` while retaining at least 127 dataset passes.

Only early-only placement is admissible:

| Placement | 0-65% counted time | 65-100% counted time | Decision |
|---|---|---|---|
| Whole-run | mixup plus smoothing | smoothed hard CE | Reject: removes the locally validated exact-label tail. |
| **Early-only** | **mixup plus smoothing** | **accepted hard CE** | **Test once.** |
| Hard-tail-only | accepted mixup | smoothed hard CE | Reject: regularizes exactly where local timing evidence requires clean refinement. |

The cutoff is the existing strict time predicate `progress <
MIXUP_END_FRACTION`, not an epoch approximation. Do not ramp epsilon, bridge
the cutoff, smooth ordinary inputs, change alpha, alter the Beta draw or
permutation, or change the iterator-boundary RandAugment transition. EXP004
and EXP020 show that 50% and 75% mixup windows both lose at normal exposure;
the 65% temporal boundary is protected, not a tunable companion variable.

## Exact Dense-Target and Gradient Semantics

Let `K = 10`, `u_k = 1/K`, `epsilon = 0.05`, batch-shared mix coefficient
`lambda`, and one-hot labels `e_a` and `e_b`. The accepted early objective is

```text
y_mix = lambda * e_a + (1 - lambda) * e_b
L_accepted = CE(logits, y_mix).
```

The candidate must be exactly

```text
q = (1 - epsilon) * y_mix + epsilon * u
L_candidate = CE(logits, q)
            = lambda * CE_LS(logits, a, epsilon)
              + (1 - lambda) * CE_LS(logits, b, epsilon).
```

This is PyTorch's uniform-over-all-classes convention, not an
`epsilon/(K-1)` convention. For a different-class pair:

```text
q_a = 0.95 * lambda + 0.005
q_b = 0.95 * (1 - lambda) + 0.005
q_k = 0.005, k not in {a, b}.
```

For a same-class pair, the coincident terms combine, giving target mass
`0.955` on that class and `0.005` on each other class, independent of lambda.
Every target row must sum to one. The production implementation should retain
the two integer-target CE calls above rather than materialize `q`; the dense
form is the independent semantic oracle.

For one unreduced example with softmax probabilities `p`, the logit gradient
is

```text
dL_candidate/dlogits = p - q
                      = (p - y_mix) + epsilon * (y_mix - u).
```

For PyTorch's default mean over a batch of size `B`, each row contributes
`(p-q)/B` to the batched logit gradient.

Thus the intervention leaves inputs and forward logits unchanged but weakens
the drive toward the mixed target and adds a uniform-prior correction. It
changes all parameter data gradients through this signal; it must not detach,
renormalize, clamp, symmetrize, or use a different lambda in either CE term.
Weight decay, momentum buffers, Nesterov ordering, optimizer groups, and the LR
curve remain accepted. In the hard tail, loss, logits, gradients, optimizer
updates, BN state, and RNG must be accepted-identical.

## Scope and Semantic Qualification

Production scope is one scalar constant and two `label_smoothing` keyword
arguments inside the existing early mixup branch. There is no model state,
extra forward/backward, activation, random draw, data transform, evaluation,
or inference change. Parameter count must remain `1,003,482`; initialization,
setup RNG, Beta sampling, permutation sequence, data-worker RNG, finite guard,
single backward/step, decay allocation, and every-fifth-plus-final evaluation
cadence must remain accepted.

Before any score, an ignored evaluator-free preflight must print diagnostics
before assertions and fail closed unless it proves:

1. The diff from `git show a7c42dc:train.py` is limited to the constant and two
   early CE keywords; `prepare.py`, all accepted hyperparameters, model,
   optimizer, data, transition, timing, RNG, and evaluator code are unchanged.
2. Accepted and candidate construction from cloned RNG states produces
   byte-identical named parameters/buffers, optimizer state/groups, logits,
   setup RNG, and parameter count.
3. On fixed same-class and different-class early fixtures using one fixed
   permutation and lambda, production paired CE agrees with both the explicit
   two-smoothed-CE expression and an independent dense
   `-(q * log_softmax).sum(-1).mean()` oracle within preregistered FP32 bounds.
   Inputs, logits, BN updates, and RNG equal accepted.
4. Dense rows have the exact masses above, sum to one, and have minimum mass
   0.005. Autograd logit gradients agree with mean-reduced `p-q`; full parameter
   gradients and one fresh plus one preseeded-momentum Nesterov update agree
   with an independent oracle and are finite/nonzero.
5. Fixed hard-tail fixtures give byte-identical accepted/candidate loss,
   gradients, updates, BN state, and RNG. Static and boundary probes prove
   smoothing occurs iff the existing strict mixup predicate is true, including
   exact behavior immediately below and at 0.65.
6. Restored-state replay is deterministic, label smoothing consumes no CPU or
   CUDA RNG, and the accepted Beta/permutation trajectory is unchanged.

Any semantic failure closes before scoring. Only a demonstrable verifier or
implementation defect may be repaired; epsilon, target convention, temporal
placement, and all accepted semantics are frozen.

## Timing and Exposure Gate

Although the extra uniform log-probability reduction should be cheap relative
to convolution, cost must be measured rather than assumed. On one idle H20,
time complete production-equivalent accepted and candidate steps in both early
mixup and hard regimes. Use at least 20 warmups and counterbalanced `A/C/C/A`
blocks providing four retained windows per arm and regime, each at least 50
synchronized steps. Each window must include H2D, LR writes, zeroing, mixup
where applicable, forward, exact loss and finite guard, backward, coupled
Nesterov step, and final synchronization, restoring identical model,
optimizer, fixture, and RNG state before every window.

Require every window CV `<=5%`, paired-ratio population CV `<=1%` in each
regime, candidate peak allocation `<2048 MiB`, and every paired projected
whole-run retention to pass

```text
retention_i =
    (0.65 / candidate_early_i + 0.35 / candidate_hard_i) /
    (0.65 / accepted_early_i  + 0.35 / accepted_hard_i)

retention_i >= 127 / 130.304 = 0.9746439096
median_projected_passes = 130.304 * median(retention_i) >= 127.0.
```

Interleave early and hard measurements in reproducible local blocks or assess
all early-by-hard pair combinations so unrelated drift cannot be hidden by
index pairing. A stable timing miss closes this exact implementation without
an accuracy score.

## Sole Score, Gates, and Closure

After qualification passes, run exactly one fixed-seed score on one NVIDIA H20:

```bash
timeout 600s uv run train.py > run.log 2>&1
```

Require exit zero, one finite final summary, `300.0-300.1` counted training
seconds, total wall time below 600 seconds, correct single mixup and
iterator-exhausted RandAugment transitions, unique compliant evaluations,
unchanged parameter count, no error signature, and realized exposure

```text
num_steps * 256 / 50000 >= 127.0 passes.
```

The decision is conjunctive:

- **Improvement:** `best_test_acc >= 94.58%` **and** realized passes `>=127`.
- **No improvement:** a valid run with `best_test_acc <94.58%`; final accuracy
  or loss cannot rescue it.
- **Mechanism unsupported:** any valid run below 127 passes, even if it remains
  the sole recorded goal result; do not rerun.

A valid normal-exposure miss closes this exact early-only, epsilon-0.05,
PyTorch-uniform treatment and immediate label-smoothing work on the accepted
learner. Do not rescue it with epsilon 0.01/0.02/0.1, ramps, whole-run or
hard-tail placement, a bridge window, mixup replacement, alpha/cutoff changes,
class-dependent priors, confidence penalties, `K-1` semantics, RNG
realignment, a second seed, or another score. Label smoothing as a universal
method would remain formally unrefuted, but revisiting it here would require a
new independent calibration diagnosis rather than this result.

## Evidence and Risk Summary

- Label smoothing can reduce overconfidence, and early-only regularization is
  consistent with the saved label-smoothing and time-matters notes.
- The strongest local counterevidence is redundancy: accepted early mixup is
  already soft and locally calibrated in strength and duration.
- EXP041-044 close auxiliary-loss, adaptive-readout, gradient-projection, and
  spatial-dispersion directions, but their failures do not causally support
  smoothing. They justify only its current opportunity cost, not confidence.
- Epsilon 0.05 has no local bracket. Its one-shot value is acceptable only
  with the strict no-rescue closure contract.
- The fixed-seed 0.10-point hurdle corresponds to ten CIFAR-10 test examples;
  success establishes benchmark improvement under this protocol, not a
  seed-averaged effect.

Evidence used is offline/local only: the goal definition, system
understanding, experiment learnings and result index; accepted `a7c42dc`
`train.py`; saved label-smoothing, mixup, and time-dependent regularization
notes; EXP004/005/020/035/041 reports; and EXP044 proposal, adversarial review,
plan review, and final report. No evaluator output, test examples, network
search, or result-conditioned tuning informed the treatment.
