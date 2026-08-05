# Proposal: Gamma-1 Focal Loss Only in the Hard-Label Tail

## Exact Treatment

Keep accepted `a7c42dc` unchanged through the existing 65% mixup cutoff. Only
replace the clean hard-target CE branch with standard, unnormalized multiclass
focal loss at exactly gamma 1:

```python
def focal_cross_entropy(logits, targets):
    per_example_ce = F.cross_entropy(logits, targets, reduction="none")
    hard_example_weight = 1.0 - torch.exp(-per_example_ce)
    return (hard_example_weight * per_example_ce).mean()
```

The accepted mixup arm remains exactly:

```python
loss = mix * F.cross_entropy(outputs, targets_a) + (
    1.0 - mix
) * F.cross_entropy(outputs, targets_b)
```

For example `i`, with `p_i = softmax(logits_i)[target_i]` and
`CE_i = -log(p_i)`, the new tail objective is

```text
L = mean_i [(1 - p_i) CE_i].
```

Do not detach the focal weight, normalize weights, add alpha/class weighting,
blend CE, smooth labels, ramp focal loss, change the cutoff, reset momentum,
or rephase LR. The first non-mixup step uses focal loss. Preserve the separate
exhausted-epoch RandAugment transition, even though this means a few hard-label
focal steps may still see RandAugment before the current iterator exhausts.
All model, pooled-head, RNG, optimizer, data, schedule, evaluation, and budget
settings remain accepted.

## Mechanism and Evidence

The accepted learner nearly interpolates its tail (smoothed train loss about
0.0028) but ends at 94.45% accuracy and 0.2456 evaluator CE. Focal loss spends
less of the fixed late budget reinforcing already-confident examples and more
on ambiguous or misclassified examples, without adding model compute.

For one example its logit gradient is exactly

```text
grad FL = q(p) * grad CE
q(p) = (1 - p) - p log(p) = (1 - p) + p CE.
```

Thus gradient direction is unchanged per example: very easy examples approach
zero weight, badly wrong examples approach CE weight, and intermediate cases
can be modestly amplified. This is confidence-based example reweighting, not
label smoothing; one-hot targets remain intact. It therefore avoids stacking
a second soft-target mechanism on accepted early mixup, consistent with the
local label-smoothing note, while the local time-regularization note supports
isolating the change to the post-mixup phase.

The only extra hard-step work is vector CE reduction, one elementwise `exp`,
multiplication, and mean. No parameter, activation, loader, or evaluator path
changes.

## Verification Gates

Use an evaluator-free harness and independent `git show a7c42dc:train.py`
oracle. Before timing or scoring require:

- production diff is only the helper plus replacement of the hard-branch CE;
  `prepare.py` and all other production files are byte-identical;
- topology, all initial parameter/buffer bytes, 1,003,482 parameter count,
  construction CPU/CUDA RNG, optimizer, transforms, schedule, and cadence are
  exact;
- on fixed confident, ambiguous, badly-wrong, and large finite logit fixtures,
  loss matches independent
  `-mean((1 - exp(log_p_t)) * log_p_t)` within
  `rtol=1e-6, atol=1e-7` and remains finite/nonnegative;
- autograd matches both the independent formula and per-example CE gradients
  scaled by `q(p)` within the same bounds, explicitly detecting detached
  weights;
- cloned accepted/candidate early mixup steps are bitwise equal for inputs,
  lambda, permutation, logits, loss, gradients, parameters, optimizer/BN state,
  and terminal RNG;
- cloned hard forwards are bitwise equal before loss construction; candidate
  loss/gradient matches the focal oracle, differs nontrivially from CE, and its
  Nesterov update matches an independent optimizer oracle;
- restoring candidate model/optimizer/input/RNG reproduces a full hard step
  bitwise;
- probes below/at/above 65% prove accepted mixup CE below the boundary and
  focal above it with unchanged LR, no new state transition, and accepted
  exhausted-epoch RandAugment ordering;
- evaluation still uses evaluator CE at most once per epoch. Final test loss
  remains comparable; smoothed train loss does not and cannot support verdict.

Print measurements before assertions. Fail closed rather than adding clamps,
epsilons, detachment, normalization, CE mixing, or looser tolerances.

For throughput, compare restored accepted/candidate complete early and hard
steps on one idle H20, including transfer, LR writes, zeroing, mixup when
active, forward/loss/backward, finite check, Nesterov step, and synchronization.
Use at least 20 warmups and four counterbalanced windows of at least 50 steps
per arm/regime with fresh deterministic fixtures. Require finite state and
population CV <=5% for every arm/regime. From median seconds per step compute

```text
retention = (0.65 / candidate_early + 0.35 / candidate_hard) /
            (0.65 / accepted_early  + 0.35 / accepted_hard)
projected_passes = 130.304 * retention.
```

Proceed only if retention >=0.9746439096 and projected passes >=127.000. A
stable miss is final; do not rerun or alter focal normalization.

## Sole Score and Closure

After gates pass, confirm one local H20, frozen files, no stale log, accepted
base, and exact diff. Run once at fixed seed 42:

```bash
timeout 600s uv run train.py > run.log 2>&1
```

Require exit zero, finite summary, training seconds in `[300.0, 300.1]`, wall
time below 600 seconds, 1,003,482 parameters, accepted transition/evaluation
ordering, and no runtime error. Record `num_steps * 256 / 50000` passes. A
completed run below 127 passes remains the sole valid score but is
exposure-confounded and cannot be rerun.

Success is only `best_test_acc >= 94.58%`, +0.10 over 94.48%. Final accuracy
>=94.45% and evaluator loss <=0.2456 are non-decisive corroboration: neither
rescues a miss nor vetoes a success.

Risks are substantive: CIFAR-10 is balanced; hard examples may be noisy or
atypical crops; suppressing easy examples can erode broad class margins; and
unnormalized focal loss lowers aggregate gradient scale near interpolation,
potentially undoing the benefit of the accepted nonzero LR floor. Existing
Nesterov buffers also cross the objective switch unchanged, deliberately, to
keep this an isolated loss test. One fixed seed cannot estimate average effect.

A valid >=127-pass miss falsifies standard unnormalized gamma-1 focal loss as
a useful standalone hard-tail refinement here. Retain CE and close immediate
rescues: gamma 0.5/2, detached or normalized weights, alpha, CE blending,
focal-on-mixup, shifted/ramped cutoff, momentum reset, LR compensation,
another seed, or a rerun require new independent evidence. This does not claim
all focal formulations are universally ineffective; it closes this local
confidence-reweighting program. A pre-score failure closes only the exact
implementation or feasibility condition that failed.

## Falsifiable Hypothesis

If repeated easy-example CE is the remaining tail inefficiency, gamma-1 focal
loss only after mixup will preserve at least 127 projected and realized passes
and raise fixed-seed best accuracy from 94.48% to at least 94.58%, with final
accuracy >=94.45% and evaluator CE <=0.2456 as support. A normal-exposure miss
falsifies that hypothesis.

Local evidence: `02-system-understanding.md`, `03-experiment-learnings.md`,
`experiments/036/04-analysis.md`, `knowledge/papers/label-smoothing.md`, and
`knowledge/papers/time-matters-regularization.md`.
