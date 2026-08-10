# Proposal: Bounded Soft-Target Poly-1

## Final Reviewer-Selected Specification

This section supersedes the initial positive-epsilon development below while retaining it as design history. Fix `POLY1_EPSILON=-0.25` before any local conditioning, timing, or accuracy result. For hard labels the optimizer-driving logit gradient is

```text
(1 - 0.25*p_y) * (p-e_y), with multiplier in [0.75,1].
```

The intervention is deliberately small at uncertain initialization (`0.975` at `p_y=0.1`) and material on solved examples (`0.875` at `p_y=0.5`, `0.775` at `p_y=0.9`). Its intended effect is a bounded 25% maximum confident-example attenuation that relatively prioritizes residual boundary examples without amplifying any hard-example gradient. This is an accuracy-blind effect-size choice, not a paper optimum; do not sweep, adapt, anneal, or change the magnitude or sign.

Use the identical negative-epsilon soft-target objective for every ordinary optimizer-driving loss, including area-corrected CutMix:

```text
L(q,z) = CE(q,p) - 0.25*(1-q dot p).
```

All initial soft-target calculus below remains valid after substituting `epsilon=-0.25`; constituent multipliers become `[0.75,1]`. The aggregate CutMix gradient ratio remains unbounded near CE cancellation. At `p=q` for unequal two-class targets, the equilibrium shift reverses relative to positive epsilon: gradient descent moves probability toward the minority constituent, reinforcing target softening rather than sharpening the majority. This can still over-soften clipped boxes and is a primary risk.

Preserve EXP-004's validated SAM adversary explicitly. On a scheduled clean SAM step, compute the first/base pass with the parent's plain hard CE and use that gradient only for the unchanged rho-0.05 perturbation. At the perturbed weights, compute hard negative-epsilon Poly-1; its gradient is the sole optimizer-driving gradient after exact restoration. Thus the experiment tests confidence-attenuated descent under the existing CE-defined adversarial neighborhood, rather than co-changing both the SAM ascent geometry and descent objective. Ordinary and CutMix steps use negative Poly-1 directly.

Do not replace the parent's `F.cross_entropy` with an FP32 NLL implementation. Preserve its CE numerics and add one `torch.softmax(logits, dim=1, dtype=torch.float32)` probability path for the Poly term; a suggested `log_softmax -> exp` reuse is rejected because true reuse would also change the CE implementation/precision, while computing both would add launches.

Correctness and production audits must distinguish `ordinary_poly`, `cutmix_poly`, `sam_ascent_ce`, and `sam_descent_poly`. Reconcile `ordinary_poly + cutmix_poly + sam_ascent_ce == num_steps`, `sam_ascent_ce == sam_descent_poly == sam_applied_batches`, and total loss calls `num_steps + sam_applied_batches`. Assert explicitly that CutMix/SAM intersection is empty. Scope the scalar multiplier identity and `[0.75,1]` bound only to hard-label `ordinary_poly` and `sam_descent_poly` calls, using a dtype-aware relative tolerance around `1e-3`; CutMix instead requires the full vector formula against autograd and never a scalar-ratio bound. The CE ascent must match parent CE, its gradients must be zeroed before the perturbed descent pass, and SAM perturbation norm/RNG replay/one-BN-update/restoration/one-optimizer-update and all EMA semantics remain unchanged.

Keep the first complete accuracy-blind latency gate decisive at median ratio `<=1.01`, every round `<=1.02`, projected at least 25,300 steps and 155 EMA samples, and no evaluator/test-loader access. One fixed metric run follows only after all formula/state/timing gates pass. Formal improvement remains `best_test_acc>=95.71`; stable mechanism support requires final-16 EMA mean `>=95.69`, at least 25,300 realized steps, at least 155 balanced EMA samples, and exact audits. A null rejects only this fixed negative-epsilon, CE-ascent/Poly-descent package.

The main causal risks after refinement are under-training confident class prototypes, excessive CutMix softening, and a mismatch between CE-defined SAM perturbations and Poly-defined descent. The sign is now aligned with the diagnosed residual-boundary limiter, but no external result establishes its effect size in this CutMix/SAM/EMA stack.

## Summary

Starting from EXP-011, replace every training cross-entropy call with one coherent
hard/soft-target Poly-1 objective:

```text
L(q, z) = CE(q, softmax(z)) + epsilon * (1 - q dot softmax(z))
epsilon = 0.25
```

Use the same objective for ordinary hard batches, area-corrected CutMix batches,
the first SAM pass that defines the perturbation, and the second SAM pass at the
perturbed parameters. Do not change the WRN-16-4, data order or augmentation,
CutMix decisions, drop path, optimizer, learning-rate schedule, SAM schedule,
EMA state or schedule, seed, evaluation, or time-budget boundary.

This is a fixed one-point experiment, not a coefficient sweep. The coefficient
comes from an accuracy-blind maximum 25% constituent-gradient inflation budget.
Its principal appeal is that it changes the training geometry on every update
while adding only a `256 x 10` FP32 softmax and gathers. Its principal weakness is
that unequal CutMix targets are no longer stationary at `p=q`: Poly-1 sharpens
the majority constituent and can partly oppose CutMix's intended softening.

## Evidence and Fit

The parent EXP-011 is the global best at `95.61%`. It completed `25,798` updates,
160 balanced EMA samples, and reached a final-16 EMA mean of `95.493125%`
(`95.44-95.61%`). The formal child threshold is therefore `95.71%`, but the
scientific target is a stable lift above the roughly `95.49%` EMA plateau rather
than another selected maximum.

The subsequent failures favor a low-overhead objective intervention. EXP-012's
extra spatial erasure reached `95.52%` with a `95.418125%` tail and slight dose
shortfall. EXP-013's fixed-scale cosine classifier was fully dosed but lowered
best accuracy to `95.11%` and its tail to `95.073750%`. EXP-014's stage-3 width
expansion was rejected before accuracy because its measured latency ratio was
`1.160975`, above its fixed gate. Poly-1 neither adds a second image perturbation
nor changes representation cost.

The ICLR 2022 PolyLoss paper motivates `CE + epsilon*(1-p_t)` and reports
classification gains, while explicitly making the useful coefficient
task-dependent. It does not justify transplanting a paper optimum into this
CutMix/SAM/EMA stack. This proposal therefore fixes a conservative coefficient
from gradient algebra and interprets failure as evidence only against this exact
`epsilon=0.25` package.

Expected impact is uncertain and plausibly modest. The intervention is strongest
on already-confident examples, so it may refine residual class boundaries, but a
stable `0.15-0.20` point lift is an upside hypothesis rather than a result implied
by the paper.

## Exact Hard-Target Mathematics

For logits `z`, probabilities `p=softmax(z)`, and one-hot target `e_y`, define

```text
L_hard = -log(p_y) + epsilon*(1-p_y).
```

The logit gradient is exactly

```text
d CE / dz                    = p - e_y
d [epsilon*(1-p_y)] / dz     = epsilon*p_y*(p-e_y)
d L_hard / dz                = (1 + epsilon*p_y)*(p-e_y).
```

With `epsilon=0.25`, each hard-example CE logit gradient retains its direction
and receives multiplier

```text
m_y = 1 + 0.25*p_y,       1 <= m_y <= 1.25.
```

The exact mean multiplier on any hard batch is
`1 + 0.25*mean(p_y)`. At uniform ten-class initialization it is `1.025`; at
`p_y=0.5` it is `1.125`; at `p_y=0.9` it is `1.225`. This is not a global learning
rate increase: examples receive different factors before their parameter
gradients are aggregated.

## Exact CutMix Soft-Target Mathematics

For original class `a`, paired class `b`, and the existing clipped-area original
weight `lambda=adjusted_lam`, define

```text
q   = lambda*e_a + (1-lambda)*e_b
p_t = q dot p = lambda*p_a + (1-lambda)*p_b

L_mix = -lambda*log(p_a) - (1-lambda)*log(p_b)
        + epsilon*(1-p_t).
```

This is exactly the weighted sum of the two constituent hard Poly-1 losses:

```text
L_mix = lambda*[-log(p_a) + epsilon*(1-p_a)]
      + (1-lambda)*[-log(p_b) + epsilon*(1-p_b)].
```

It must use `adjusted_lam` returned after clipped box-area measurement, never the
sampled pre-clipping lambda. No dense target tensor is needed. The exact gradient
is

```text
d L_mix / dz
  = p - q + epsilon * p * (p_t - q)
  = lambda*(1+epsilon*p_a)*(p-e_a)
    + (1-lambda)*(1+epsilon*p_b)*(p-e_b),
```

where the first `*` in the middle expression is elementwise. Componentwise,

```text
g_a = p_a-lambda     + epsilon*p_a*(p_t-lambda)
g_b = p_b-(1-lambda) + epsilon*p_b*(p_t-(1-lambda))
g_j = p_j            + epsilon*p_j*p_t, j not in {a,b}.
```

Each constituent multiplier is in `[1,1.25]`. The combined CutMix norm ratio
`||g_poly1||/||p-q||` is not bounded by `1.25`, because the two CE constituent
vectors may cancel while their differently scaled versions do not. Correctness
tests must never apply the hard-label bound to the aggregate soft-target ratio.

Endpoints are unambiguous: `lambda` equal to zero or one becomes hard Poly-1; a
same-class pair becomes hard Poly-1 at every lambda; a distinct-class pair with
`lambda=0.5` preserves symmetry. At `p=q` for two distinct classes, the Poly-only
gradient on classes `a,b` is

```text
epsilon*lambda*(1-lambda)*(1-2*lambda),
epsilon*lambda*(1-lambda)*(2*lambda-1),
```

respectively. It vanishes at `lambda=0.5`; otherwise gradient descent moves mass
toward the majority target. This deliberate consequence is the main interaction
with CutMix, not an implementation artifact.

## Fixed Coefficient

Preregister

```text
POLY1_MAX_CONSTITUENT_INFLATION = 1.25
POLY1_EPSILON = POLY1_MAX_CONSTITUENT_INFLATION - 1.0  # 0.25
```

Because every constituent probability is in `[0,1]`, its multiplier is provably
in `[1,1.25]`. A 25% ceiling is large enough to differentiate confident examples
while avoiding the 2x maximum multiplier from `epsilon=1` in a recipe whose LR,
SAM radius, and EMA were validated with CE. Initially uniform examples change by
only 2.5%. This is an optimizer-compatibility bound, not an estimate of the best
CIFAR-10 coefficient.

Do not sweep, anneal, normalize, change the sign, make epsilon phase-dependent,
or select another value after timing or accuracy. A failed run rejects only this
fixed package.

## Production Code Shape

Add one helper in `train.py` and route every training loss call through it:

```python
def soft_target_poly1_loss(logits, targets_a, targets_b=None, lam=1.0):
    probabilities = logits.float().softmax(dim=1)
    probability_a = probabilities.gather(1, targets_a[:, None]).squeeze(1)
    ce_a = F.cross_entropy(logits, targets_a)

    if targets_b is None:
        cross_entropy = ce_a
        target_probability = probability_a
    else:
        probability_b = probabilities.gather(1, targets_b[:, None]).squeeze(1)
        cross_entropy = lam * ce_a
        cross_entropy += (1.0 - lam) * F.cross_entropy(logits, targets_b)
        target_probability = lam * probability_a + (1.0 - lam) * probability_b

    poly1 = POLY1_EPSILON * (1.0 - target_probability).mean()
    return cross_entropy + poly1
```

The explicit FP32 softmax avoids BF16 probability quantization while preserving
the parent's cross-entropy implementation. `lam` is the existing Python float,
so this adds no host transfer. The helper consumes no RNG, creates no persistent
state, and changes neither model parameter/state inventory nor evaluation loss.

Use it in the primary forward for both hard and CutMix batches. Use it again with
the hard target in the SAM second pass. Preserve the existing no-overlap assertion:
production SAM begins when CutMix ends, but both SAM passes must still use Poly-1.
Using CE in either pass would create an unplanned hybrid objective.

Keep the operation inside the current charged `t0 -> synchronize -> dt` region.
Print `loss=soft_target_poly1`, `poly1_epsilon=0.25`,
`max_constituent_inflation=1.25`, and `poly1_probability_dtype=float32` in startup
configuration. Any diagnostic values used in the metric process must be detached
without new mid-step synchronization; heavy gradient checks belong only in the
preflight harness.

## SAM and EMA Interaction

The first hard Poly-1 SAM pass changes the aggregate parameter-gradient direction
through confidence-dependent example weighting. SAM's global normalization still
makes the perturbation norm exactly `rho=0.05`; it does not preserve perturbation
direction. The second perturbed Poly-1 pass changes both gradient direction and
magnitude. Preserve the parent's CUDA RNG replay, one BatchNorm update, exact base
parameter restoration, and one Nesterov/momentum update.

EMA remains a passive full-state summary of the altered online trajectory. Its
cadence, time-derived decay, parameter/buffer coverage, integer-buffer policy,
ordinary/SAM parity, and evaluation swap are unchanged. Poly-1 may move the online
trajectory into a better basin, or EMA may attenuate/lag the effect. Because late
evaluation observes EMA rather than a simultaneous live control, the metric run
cannot isolate those mechanisms.

## Accuracy-Blind Correctness Gates

Before any metric launch, require one clean preflight on physical GPU 0 with
`CUDA_VISIBLE_DEVICES=0`. It may inspect training data and synthetic inputs but
must not iterate the test loader, call the evaluator, or reveal test accuracy.

The preflight must pass all of these gates:

1. FP64 hard-label loss and autograd gradients match the closed form at uniform,
   low-, medium-, and high-confidence logits; the measured multiplier matches
   `1+0.25*p_y` and remains in `[1,1.25]`.
2. For CutMix lambda in `{0, 0.01, 0.25, 0.5, 0.75, 0.99, 1}`, including
   same-class and zero-area cases, the sparse helper matches both a dense-`q`
   implementation and the weighted constituent implementation in value and
   gradient. The analytic gradient matches autograd. Demonstrate, rather than
   reject, at least one cancellation case whose aggregate ratio exceeds `1.25`.
3. With test-only epsilon set to zero, helper loss and gradients match parent CE
   to numerical tolerance. Calling the helper leaves CPU and CUDA RNG states
   byte-identical.
4. CPU FP32 and GPU-0 BF16/channels-last full-WRN forward/backward paths are finite;
   probabilities and Poly term are FP32; parameter and persistent-state inventory
   remains exactly the parent's `2,748,890` parameters and state keys.
5. A scheduled SAM smoke proves helper invocation once at base and once at
   perturbed parameters, exact drop-path RNG replay, one BN-stat update, perturbation
   norm `0.05` within tolerance, exact parameter restoration, and one optimizer
   update. Ordinary and SAM cadence-31 EMA smokes retain eligibility, parity,
   state coverage, RNG, swap/restore, and optimizer identity.
6. A fixed 200-step real-CIFAR conditioning trace gives parent and candidate the
   same batches, augmentations, CutMix choices, SAM schedule, and EMA eligibility.
   Candidate divergence after its first update is expected, but all losses,
   gradients, parameters, and buffers must stay finite and constituent bounds must
   hold. Accuracy and loss differences are informational and cannot tune epsilon.

Only exceptions, malformed/missing output, or assertion defects before a complete
numeric gate vector qualify as harness errors. Once complete numeric timing or
stability gates print, their pass/fail result is decisive; do not relabel an
unfavorable value as a harness problem.

## Accuracy-Blind Latency Gates

After warmup, run five alternating parent/candidate rounds covering the production
mixture of early hard, early CutMix, late ordinary, late SAM, and cadence-31 EMA
work. Use identical input shapes and decisions; do not include test evaluation in
the charged path. Weight path timings by the parent schedule rather than selecting
the fastest microbenchmark.

Proceed only if all conditions hold on the first complete gate vector:

- physical GPU 0 resolves to the approximately 98 GB NVIDIA H20;
- parent round drift `(max-min)/median <= 0.03`;
- paired-ratio median absolute deviation divided by median `<= 0.005`;
- median production-weighted candidate/parent latency ratio `<= 1.01`, and no
  individual complete round ratio exceeds `1.02`;
- same-harness projections are at least `25,300` optimizer steps, 155 EMA samples,
  and 130 epochs, with projected total runtime below 600 seconds;
- candidate peak allocated memory is below `1,350 MiB` and all correctness gates
  remain passing.

The latency hypothesis is `1.002-1.008x`: roughly 28,269 loss calls at the parent
dose each add only a 2,560-element softmax and gathers. The gates, not this
estimate, decide feasibility. A numeric gate failure terminates EXP-015 before
accuracy. There is no optimized helper fallback, epsilon change, diagnostic
removal, or timing rerun.

## One Metric Launch and Decision Rules

After and only after a complete preflight pass, launch exactly once:

```bash
timeout 600s env CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1
```

Formal tree success requires all of the following:

- `best_test_acc >= 95.71%`, exactly 9,571 or more correct examples on the fixed
  10,000-example test set;
- exit zero, complete final summary, `299.5-301.0` charged training seconds, total
  runtime below 600 seconds, and at most one evaluation per epoch;
- physical GPU 0, fixed seed 42, and only `train.py` changed;
- no formula, range, RNG, state, restoration, nonfinite, CUDA, OOM, or timeout
  failure.

Preregister the parent-relative tail interpretation separately:

```text
parent best                         = 95.61
formal child threshold              = 95.71  (+0.10 vs parent best)
parent final-16 EMA mean            = 95.493125
stable-support threshold            = 95.643125 (+0.15 vs parent tail)
strong stable-support threshold     = 95.693125 (+0.20 vs parent tail)
```

Mechanism support requires at least `25,300` realized optimizer steps, at least
155 EMA samples with ordinary/SAM imbalance at most one, and final-16 EMA mean at
least `95.643125%`. Strong support requires the `95.693125%` tail threshold. If
the formal best passes but the tail mean is below `95.643125%`, record a tree
improvement but weak evidence for a stable Poly-1 gain.

The exact package is formally falsified as an improving child if best accuracy is
below `95.71%`. At adequate realized dose, its stable-tail hypothesis is falsified
if the final-16 EMA mean is at or below the parent's `95.493125%`; a mean strictly
between `95.493125%` and `95.643125%` is inconclusive for stability. A realized
dose miss never changes the formal metric verdict, but prevents a clean full-dose
causal interpretation. Do not rerun, change epsilon, disable Poly-1 on one phase,
or choose checkpoints after observing accuracy.

## Risks and Causal Limits

- Poly-1 upweights confident hard examples. This may improve residual boundaries,
  but may instead spend capacity on easy examples, worsen calibration, or reduce
  attention to hard/mislabeled examples.
- Unequal CutMix targets are sharpened. Although each constituent is bounded, the
  aggregate soft-target gradient ratio is unbounded near CE cancellation and can
  partly undo a mechanism that already contributed to the best branch.
- The experiment changes hard loss, CutMix soft loss, both SAM passes, the SAM
  perturbation direction, and the trajectory consumed by EMA. One metric run
  cannot attribute an outcome among those interactions.
- SAM preserves perturbation radius but not direction. EMA can smooth, lag, or
  amplify the changed trajectory; no contemporaneous live tail control exists.
- Even tiny added latency changes update count and phase dose under a wall-clock
  budget. Tail and best deltas must be interpreted with realized steps, epochs,
  SAM pulses, and EMA samples.
- FP32 softmax is numerically safer than BF16 but duplicates work already implicit
  in CE. Kernel launch overhead, not arithmetic, could still trip the strict gate.
- A 25% constituent ceiling is principled as a safety bound but has no evidence of
  being CIFAR-10-optimal. A null result does not reject other coefficients; a gain
  does not establish optimality.
- This is one fixed-seed candidate compared with a historical parent. Sub-0.30
  point differences are within observed checkpoint/run resolution. The helper
  does not change setup construction or consume RNG, and the preflight must verify
  matched streams, but one run still cannot establish statistical significance.
- `best_test_acc` selects a maximum over many evaluations. The preregistered
  final-16 mean is required to distinguish stable movement from a favorable peak,
  but it is diagnostic and cannot override the formal tree rule.

## Recommendation

Advance this candidate with medium confidence in feasibility and low-to-medium
confidence in clearing `95.71%`. It is mechanism-distinct from the three failed
children, has mathematically exact hard and soft-target behavior, and should
preserve almost all optimizer exposure. The explicit CutMix sharpening and modest
bounded effect size are the reasons to regard it as a disciplined test rather
than a likely breakthrough.
