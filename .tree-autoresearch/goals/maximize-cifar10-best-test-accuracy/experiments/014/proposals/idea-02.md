# Proposal: Bounded Soft-Target Poly-1 on EXP-011

## Summary

Change only the training classification objective on the EXP-011 global-best
branch. Replace hard-label and area-corrected CutMix cross-entropy with one
coherent Poly-1 loss,

```text
L(q,z) = CE(q, softmax(z)) + epsilon * (1 - q dot softmax(z)),
epsilon = 0.25.
```

Use the identical objective in the ordinary first pass, the first SAM pass that
defines the perturbation, and the second SAM pass at perturbed weights. Preserve
the WRN-16-4, random crop/flip, drop path, CutMix schedule and private generators,
SGD/Nesterov settings, wall-clock LR schedule, clean-tail period-two SAM,
cadence-31 full-state EMA, evaluator, seed, batch size, and all timing boundaries.

The coefficient is preregistered from a maximum 25% hard-label/constituent
gradient-inflation budget, not copied from a task-dependent paper optimum and not
selected with CIFAR-10 test accuracy. This is a low-compute loss intervention,
but it is not fully orthogonal to CutMix: on unequal two-class soft targets it
systematically sharpens the majority target and changes the target optimum.

## Motivation and Evidence

EXP-011 remains the global best at `95.61%`, with `25,798` updates, 160 balanced
EMA samples, and negligible EMA overhead. Its final 16 EMA checkpoints averaged
`95.493125%`, ranged `95.44-95.61%`, and ended at `95.46%`; the child threshold is
`95.71%`. The diagnosed limiter is therefore a stable generalization or
decision-boundary improvement rather than throughput, evaluation frequency, or
memory capacity (`02-system-understanding.md`; `experiments/011/04-analysis.md`).

The two children of EXP-011 do not test this mechanism. Complementary Cutout
reached only `95.52%` and missed its realized-dose floor, while a fully dosed
fixed-scale cosine classifier produced a tight `95.07%` EMA tail and `95.11%`
best. Those outcomes argue against another spatial erasure or fixed cosine
geometry attempt, not against a low-overhead confidence-dependent loss
(`experiments/012/04-analysis.md`; `experiments/013/04-analysis.md`).

PolyLoss represents classification losses in a polynomial basis and proposes
Poly-1 as `CE + epsilon*(1-p_t)`. The paper reports image-classification gains but
also makes the useful coefficient task-dependent. The goal knowledge therefore
supports a single bounded coefficient and exact soft-target analysis rather than
a published-value transplant or test-set sweep
(`experiments/014/papers/poly-loss.md`; `knowledge/papers/polyloss.md`).

This idea has medium mechanistic fit and low implementation cost, but only
moderate expected impact. The conservative bound makes optimizer disruption
small; it also makes a stable 0.25-point gain an upside outcome rather than a
well-supported median prediction.

## Exact Hard-Label Objective and Gradient

Let `z` be one example's logits, `p = softmax(z)`, and `e_y` the one-hot hard
target. Define

```text
L_hard(z,y) = -log(p_y) + epsilon * (1-p_y).
```

Since

```text
d CE / d z              = p - e_y
d [epsilon*(1-p_y)]/d z = epsilon*p_y*(p-e_y),
```

the exact candidate gradient is

```text
d L_hard / d z = (1 + epsilon*p_y) * (p-e_y).
```

Thus Poly-1 preserves each hard example's CE logit-gradient direction and
rescales it by

```text
m_y = 1 + 0.25*p_y,    1 <= m_y <= 1.25.
```

At a uniform ten-class prediction `m_y=1.025`; at `p_y=0.5`, `m_y=1.125`; and at
`p_y=0.9`, `m_y=1.225`. The intervention is weak during uncertain early training
and increasingly upweights confident examples. It is not equivalent to one
global LR increase because examples receive different multipliers and their
parameter gradients aggregate after this reweighting.

## Exact CutMix Soft-Target Objective and Gradient

For the original class `a`, paired class `b`, and the existing area-corrected
original-label weight `lambda`, define

```text
q   = lambda*e_a + (1-lambda)*e_b
p_t = q dot p = lambda*p_a + (1-lambda)*p_b

L_mix = -lambda*log(p_a) - (1-lambda)*log(p_b)
        + epsilon*(1-p_t).
```

This is exactly the weighted sum of the two constituent hard Poly-1 losses:

```text
L_mix = lambda * [-log(p_a) + epsilon*(1-p_a)]
      + (1-lambda) * [-log(p_b) + epsilon*(1-p_b)].
```

It uses `adjusted_lam` returned after clipped box-area measurement, never the
sampled pre-clipping lambda, and needs no dense target tensor. The exact logit
gradient is

```text
d L_mix / d z
  = p - q + epsilon * p elementwise_multiplied_by (p_t - q)

  = lambda*(1+epsilon*p_a)*(p-e_a)
    + (1-lambda)*(1+epsilon*p_b)*(p-e_b).
```

Componentwise this is

```text
g_a = p_a-lambda       + epsilon*p_a*(p_t-lambda)
g_b = p_b-(1-lambda)   + epsilon*p_b*(p_t-(1-lambda))
g_j = p_j              + epsilon*p_j*p_t,  j not in {a,b}.
```

Each constituent multiplier lies in `[1,1.25]`, but this does **not** bound
`||g_mix|| / ||p-q||` by 1.25. The two constituent CE vectors can cancel, making
the parent soft-gradient norm arbitrarily small while the differently weighted
Poly terms remain nonzero. The audit must report small denominators and the
combined ratio descriptively; it must never use 1.25 as a pass bound for the
combined CutMix gradient.

Endpoint and degeneracy behavior is exact. `lambda` equal to zero or one reduces
to hard Poly-1. A same-class pair also reduces to hard Poly-1 regardless of
lambda. At `lambda=0.5`, the symmetric two-label case preserves symmetry.

## Coefficient Preregistration

Fix

```text
POLY1_MAX_CONSTITUENT_INFLATION = 1.25
POLY1_EPSILON = POLY1_MAX_CONSTITUENT_INFLATION - 1 = 0.25.
```

For any hard target or CutMix constituent `k`, `0 <= p_k <= 1`, so

```text
1 <= 1 + epsilon*p_k <= 1 + epsilon = 1.25.
```

The 25% ceiling is an accuracy-blind optimizer-compatibility choice. It is large
enough to alter confident-example weighting, yet avoids the 2x worst-case
constituent multiplier from `epsilon=1` in a recipe whose LR, SAM radius, and EMA
were validated under CE. It leaves initially uniform examples within 2.5% of
their parent gradient scale. This is a bounded engineering hypothesis, not an
estimate of the optimal CIFAR-10 coefficient.

Do not sweep, anneal, normalize, sign-change, or adapt epsilon. Do not choose a
second coefficient after preflight or metric results. A failure rejects only
this fixed `epsilon=0.25` package, not PolyLoss generally.

## Intended `train.py` Change

Implement one loss helper used by every training loss call. Preserve the current
scalar `F.cross_entropy` calculations for the CE component and compute only the
Poly term from an explicit FP32 softmax over the `256 x 10` logits:

```text
p = softmax(logits.float(), dim=1)
p_a = gather(p, target_a)

hard:
    ce = cross_entropy(logits, target_a)
    p_t = p_a

CutMix:
    p_b = gather(p, target_b)
    ce = lambda*cross_entropy(logits, target_a)
       + (1-lambda)*cross_entropy(logits, target_b)
    p_t = lambda*p_a + (1-lambda)*p_b

loss = ce + 0.25*mean(1-p_t)
```

The explicit FP32 probability path avoids BF16 probability error while leaving
the parent's CE implementation intact. The loss returns audit values detached
from autograd. It must not allocate persistent model state, consume RNG, alter
targets, change evaluation loss, or move work outside the existing charged step.

Everything else in `train.py` remains EXP-011. In particular, do not change the
model, initialization order, parameter count (`2,748,890`), optimizer, LR,
weight decay, drop path, batch size, data stream, CutMix gate/box/permutation,
SAM configuration, EMA horizon/cadence/state inventory, or evaluation routing.

## SAM Semantics

Production CutMix ends at progress `0.75`, exactly when SAM eligibility starts,
so SAM sees hard targets. Poly-1 must still be present in **both** SAM passes:

1. Compute hard Poly-1 at the base parameters and backpropagate it. Use that
   gradient in the unchanged global Euclidean `rho=0.05` perturbation.
2. Clear gradients, restore the saved CUDA RNG state, disable second-pass BN
   tracking exactly as in EXP-011, and compute hard Poly-1 again at the perturbed
   parameters on the same input and target.
3. Backpropagate the perturbed Poly-1, restore the base parameters and BN flags,
   and perform the sole Nesterov update.

Using CE in either pass creates an unplanned hybrid. Keep one BN update, replayed
drop-path randomness, exact parameter restoration, one optimizer/momentum update,
and the existing no-CutMix/SAM-overlap assertion.

SAM normalizes the first aggregate parameter-gradient norm, so Poly-1 does not
change the perturbation's radius. It can change its direction by reweighting
examples and tensors. The unnormalized second-pass gradient then changes both
direction and magnitude. A result therefore measures Poly-1 composed with SAM,
not Poly-1 under ordinary SGD alone.

## Interaction and Overlap With CutMix

CutMix mixes both pixels and labels during the first 75% of charged time. Poly-1
does not add another image perturbation, but its soft-target behavior overlaps
with CutMix's label interpolation:

- CE is minimized at `p=q`; Poly-1 adds a linear reward for increasing `q dot p`,
  so the candidate generally has a different optimum.
- At `p=q` for two distinct labels, the Poly-only gradient on the two supported
  classes is

  ```text
  epsilon*lambda*(1-lambda)*(1-2*lambda) on class a,
  epsilon*lambda*(1-lambda)*(2*lambda-1) on class b.
  ```

  It is zero for `lambda=0.5` but otherwise gradient descent sharpens probability
  toward the majority target. Because clipped CutMix boxes often leave the
  original label as the majority, this can partially undo CutMix's softening.
- The constituent interpretation remains coherent and bounded, but the summed
  soft gradient is non-collinear with mixed CE and can have a large relative
  ratio near CE cancellation.
- Poly-1 also acts on early non-CutMix hard batches and on the entire clean
  SAM/EMA tail, where no soft-target overlap exists.

This overlap is the principal scientific risk. A gain could come from better
hard-example weighting, useful CutMix target sharpening, their interaction with
SAM, or EMA smoothing. A regression could likewise reflect excessive focus on
already confident examples or damage to CutMix calibration. This one run cannot
separate those explanations.

## Runtime and Memory Estimate

Each training loss call adds one FP32 softmax over 2,560 logits, one or two
gathers, and small vector reductions. There are `num_steps + sam_applied_batches`
loss calls: about `25,798 + 2,471 = 28,269` at the EXP-011 dose. The temporary
probability tensor is about 10 KiB in FP32 per call and no model-sized persistent
state is added.

The added work is launch-bound rather than arithmetic-bound. A reasonable
preflight hypothesis is `20-80 us` per call, or roughly `0.6-2.3` charged seconds
over the run before sparse audits. The expected weighted latency ratio is
`1.002-1.008`; peak allocation should remain near EXP-011's `1,222.4 MiB` and
below 1.30 GiB. These are estimates, not feasibility evidence.

Keep heavy analytic gradient-ratio audits sparse and deterministic so
observability does not become the main intervention. All routine diagnostic
accumulators remain on GPU and are read only after the existing per-step
synchronization or at terminal time; no new mid-step `.item()` synchronization
is allowed.

## Audit Contract

Print startup configuration identifying `loss=soft_target_poly1`,
`poly1_epsilon=0.25`, `max_constituent_inflation=1.25`,
`poly1_probability_dtype=float32`, and unchanged parent constants.

Maintain separate categories for ordinary primary hard calls, primary CutMix
calls, SAM first calls, and SAM second calls. For each, report calls/examples,
mean CE and Poly components, Poly/CE ratio, `p_t` min/mean/max, multiplier
min/mean/max, and nonfinite/range-failure counts. For CutMix also report adjusted
lambda min/mean/max, same-class count, zero-area count, and separate `m_a`/`m_b`
statistics.

On a fixed sparse cadence such as every 127th eligible call, compute detached
analytic logit gradients and record:

- hard `||g_poly1||/||g_ce||`, which must match `1+0.25*p_y` numerically;
- CutMix CE and candidate gradient norms;
- CutMix combined norm ratio when the CE norm exceeds a fixed FP32 denominator
  floor, with below-floor samples counted separately;
- maximum analytic-versus-autograd discrepancy in correctness smokes, not in the
  metric loop.

Reconcile terminal counts exactly:

```text
ordinary_primary_hard + primary_cutmix + sam_first = num_steps
primary_cutmix = cutmix_applied_batches
sam_first = sam_second = sam_applied_batches
all_loss_calls = num_steps + sam_applied_batches
```

Require all hard/constituent multipliers within `[1,1.25]`, all probabilities in
`[0,1]`, finite CE/Poly/loss values, and no audit failure. Do not impose a bound
on the combined CutMix gradient ratio.

Retain the complete EXP-011 audits for CutMix dose, SAM start/cadence and exact
restore, EMA update parity/state coverage/swap restore/RNG/mode/BN behavior,
evaluation source count, and complete final summary. Record evaluation progress
and preserve the final 16 EMA accuracies for plateau comparison.

## Correctness Smokes

1. In FP64, compare hard loss and autograd logits gradients with the closed form
   over uniform, low-, medium-, and high-confidence logits; require the exact
   multiplier and bound.
2. Compare CutMix loss and gradient against both a dense-`q` implementation and
   the weighted sum of constituent hard Poly-1 losses for `lambda` in
   `{0, 0.01, 0.25, 0.5, 0.75, 0.99, 1}`, including same-class pairs and clipped
   zero-area boxes.
3. Verify `p-q+epsilon*p*(p_t-q)` and the constituent decomposition against
   autograd. Explicitly demonstrate that the combined ratio may exceed 1.25
   under cancellation rather than treating it as a failure.
4. With epsilon forced to zero in a test-only reference helper, require parent CE
   loss and logits gradients to agree to numerical tolerance and verify no RNG
   consumption.
5. Run CPU FP32 and physical-GPU-0 BF16/channels-last full-WRN forward/backward
   smokes. Require finite outputs, FP32 probabilities/loss, finite gradients, and
   unchanged parameter/state inventory.
6. Instrument a scheduled SAM step to prove Poly-1 is invoked once at base and
   once at perturbed weights, drop-path RNG is replayed, BN updates once,
   perturbation norm remains `0.05`, parameters restore exactly, and one momentum
   update occurs.
7. Exercise ordinary and SAM cadence-31 EMA samples and live/EMA evaluation;
   require parent-identical eligibility, exact state restoration, optimizer
   identity, RNG, coverage, and evaluation routing.
8. Reconcile every loss/audit counter on a scripted mixed trace and compile/lint
   the sole tracked `train.py` change.

## Accuracy-Blind GPU-0 Preflight

Confirm `CUDA_VISIBLE_DEVICES=0` maps to the physical NVIDIA H20 with about
97,871 MiB. Use one preflight only, with harness-only corrections allowed before
any complete result; do not inspect test accuracy.

Run five alternating, warmed, production-faithful paired parent/candidate rounds
covering early hard, early CutMix, late ordinary, late SAM, EMA cadence work, and
one swap/evaluation path. Include a paired 200-step real-CIFAR conditioning trace
on identical batches and decisions to detect nonfinite loss, explosive gradient
growth, RNG drift, or unintended state/control-flow differences. Divergent
weights after the first update are expected; data indices, augmentation choices,
CutMix draws, SAM schedule, EMA eligibility, and evaluation cadence must remain
structurally matched.

Proceed to the one metric run only if:

- both parent and candidate complete all paths with finite state and passing
  formula/SAM/EMA audits;
- parent round drift is at most 3%, paired-ratio MAD/median is at most 0.005, and
  the median production-weighted candidate/parent latency ratio is at most 1.01;
- the same-harness projection is at least 25,300 optimizer steps and 155 EMA
  samples, peak allocation is below 1.30 GiB, and projected total runtime is
  below 600 seconds;
- the 200-step candidate trace has no nonfinite values or hard/constituent bound
  violation. Loss magnitude and gradient differences are informational, not a
  tuning signal.

The parent-relative latency gate passes the unchanged parent by construction.
The `25,300` projection is a feasibility floor, not a promise: EXP-012 showed
that paired latency can overpredict realized dose. The metric run therefore
reports realized dose independently and is never retried after a projection
miss.

## Metric Run, Success Criteria, and Falsification

After preflight, launch exactly once with an outer ten-minute limit:

```bash
timeout 600s env CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1
```

Formal success requires `best_test_acc >= 95.71%`, exit zero, `299.5-301.0`
charged seconds, total runtime below 600 seconds, at most one evaluation per
epoch, a complete summary, physical GPU 0, only `train.py` changed, and no
formula, RNG, state, restoration, nonfinite, CUDA, OOM, or timeout failure.

Mechanism support additionally requires at least 25,300 steps, at least 155 EMA
samples with ordinary/SAM imbalance at most one, final-16 EMA mean at least
`95.64%`, and final accuracy at least `95.60%`. A formal threshold pass without
the tail targets is still a tree improvement but weak evidence for a stable
generalization gain. An accuracy miss is no improvement even if every mechanism
audit passes. A realized-dose miss does not override the formal metric verdict,
but prevents a clean full-dose causal conclusion.

The preregistered plausible outcome band is `95.66-95.81%`; clearing `95.71%` is
plausible but not high confidence. A result at or above `95.86%` with a final-16
mean at or above `95.70%` would be a mechanism-sized upside result. Do not rerun,
change epsilon, disable the loss on one path, alter the dose floor, or choose an
evaluation checkpoint after observing the metric.

## Causal Limits and Risks

- This is one fixed-seed candidate against a historical parent. Sub-0.30-point
  deltas are within observed checkpoint/protocol resolution, so best accuracy
  must be interpreted with tail mean/range/final and exact exposure.
- The experiment jointly changes hard loss, CutMix soft loss, the SAM
  perturbation direction, the SAM second-pass update, and the trajectory consumed
  by EMA. It cannot identify which interaction causes the result.
- Poly-1 upweights confident hard examples. That may refine residual boundaries,
  or waste gradient budget on easy examples and worsen calibration.
- Unequal CutMix targets are sharpened and their combined gradient ratio is not
  bounded. The intervention can partially counteract a validated mechanism even
  though it preserves every CutMix image and constituent weight.
- SAM fixes perturbation norm but not direction; EMA may attenuate, amplify, or
  lag the altered trajectory. Tail evaluation observes only the EMA state, not a
  contemporaneous live control.
- Even sub-1% overhead changes step and phase exposure under a wall-clock budget.
  A small delta cannot be attributed solely to loss geometry if realized dose
  differs materially.
- Failure of `epsilon=0.25` does not establish that Poly-1 is ineffective; success
  does not establish that 0.25 is optimal. No coefficient-family conclusion is
  licensed by one preregistered point.

## Recommendation

Advance this as a low-overhead, mechanism-distinct candidate with medium
confidence in experimental cleanliness and low-to-medium confidence in clearing
the threshold. Its strongest advantage is that it preserves the validated model,
data, SAM, and EMA machinery while touching every training gradient at negligible
expected cost. Its largest weakness is not coefficient uncertainty alone, but
the mathematically explicit sharpening overlap with CutMix. That overlap and the
conservative effect size should weigh heavily against candidates with stronger
same-stack evidence, but they do not make the experiment infeasible or redundant.
