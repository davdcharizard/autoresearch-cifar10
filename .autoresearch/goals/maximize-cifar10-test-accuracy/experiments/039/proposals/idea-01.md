# Proposal: Regime-Aligned Hard-Tail Cosine Rephase

## Recommendation

Test the **continuous hard-tail cosine rephase alone** as EXP039. Preserve the
accepted learner exactly through the mixed-target boundary at 65% counted
training time. At that boundary, derive the current accepted learning rate and
use it as the start of a new cosine over the remaining 35%, ending at the
accepted `MIN_LR = 0.002`. Do not reset momentum in the first test.

This is the cleanest member of a three-part regime-transition family:

1. **Tail cosine rephase** (recommended now): changes only future scalar LR.
2. **One-time momentum reset** (defer): changes only inherited optimizer state.
3. **Combined restart** (defer): applies both at the same boundary, but cannot
   reveal which component caused the result.

The ordering matters. The accepted trajectory has a known objective
discontinuity but no observed instability. Rephasing supplies a sustained,
mechanistically legible change over roughly 9,000 hard-label steps. A momentum
reset affects only a short transient: under momentum 0.9, inherited-buffer
memory falls to `0.9^44 = 0.0097` after 44 steps, approximately the observed
EXP036 lag from mixup cutoff to the exhausted-epoch RandAugment cutoff. Combining
the two before either is isolated would spend a scored run on ambiguous causal
evidence.

## Diagnosis and Local Evidence

The accepted EXP036 learner scores 94.48% best / 94.45% final with 0.2456 loss
at 130.304 passes. It nearly interpolates its clean training tail, yet its
optimizer treats the whole run as one stationary objective. At 65%, two strong
regularizers begin leaving the system:

- batch-shared mixup switches exactly from paired soft targets to hard labels;
- worker-side RandAugment switches after the currently active iterator is
  exhausted, typically shortly after 65% (44 steps later in EXP036).

The global cosine and all Nesterov buffers currently continue through this
change without re-alignment. At 65%, the accepted LR is only
`0.06123215295935605`, already 70% below its peak, and continues falling on the
old 5%-to-100% phase. A new cosine beginning from that exact value preserves
continuity while assigning the clean objective its own remaining-time phase.

This is not a rescue of EXP008. EXP008 lowered the endpoint from 0.002 to zero,
reduced useful late update amplitude, and regressed to 93.80% / 0.2629 loss at
normal exposure. The proposed curve retains the empirically protected 0.002
floor and raises only the interior of the hard-label tail. Its purpose is
post-transition adaptation, not endpoint settling.

The local `Time Matters in Regularizing Deep Networks` note supports temporal
removal of early regularization, but does not establish this LR shape. It is
used only to motivate treating the late clean phase as meaningfully distinct.
The experiment remains a falsifiable local optimization test rather than a
literature-derived guarantee.

## Exact Intervention

Keep the accepted schedule unchanged for `progress < MIXUP_END_FRACTION`. For
`progress >= MIXUP_END_FRACTION`, start a second cosine at the value the
accepted global cosine has exactly at the boundary:

```python
def learning_rate(training_time):
    progress = min(max(training_time / TIME_BUDGET_S, 0.0), 1.0)
    if progress < WARMUP_FRACTION:
        warmup_progress = progress / WARMUP_FRACTION
        return MIN_LR + (LR - MIN_LR) * warmup_progress

    cosine_progress = (progress - WARMUP_FRACTION) / (1.0 - WARMUP_FRACTION)
    accepted_lr = MIN_LR + 0.5 * (LR - MIN_LR) * (
        1.0 + math.cos(math.pi * cosine_progress)
    )
    if progress < MIXUP_END_FRACTION:
        return accepted_lr

    transition_cosine_progress = (
        MIXUP_END_FRACTION - WARMUP_FRACTION
    ) / (1.0 - WARMUP_FRACTION)
    transition_lr = MIN_LR + 0.5 * (LR - MIN_LR) * (
        1.0 + math.cos(math.pi * transition_cosine_progress)
    )
    tail_progress = (progress - MIXUP_END_FRACTION) / (
        1.0 - MIXUP_END_FRACTION
    )
    return MIN_LR + 0.5 * (transition_lr - MIN_LR) * (
        1.0 + math.cos(math.pi * tail_progress)
    )
```

An equivalent helper decomposition is acceptable if the numeric function and
source scope are exact. Do not cache transition time from a runtime step, use
epoch indices, restart from `LR`, move the 65% boundary, or add another
schedule scalar. The only anchors are existing constants:

- start: accepted LR evaluated at `MIXUP_END_FRACTION = 0.65`;
- duration: existing remaining fraction `1 - MIXUP_END_FRACTION = 0.35`;
- endpoint: accepted `MIN_LR = 0.002`.

The candidate is value-continuous at 65% and exactly matches the accepted LR at
0%, 5%, and every point before 65%. It also matches the accepted endpoint at
100%. It intentionally changes the right derivative at 65%; a cosine restart
begins with zero slope while the accepted curve continues descending.

Precomputed reference values are:

| Progress | Accepted LR | Rephased LR | Delta |
|---:|---:|---:|---:|
| 65.0% | 0.06123215 | 0.06123215 | 0 |
| 70.0% | 0.04685213 | 0.05829924 | +0.01144711 |
| 75.0% | 0.03394912 | 0.05008140 | +0.01613227 |
| 82.5% | 0.01812052 | 0.03161608 | +0.01349556 |
| 90.0% | 0.00736409 | 0.01315075 | +0.00578666 |
| 95.0% | 0.00335023 | 0.00493291 | +0.00158268 |
| 100.0% | 0.00200000 | 0.00200000 | 0 |

The integrated LR area over 65%-100% rises from approximately 0.00793445 to
0.01106563, a 39.46% increase; the tail-average LR rises from 0.02266987 to
0.03161608. This is a material optimization intervention despite zero added
model work.

Everything else remains accepted: `(2,2,3)` WRN at widths `[32,64,128]`, the
bias-free `128 -> 64 -> 128` scale-0.1 pooled residual MLP and isolated seed,
FP32, batch 256, SGD with momentum 0.9 and Nesterov, continuous matrix-only
`5e-4` decay including `fc.weight`, alpha-0.2 batch-shared mixup, early N1/M5
RandAugment, crop/flip, seed 42, loader, evaluator cadence, and limits.

## Deferred Family Members

### One-Time Momentum Reset

The isolated state intervention would leave `learning_rate` byte-identical and,
inside the existing one-way `mixup_enabled and not use_mixup` branch, clear
every live SGD `momentum_buffer` before the first hard-label forward/backward:

```python
for state in optimizer.state.values():
    buffer = state.get("momentum_buffer")
    if buffer is not None:
        buffer.zero_()
```

It must clear exactly one buffer for every trainable parameter with optimizer
state, preserve parameter bytes and RNG, and execute once before the first
hard-label `optimizer.step()`. Reconstructing the optimizer is not equivalent:
it risks changing groups/defaults and makes membership harder to audit. Setting
momentum to zero or tapering it is also a different intervention.

Its rationale is to discard velocity estimated under mixed images and paired
soft targets. Its main weakness is duration: the old-buffer contribution
decays below 1% in about 44 updates even without a reset, while the hard tail
contains roughly 9,000 steps. A miss would close the isolated one-time reset,
not other state schedules.

### Combined Regime-Aligned Restart

The combination would apply the exact rephased LR and exact one-time buffer
clear at the same first hard-label boundary. It is coherent as a two-phase
optimizer restart and has no extra scalar, but it should follow only if an
isolated member shows positive evidence or a later diagnosis demonstrates both
sustained under-adaptation and harmful inherited velocity. As EXP039's first
test, it would make success and failure difficult to assign and would not
justify tuning either component.

## Semantic Preflight

Use an evaluator-free disposable harness and an independent
`git show a7c42dc:train.py` oracle. Before timing or scoring, require:

- the production diff is confined to `learning_rate`; `prepare.py` is
  byte-identical and no test data/evaluator is constructed in preflight;
- accepted/candidate model topology, every initialized tensor byte, parameter
  count 1,003,482, optimizer groups/options/state, data/transforms, constants,
  and post-construction CPU/CUDA RNG states are exact;
- candidate LR is finite and within `[0.002, 0.2]` on a dense grid, matches the
  accepted function exactly from 0 through just below 65%, is continuous at
  65% within floating-point tolerance, and equals 0.002 at 100%;
- direct independent formulas reproduce both schedules at 0%, 5%,
  `65%-epsilon`, 65%, `65%+epsilon`, 70%, 75%, 82.5%, 90%, 95%, and 100%,
  including the reference values above;
- candidate LR is non-increasing over the full run and strictly above accepted
  for sampled interior points in `(65%, 100%)`;
- cloned early-mixup steps before 65% are bitwise identical end to end,
  including LR, input, lambda/permutation, logits, loss, gradients, parameter
  update, momentum buffers, and CPU/CUDA RNG;
- at 65% and representative hard-tail points, candidate/accepted arms start
  from exact cloned model, optimizer, fixture, and RNG state; gradients before
  `optimizer.step()` are bitwise identical, while each update and momentum
  buffer matches an independent PyTorch SGD/Nesterov oracle at its prescribed
  LR;
- no randomness is consumed by the schedule and restoring candidate state/RNG
  reproduces candidate updates exactly;
- cutoff probes prove the first `progress >= 0.65` step simultaneously uses
  hard labels and the derived transition LR, while RandAugment retains its
  existing exhausted-iterator semantics rather than being forced off early;
- source/static audit confirms one-way transitions, continuous weight decay,
  one evaluation at most per epoch, and summary behavior are untouched.

Print all measured LR values, maximum pre-boundary difference, boundary jumps,
monotonicity margin, state equalities, and oracle deltas before assertions. A
failed semantic gate closes only the exact implementation and must not be
repaired by adding a free phase length, peak, floor, or boundary.

## Throughput and Exposure Gate

The candidate adds only host scalar arithmetic per step, so exposure should be
unchanged, but the accepted head operates near a protected 127-pass floor.
Compare complete production-equivalent accepted and candidate early-mixup and
hard-label steps on the idle H20. Include H2D, LR calculation/write, zeroing,
mixup when active, forward, loss, finite guard, backward, Nesterov update, and
final synchronization. Use at least 20 warmups and four counterbalanced windows
of at least 50 steps per arm with fresh deterministic fixtures per replicate.

Print all raw windows before assertions. Require finite measurements,
population CV no greater than 5% for every arm/regime, and compute:

```text
retention =
    (0.65 / candidate_early_ms + 0.35 / candidate_hard_ms) /
    (0.65 / accepted_early_ms + 0.35 / accepted_hard_ms)
projected_passes = 130.304 * retention
```

Proceed only if retention is at least `127 / 130.304 = 0.97464391` and
projected passes are at least 127. A stable miss is final; do not rerun timing,
relax the threshold, or simplify schedule arithmetic. No real-loader benchmark
is required because transforms, workers, shapes, and consumer work are source
identical.

## Sole Scored Run and Decision Contract

After the gates pass, reconfirm baseline 94.48% at `a7c42dc`, one idle NVIDIA
H20, local CIFAR-10, frozen `prepare.py`, no stale log, and the exact production
diff. Run exactly once:

```bash
timeout 600s uv run train.py > run.log 2>&1
```

Require exit 0, one finite complete summary, reported 300.0 counted seconds,
total below 600 seconds, 1,003,482 parameters, one mixup transition at the first
pre-step progress at or above 65%, the later exhausted-epoch RandAugment
transition, unique every-fifth-epoch evaluations plus final partial epoch, and
no traceback, OOM, worker, or non-finite error. Record realized passes as
`num_steps * 256 / 50000` and audit the transition LR against 0.06123.

Primary success requires `best_test_acc >= 94.58%`, exactly 0.10 points above
the accepted 94.48%. Pre-register `final_test_acc >= 94.45%` and
`final_test_loss <= 0.2456` as non-decisive corroboration that increased tail
motion did not create a fragile peak or degrade confidence. Neither can rescue
a primary miss; a primary success remains valid if they fail, but its
mechanistic interpretation must be called fragile.

A valid score below 127 realized passes counts and cannot be rerun, but it is
operationally inconclusive for the intended normal-exposure schedule mechanism.
Timeout, malformed output, semantic contamination, wrong transition, or
multiple evaluations in an epoch is a failure, not a weak score.

## Falsifiable Interpretation and Family Closure

**Success interpretation:** If the candidate retains at least 127 passes and
reaches 94.58%, the evidence supports the narrow claim that the accepted clean
tail was under-updated after the mixed-to-hard objective change and that a
continuous boundary-derived rephase improves this fixed-seed trajectory.
Endpoint/loss corroboration would strengthen the claim. Success would not prove
that cosine is uniquely optimal, that momentum inheritance is harmful, or that
65% should be tuned.

**Normal-exposure miss:** If at least 127 passes score below 94.58%, retain the
accepted global cosine and close immediate hard-tail **LR rephase** variants:
do not tune the tail peak, floor, duration, cosine exponent, restart boundary,
or add warmup/cycles as a rescue. A miss specifically indicates that increasing
interior hard-tail LR by this parameter-free phase realignment did not improve
the accepted learner. It does not by itself falsify the isolated momentum reset,
which targets optimizer-state inheritance rather than update amplitude.

**Combined-family rule:** Do not run the combined restart after an isolated
rephase miss merely as a rescue. It becomes justified only by independent
evidence for harmful inherited velocity, such as an isolated momentum-reset
success. Conversely, a reset miss does not license combining two misses.

**Low-exposure or pre-score failure:** Below 127 realized passes, close only the
exact candidate; mechanism attribution is inconclusive. A semantic or timing
failure supplies no accuracy evidence and closes only that implementation.

## Risks

- The accepted clean tail already nearly interpolates and finishes within 0.03
  points of its best. A 39.46% larger tail LR area may increase stochastic
  wandering, hurt margin settling, or worsen test loss.
- The objective transition is staggered: mixup stops at the exact time boundary,
  while RandAugment remains active until the iterator exhausts. The rephase is
  aligned to hard targets, not to a perfectly clean image distribution.
- Value continuity does not imply derivative continuity. The zero-slope cosine
  restart briefly flattens LR decay at the boundary; that pause is intrinsic to
  the treatment.
- Since weight decay is coupled SGD decay, larger tail LR also increases the
  effective late shrinkage integral. Static decay remains protected, but the
  experiment cannot claim to change only data-gradient amplitude.
- The schedule is time-based. Tiny runtime differences can shift transition
  steps even with unchanged throughput; exposure and ordered-transition audits
  remain binding.
- One fixed seed cannot establish average treatment effect, and rerolling or an
  adjacent schedule sweep is prohibited.

## Falsifiable Hypothesis

If the accepted global cosine decays too far before the 65% mixed-to-hard
objective transition, then preserving its exact value at that boundary but
rephasing a cosine over the remaining 35% to the protected 0.002 floor will
retain at least 127 projected and realized passes and raise fixed-seed
`best_test_acc` from 94.48% to at least 94.58%, with final accuracy at least
94.45% and final loss no worse than 0.2456 as corroboration.

A valid normal-exposure score below 94.58% falsifies this parameter-free
hard-tail LR rephase as a useful standalone refinement and closes adjacent
rephase shapes and amplitudes. It leaves only the separately attributable
one-time momentum reset open within this regime-transition family.

## Local Evidence

- `experiments/036/04-analysis.md`: accepted pooled-head frontier is 94.48%
  best, 94.45% final, 0.2456 loss, and 130.304 passes.
- `experiments/008/04-analysis.md`: cosine-to-zero scored 93.80% and worsened
  loss, protecting the nonzero 0.002 endpoint and continued late motion.
- `03-experiment-learnings.md`: static classifier decay, mixup strength and
  duration, averaging, SAM, and several capacity/exposure variants are closed;
  a low-overhead orthogonal mechanism is required.
- `02-system-understanding.md`: compute is binding, the learner nearly
  interpolates its tail, and generalization/boundary quality remain limiting.
- `knowledge/papers/time-matters-regularization.md`: early regularization may
  be removed after its critical period, motivating explicit temporal phases
  without specifying this candidate schedule.
