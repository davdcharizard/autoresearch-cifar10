# Proposal: Weaker Alpha-0.1 Batch-Shared Mixup

## Recommendation

Starting exactly from accepted commit `67c8e98`, change only
`MIXUP_ALPHA = 0.2` to `MIXUP_ALPHA = 0.1`. Preserve the batch-shared scalar
coefficient, strict 65% counted-time cutoff, `(2,2,3)` WRN, early worker-safe
`RandAugment(num_ops=1, magnitude=5)`, batch 256, FP32 SGD/Nesterov, time-based
LR curve, seed 42, data pipeline, and frozen evaluator.

This is a low-confidence closure experiment rather than a monotonic inference
from the alpha-0.4 failure. Its value is clean attribution: the stronger side
of batch-shared mixup strength has failed, while the weaker side remains the
only unmeasured one-constant bracket around accepted alpha 0.2. A valid
normal-exposure miss must end immediate mixup-strength tuning.

## Evidence and Rationale

The current 94.32% baseline is EXP-027's composition of an extra final stage-3
block and worker-RNG-isolated early RandAugment. It retained alpha-0.2
batch-shared mixup through 65% of counted time and completed 133.00736 data
passes, with 94.22% final accuracy and 0.2523 final loss. The system diagnosis
attributes the remaining error to generalization and boundary quality rather
than memory or input throughput, and identifies near-zero-compute
regularization changes as one of the few ways to improve without surrendering
the narrow 133-pass operating regime.

Alpha 0.1 weakens the already-present interpolation instead of stacking a new
regularizer or masking model features. There is a specific, though weak,
interaction rationale: the alpha-0.4 score was obtained on the shallower
`(2,2,2)` model without RandAugment, while the accepted deeper learner already
receives a second early invariance treatment. Its optimal interpolation
severity could therefore be lower. EXP-030's 93.91% targeted drop-path miss
reinforces that the added block should remain fully active, and EXP-032's
unscored reflection-padding feasibility miss leaves this GPU-local one-line
treatment as the cleanest remaining closure test.

Contrary evidence is substantial and must control interpretation:

- EXP-002 established alpha 0.2 through 65% as a +0.69-point treatment on the
  original WRN recipe.
- EXP-005 changed only alpha to 0.4 and scored 93.57% at normal exposure, with
  worse final loss, but that does not imply that every weaker alpha is better.
- EXP-004 and EXP-020 scored 93.91% and 93.82% at 50% and 75% cutoffs,
  bracketing the accepted duration and showing a non-monotonic
  regularization/refinement response.
- EXP-015 scored 93.79% with independent per-example alpha-0.2 coefficients,
  so batch-level coefficient coherence is protected.
- EXP-027 improved accuracy and loss; it provides no direct symptom that the
  accepted deeper-plus-RandAugment learner is over-regularized.

The local mixup evidence supports convex input/target interpolation as a
low-cost CIFAR generalizer, but supplies no direct evidence that alpha 0.1 is
better. Expected upside is therefore small; the experiment is justified by
closure quality and exceptionally low implementation ambiguity.

## Exact Intervention

The complete production diff must be one constant line in `train.py`:

```python
MIXUP_ALPHA = 0.1  # accepted: 0.2
```

No helper, branch, logging, cutoff, loss, pairing, seed, or evaluator change is
allowed. Preserve the accepted operation exactly:

```python
mix = distribution.sample()
permutation = torch.randperm(inputs.size(0), device=inputs.device)
mixed_inputs = mix * inputs + (1.0 - mix) * inputs[permutation]
loss = mix * CE(outputs, targets) + (1.0 - mix) * CE(outputs, targets[permutation])
```

The coefficient remains one CUDA FP32 scalar shared by all examples in a
batch. Do not clamp or symmetrize it, sample per example, use a private
generator, cache accepted draws, force deranged pairs, or alter target
weighting. Mixup remains active only while pre-step progress is `<0.65`; the
last 35% follows the exact accepted hard-label path. RandAugment retains its
separate exhausted-iterator transition, so the current active iterator may
finish shortly after the exact mixup boundary.

Everything else remains accepted: 987,098 trainable parameters, stage depths
`(2,2,3)`, widths `[32,64,128]`, Kaiming initialization, batch 256, LR
`0.2 -> 0.002` with 5% warmup, momentum 0.9, Nesterov, matrix-only `5e-4`
weight decay, crop/flip plus early N1/M5 RandAugment, eight persistent workers,
seed 42, 300 counted seconds, and no more than one evaluation per epoch.

## Intended Distribution Change

For symmetric `Beta(a,a)`, both concentrations have mean 0.5. Reducing alpha
changes concentration toward the endpoints rather than biasing the mixture
toward the original example. The theoretical variance rises from
`1 / (4 * (2 * 0.2 + 1)) = 0.178571` to
`1 / (4 * (2 * 0.1 + 1)) = 0.208333`.

| Event | Beta(0.2, 0.2) | Beta(0.1, 0.1) |
|---|---:|---:|
| `0.2 <= lambda <= 0.8` | 21.46% | 12.06% |
| `lambda <= 0.1 or lambda >= 0.9` | 67.34% | 81.28% |
| `lambda <= 0.01 or lambda >= 0.99` | 41.96% | 64.06% |

A coefficient near zero is still a near-clean paired example because the image
and target use the same orientation. Alpha 0.1 therefore creates many more
collectively near-endpoint batches while preserving the expected coefficient,
ordinary pairing, and useful batch-level coherence.

## RNG Semantics

Keep `torch.manual_seed(42)` and `torch.cuda.manual_seed(42)` unchanged and
construct the symmetric CUDA concentration tensors in the accepted location.
Model initialization, loader construction, CPU state, and pre-training CUDA
state must match the accepted oracle because changing the concentration value
does not itself draw randomness.

From the first mixed step onward, accepted/candidate CUDA trajectory identity
is not a semantic requirement. Beta sampling uses concentration-dependent
gamma sampling; EXP-005 already established that changing alpha changes the
fixed-seed coefficient process and later device permutations. This is an
intrinsic part of the alpha treatment, not a seed reroll. The alpha-0.1 run is
the sole deterministic seed-42 realization, and neither a second seed nor a
trajectory-realignment rescue is allowed.

Worker-side crop/flip/RandAugment and sampler behavior remain protected by
source identity. Main-process CUDA Beta and `randperm` draws cannot advance
worker-private CPU streams, and `MIXUP_ALPHA` is not read in worker code.

## Semantic and Distribution Preflight

Use an evaluator-free harness with an independent
`git show 67c8e98:train.py` oracle. The harness must not construct test data or
call evaluation. Before scoring, require:

- the production diff against `67c8e98` is exactly the alpha constant change,
  and `prepare.py` is byte-identical;
- accepted and candidate topology, initial model state, 987,098 parameter
  count, optimizer groups, schedule function, loader constants, transforms,
  and post-construction CPU/CUDA RNG states match exactly;
- concentration tensors are symmetric scalar CUDA FP32 values, exactly 0.2
  for the oracle and 0.1 for the candidate;
- at least 100,000 isolated fixed-seed candidate draws are finite and in
  `[0,1]`, with mean in `[0.495,0.505]`, variance in `[0.203,0.214]`, central
  `[0.2,0.8]` mass in `[0.115,0.127]`, and endpoint `<=0.1 or >=0.9` mass in
  `[0.806,0.820]`;
- separately seed-reset accepted draws confirm that alpha 0.1 has greater
  variance and endpoint mass but lower central mass than alpha 0.2;
- restoring the same pre-call candidate RNG state reproduces lambda,
  permutation, mixed inputs, paired targets, loss, gradients, and one optimizer
  update exactly; accepted/candidate post-call CUDA states may differ;
- a batch-structure probe proves one scalar applies consistently to every image
  and both target losses, including a direct pixel/target interpolation oracle;
- under cloned hard-path state and RNG, candidate and accepted hard-label loss,
  gradients, optimizer update, and resulting RNG state are bitwise identical;
- progress values just below, at, and above 0.65 prove strict one-way mixup
  transition behavior while LR values remain accepted;
- exact source comparison proves `EarlyRandAugment`, transforms, DataLoader,
  shared-byte control, exhausted-iterator transition, evaluation cadence, and
  output summary are untouched.

Print all measured distribution statistics before enforcing assertions. Run
the distribution probe in a disposable process or restore every RNG state so
it cannot contaminate the scored process. The alpha-defined divergence after
the first mixed draw is evidence of treatment, not grounds to fail preflight.
A new real-loader timing test is unnecessary because CPU transforms, batch
shape, workers, and consumer structure are source-identical.

## Throughput Feasibility Gate

Tensor shapes and the training graph are unchanged, but concentration-dependent
Beta sampling can alter a small amount of GPU work and the accepted run has
only about three passes above the protected 130-pass regime. Measure this once
before scoring.

On the idle H20, compare independent accepted and candidate modules from equal
model/optimizer snapshots. Measure early mixup and hard-label regimes
separately using fixed pinned batches and production-equivalent nonblocking
H2D, LR write, zero-grad, Beta draw/permutation/interpolation when active,
forward, paired loss, finite guard, backward, SGD/Nesterov step, and final CUDA
synchronization. Use at least 20 warmups and three balanced windows of at least
50 steps per arm, alternating arm order and using fresh deterministic fixtures
per replicate. Print all raw windows and derived values before assertions.

Require finite measurements and population CV no greater than 5% for every
arm. Compute fixed-time retention from the regime medians:

```text
retention =
    (0.65 / candidate_early_ms + 0.35 / candidate_hard_ms) /
    (0.65 / accepted_early_ms + 0.35 / accepted_hard_ms)
projected_passes = 133.00736 * retention
```

Proceed only if retention is at least `0.9774` and projected passes are at
least 130.0. The accepted/candidate hard-path medians should agree within
measurement noise; a material difference is a harness failure. Do not rerun a
stable gate miss, lower the bound, or change alpha to rescue feasibility.

## Sole Scored Run and Decision Rule

After all gates pass, reconfirm baseline 94.32% at `67c8e98`, one idle NVIDIA
H20, local CIFAR-10 availability, no stale log, and the exact production diff.
Run exactly once:

```bash
timeout 600s uv run train.py > run.log 2>&1
```

Require exit 0, one complete finite summary, 300.0-300.1 counted seconds, total
below 600 seconds, 987,098 parameters, exactly one mixup transition at the first
step whose progress is at least 65%, one later exhausted-epoch RandAugment
transition, unique every-fifth-epoch evaluations plus the final partial epoch,
and no traceback, OOM, worker, or non-finite error. Record realized exposure as
`num_steps * 256 / 50000`.

A valid completed score below 130 passes still counts and may not be rerun, but
the proposed strength mechanism is operationally inconclusive because it left
the protected exposure regime. Timeout, malformed output, semantic
contamination, or a wrong transition is a crash rather than a weak result.

The objective succeeds only if `best_test_acc >= 94.42%`, exactly 0.10 points
above the 94.32% accepted baseline. Pre-register two non-decisive corroboration
signals:

- `final_test_acc >= 94.32%` supports a stable endpoint rather than an isolated
  sparse-evaluation maximum;
- `final_test_loss <= 0.2523` supports preserved or improved boundary quality.

Neither signal can override the primary metric. A valid 94.42% best score is an
improvement even if corroboration fails, but the mechanism claim must be
reported as fragile. Better endpoint accuracy or loss cannot rescue a best
score below 94.42%.

## One-Run Family Closure

If a valid run with at least 130 realized passes scores below 94.42%, retain
alpha 0.2 and close the immediate batch-shared mixup-strength family. Do not
try alpha 0.05, 0.15, 0.25, 0.3, another seed, coefficient symmetrization,
private-RNG realignment, per-example sampling, or a different cutoff. Alpha
0.4 has closed the stronger side, alpha 0.1 closes the weaker side, and the
50%/75% results already protect the 65% duration.

If alpha 0.1 succeeds, it may replace alpha 0.2 on the frontier based only on
the primary objective and hard constraints. Success still does not license an
adjacent-alpha sweep without a new independent mechanism. This experiment
tests one fixed-seed treatment, not the continuous optimum of alpha.

A pre-score semantic or stable feasibility failure closes only this exact
alpha-0.1 implementation under its tested protocol; it supplies no accuracy
evidence. A valid run below 130 passes closes the exact treatment but does not
support a strength-family conclusion because the operating regime changed.

## Risks

- **Likely under-regularization:** 81.28% of alpha-0.1 coefficients lie outside
  `[0.1,0.9]`, so many early batches are nearly clean/swapped and may lose the
  interpolation bias that made EXP-002 successful.
- **Negative local neighborhood:** every scored mixup strength, duration, or
  coefficient-correlation perturbation has regressed. Expected gain is lower
  than for a genuinely new mechanism.
- **Alpha-defined RNG divergence:** the CUDA coefficient/permutation sequence
  diverges by design. One seed cannot separate average treatment effect from
  its deterministic realization, and the no-reroll rule is binding.
- **Interaction uncertainty:** deeper capacity plus RandAugment could need less
  target softness, or could need exactly alpha 0.2 to regularize that capacity;
  existing evidence does not identify the sign.
- **Small decision margin:** the required improvement is ten additional correct
  test examples. Endpoint and loss corroboration constrain interpretation but
  cannot establish multi-seed robustness.
- **Throughput headroom:** accepted exposure is only 133.00736 passes; an
  unexpected Beta-sampling slowdown can disqualify the intended normal-exposure
  mechanism even though shapes are unchanged.

## Falsifiable Hypothesis

If alpha-0.2 interpolation is slightly too strong for the accepted
deeper-plus-early-RandAugment learner, then changing only the batch-shared
symmetric Beta concentration to 0.1 will retain at least 130 projected and
realized passes, raise fixed-seed `best_test_acc` from 94.32% to at least
94.42%, retain `final_test_acc >= 94.32%`, and avoid worsening final test loss
above 0.2523.

A valid normal-exposure score below 94.42% falsifies alpha 0.1 as a useful
standalone refinement and closes adjacent mixup-strength tuning. Endpoint and
loss remain mechanism corroboration only; the goal verdict is determined by
the primary metric and hard constraints.

## Local Evidence

- `experiments/002/04-analysis.md`: early alpha-0.2 mixup through 65% improved
  the original WRN from 93.38% to 94.07%.
- `experiments/004/04-analysis.md` and `experiments/020/04-analysis.md`: 50%
  and 75% cutoffs both regressed, protecting the 65% duration.
- `experiments/005/04-analysis.md`: alpha 0.4 scored 93.57% at normal exposure
  and documented concentration-dependent CUDA trajectory divergence.
- `experiments/015/04-analysis.md`: per-example alpha-0.2 coefficients scored
  93.79% at normal exposure, protecting batch-shared coherence.
- `experiments/027/04-analysis.md`: the accepted composition scored 94.32%
  best, 94.22% final, 0.2523 loss, and 133.00736 passes.
- `experiments/030/04-analysis.md`: isolated early p=0.05 drop-path scored
  93.91% at 132.72064 passes, protecting full residual participation.
- `experiments/032/04-analysis.md`: reflection padding failed a worker-timing
  stability gate without scoring, leaving the accepted baseline unchanged.
- `02-system-understanding.md`: compute is binding, memory and I/O are not, and
  generalization/boundary quality limits the accepted learner.
- `knowledge/papers/mixup.md`: convex image/target interpolation is a validated
  low-overhead CIFAR generalizer; local results provide the treatment-specific
  constraints.
