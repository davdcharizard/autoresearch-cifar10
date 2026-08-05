# Proposal: Per-Example Mixup Strengths

## Recommendation

Keep the accepted EXP-002 recipe unchanged except for the granularity of the
mixup coefficient. During the existing first 65% of counted training time,
sample one independent `Beta(0.2, 0.2)` coefficient for each of the 256
examples, broadcast the resulting vector over image dimensions, and weight
each example's two unreduced cross-entropies with its own coefficient. Retain
the existing single random permutation, hard-label tail, architecture,
optimizer, schedule, augmentation, seed, loader, and evaluation cadence.

Do not add deranged pairing to this experiment. A uniform permutation has one
fixed point in expectation, so only about 0.39% of batch entries become an
ordinary unmixed example because of self-pairing. Eliminating those entries is
too small an intervention to justify changing pairing semantics alongside the
coefficient-correlation treatment. It would also obscure whether any result
came from within-batch coefficient diversity or forced non-self pairs.

## Diagnosis And Rationale

The accepted run reaches 94.07% after 141.9 dataset-equivalent passes. Its key
regularizer is alpha-0.2 mixup through 65% counted time followed by a 35%
hard-label refinement phase. Adjacent attempts indicate that the *amount* of
regularization is already well calibrated: alpha 0.4 fell to 93.57%, ending
mixup at 50% fell to 93.91%, shared-rectangle CutMix fell to 93.72%, and adding
residual dropout fell to 93.52%. The opportunity is therefore not a larger
average soft-target dose, but a less correlated way to realize the already
validated objective.

The accepted code draws one scalar coefficient per batch. Every example in an
update is consequently almost hard or materially mixed together. For
`Beta(0.2, 0.2)`, the coefficient has mean 0.5 and variance about 0.1786, so a
batch-shared draw imposes substantial update-to-update variation in effective
regularization strength. Drawing 256 independent coefficients preserves that
same marginal law for every training example and preserves the expected mixup
objective, while reducing the standard deviation of the batch-average
coefficient by a factor of 16 under independence. Each update then contains a
spectrum of nearly clean and strongly interpolated examples instead of tying
all examples to one strength.

This is a gradient-noise and constraint-diversity treatment, not stronger
mixup. It may stabilize the direction of each large-batch update and expose
more varied interpolation constraints before the unchanged hard-label tail.
The mechanism is particularly plausible at batch 256, where there are enough
examples for the empirical coefficient distribution to be representative on
each step.

## Exact Mechanism

Change `mixup_batch` so `mix` has shape `[batch]` and images use a broadcast
view with shape `[batch, 1, 1, 1]`:

```python
def mixup_batch(inputs, targets, distribution):
    mix = distribution.sample((inputs.size(0),))
    permutation = torch.randperm(inputs.size(0), device=inputs.device)
    image_mix = mix[:, None, None, None]
    mixed_inputs = image_mix * inputs + (1.0 - image_mix) * inputs[permutation]
    return mixed_inputs, targets, targets[permutation], mix
```

In the mixup loss path, request unreduced cross-entropies and take one mean
after per-example weighting:

```python
loss_a = F.cross_entropy(outputs, targets_a, reduction="none")
loss_b = F.cross_entropy(outputs, targets_b, reduction="none")
loss = (mix * loss_a + (1.0 - mix) * loss_b).mean()
```

Do not clamp, symmetrize with `max(mix, 1 - mix)`, sort, normalize, or reuse
coefficients. Those operations change the distribution or its relation to the
pair orientation. Do not change `MIXUP_ALPHA = 0.2`, the 65% cutoff, or the
single `torch.randperm` pairing rule. The hard-label branch must remain
byte-for-byte behaviorally equivalent to accepted training.

The existing device-resident scalar concentration tensors are suitable:
`distribution.sample((BATCH_SIZE,))` returns a CUDA FP32 vector. The final
partial-batch case is irrelevant because `drop_last=True`, but using
`inputs.size(0)` keeps the helper correct independently of that loader setting.

## RNG Semantics

Retain seed 42 and the global RNG policy. Sampling 256 beta variates instead of
one necessarily advances the CUDA generator differently, so later mixup
coefficients and permutations will not be bit-identical to the accepted run.
That RNG consumption is intrinsic to the treatment and is not seed rerolling.
Do not introduce a private generator, cache draws, or attempt to realign the
global stream: `torch.distributions.Beta.sample` does not expose a generator,
and changing the random-number plumbing would add another treatment.

The DataLoader workers have their own seeded processes, so their crop and flip
streams are not driven by the main CUDA generator. Model initialization occurs
before training mixup draws and remains identical. A preflight must verify
identical initialized parameters under a reset seed and exact hard-label-step
equivalence; full mixed-step parameter equality is neither expected nor a
valid gate once the coefficient vector differs.

## Predicted Metric Impact

The prior is a small positive movement because the expected objective and
regularization strength are unchanged. Predict `best_test_acc` in the
94.15-94.30% range, centered near 94.20%, with at least 95% of accepted
throughput. The formal success threshold remains 94.17%, exactly the 94.07%
baseline plus the required 0.10 percentage-point margin.

The effect could be smaller than a CIFAR-10 seed's ordinary discretization
because the test metric changes in 0.01-point increments and the accepted
recipe is already mature. Perform one fixed-seed scored run only; do not rerun
a near miss or tune alpha based on the result.

## Implementation Scope

Modify only `train.py`:

1. Change the beta sample from scalar to length `inputs.size(0)`.
2. Broadcast the vector only for image interpolation.
3. Compute both paired cross-entropies with `reduction="none"`, weight them by
   the original one-dimensional coefficient vector, and average once.
4. Optionally add one concise startup log identifying per-example mixup, but do
   not add per-step logging or synchronization.

Preserve all of the following:

- WRN-16-2 with 691,674 parameters and the existing Kaiming initialization.
- FP32 SGD, momentum 0.9, Nesterov, matrix-only `5e-4` weight decay.
- LR 0.2, 5% warmup, cosine schedule, 0.002 floor, and time-based progress.
- Batch size 256, crop/flip transform, mean subtraction, persistent workers,
  fixed seed 42, `MAX_STEPS`, and one H20.
- Mixup alpha 0.2 through 65% counted time and the unchanged hard-label tail.
- Existing finite-loss guard, sparse evaluation schedule, and at most one
  evaluation per epoch.

## Unscored Preflight Discriminators

Run a local evaluator-free preflight in a fresh process before the scored run.
Patch `prepare.Eval` to a fail-closed dummy before importing `train.py`, use
synthetic or already-loaded training-shaped inputs only, and never inspect test
accuracy. The preflight must establish both semantics and cost.

### Statistical semantics

With fixed synthetic inputs, labels, and a reset CUDA seed:

- Assert the coefficient has shape `[256]`, dtype FP32, is finite, remains in
  `[0, 1]`, and has more than one distinct value.
- Across at least 4,096 sampled coefficients, require empirical mean in
  `[0.47, 0.53]`, variance in `[0.15, 0.21]`, and nonzero median within-batch
  standard deviation. These broad gates catch scalar broadcasting and wrong
  concentration without treating sampling noise as a result.
- Verify each mixed image exactly follows
  `lambda_i * x_i + (1-lambda_i) * x_perm_i` within FP32 tolerance and that the
  same `lambda_i` weights the corresponding two target losses.
- Verify a manually constant vector `lambda_i = c` produces the same scalar
  loss and parameter gradients, within `rtol=1e-5, atol=1e-6`, as the accepted
  scalar formula using the same model, inputs, permutation, and `c`. This proves
  the unreduced-loss rewrite changes no mathematics when coefficients agree.
- Assert the candidate's randomly sampled vector is not equivalent to a single
  broadcast coefficient by checking multiple mixed examples against their
  recovered interpolation ratios.
- Confirm a hard-label forward/backward/SGD step from cloned initialized model
  and optimizer states is exactly aligned with the accepted path under matched
  inputs, establishing that no late-tail behavior changed.

### Throughput

Benchmark accepted scalar and candidate vector mixup production steps on the
single H20 in balanced order. Use cloned model and optimizer states, pinned host
inputs, real nonblocking transfers, LR writes, beta sampling, permutation,
mixing, forward, finite guard, backward, optimizer step, and final CUDA
synchronization. Warm each path for at least 25 steps, then collect three
50-step windows per path. Require finite `[256, 10]` logits and finite loss,
population CV no greater than 5% for each path, and no OOM.

Use median window time. Define retention as
`accepted_scalar_ms / candidate_vector_ms` and projected passes as
`141.9 * retention`. Launch the scored run only if retention is at least 95%
and projected passes are at least 134.8. The vector beta draw and unreduced CE
allocate only small batch-length tensors, so a larger loss would indicate an
implementation problem or an unexpectedly costly distribution kernel. The
hard-label portion is identical; using the all-mixup ratio for the projection
is conservative because only 65% of counted time pays the new overhead.

Do not use preflight loss magnitude or any evaluator output to select between
variants. There is exactly one candidate: independent per-example alpha-0.2
coefficients with ordinary random permutation.

## Why This Is Materially Different From Prior Failures

- **Versus alpha 0.4 (EXP-005):** alpha 0.4 changes the marginal coefficient
  law toward central, strongly mixed examples and therefore raises average
  target/input ambiguity. This proposal retains alpha 0.2 exactly; only
  correlations among examples in the same update change.
- **Versus CutMix (EXP-003):** CutMix substitutes spatial rectangles and
  area-derived labels, creating local image discontinuities and a different
  label-image assumption. This proposal retains global convex pixel
  interpolation and the validated mixup target rule.
- **Versus the 50% cutoff (EXP-004):** the temporal exposure to mixup remains
  exactly 65%. The final 35% remains hard-label refinement.
- **Versus residual dropout (EXP-006):** no feature path is masked and no second
  regularizer is stacked. Expected per-example loss remains the accepted mixup
  objective.

## Failure Modes And Interpretation

- **Useful scalar-step noise is removed:** Batchwise coefficient variation may
  act like beneficial optimizer noise. A lower score with normal exposure would
  reject the hypothesis that reducing this correlation improves generalization.
- **Mixed examples interact through BatchNorm:** Even though the expected
  per-example objective is unchanged, a within-batch coefficient spectrum
  changes batch activation statistics relative to uniformly strong/weak
  batches. This is part of the proposed mechanism, not an implementation bug.
- **Coefficient orientation imbalance:** Independent coefficients do not force
  symmetric pair contributions. This matches accepted mixup and is unbiased in
  expectation. Symmetrization would be a separate experiment.
- **RNG trajectory changes:** The candidate consumes more CUDA random numbers.
  One fixed-seed result tests the complete deterministic treatment. Do not
  attribute a near miss solely to RNG or reroll it.
- **Kernel overhead reduces exposure:** If preflight misses the 95% gate, reject
  the implementation before scoring. If a scored run unexpectedly realizes
  fewer than 134.8 projected-pass equivalents, accuracy remains authoritative,
  but a miss is operationally confounded and should not motivate a rerun.
- **Effect is too small:** Preserving the expected objective may produce no
  meaningful decision-boundary shift. A valid score below 94.17% closes this
  exact standalone treatment.

## Full-Run Verification

After a passing preflight, remove stale `run.log` and execute exactly once:

```bash
timeout 600s uv run train.py > run.log 2>&1
```

Require exit code 0, one NVIDIA H20, a complete final summary, about 300 counted
training seconds, total time below 600 seconds, and no more than one evaluation
per epoch. Confirm mixup disables exactly once near 195 counted seconds, the
remaining steps use hard labels, the parameter count remains 691,674, and all
losses are finite. Record `best_test_acc`, final accuracy/loss, steps, epochs,
peak VRAM, transition step/time, and realized passes
`num_steps * 256 / 50_000`.

## Falsifiable Hypothesis

Replacing the batch-shared mixup coefficient with independent per-example
`Beta(0.2, 0.2)` coefficients, while retaining the accepted 65% cutoff and all
other behavior, will preserve at least 95% matched mixup throughput and raise
fixed-seed `best_test_acc` from 94.07% to at least 94.17% within the 300-second
budget.

A valid scored result below 94.17% is a no-improvement and rejects coefficient
decorrelation as a sufficient standalone refinement. Do not rescue it by adding
derangement, changing alpha, symmetrizing coefficients, changing the cutoff,
or running another seed in EXP-015.

## Evidence

- `knowledge/papers/mixup.md`: mixup trains on convex input/target pairs and
  improves CIFAR generalization; this proposal keeps that mechanism and its
  accepted alpha while changing only batchwise coefficient correlation.
- `knowledge/papers/time-matters-regularization.md`: early regularization can
  retain its generalization effect after removal, supporting the unchanged 65%
  transition and hard-label tail.
- `experiments/002/04-analysis.md`: alpha-0.2 mixup through 65% reached 94.07%
  with 141.9 passes and final accuracy equal to best, establishing the recipe
  whose granularity is being refined.
- `03-experiment-learnings.md` and `04-results.tsv`: stronger mixup, earlier
  removal, CutMix, and additive dropout all regress at normal exposure, arguing
  for objective-preserving diversity rather than more regularization.
