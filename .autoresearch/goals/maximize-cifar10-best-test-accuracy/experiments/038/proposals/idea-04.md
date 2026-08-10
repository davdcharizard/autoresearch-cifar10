# Proposal: Intrinsically Bounded Average-plus-RMS Readout

## Decision and exact formulation

Keep the accepted EXP010 model, classifier, optimizer, curriculum, and schedule, but replace
only the final global-average descriptor by a fixed convex average/RMS descriptor.  For the
nonnegative post-ReLU final map `A` of shape `[B, C, 8, 8]`, let `n=64` and

```text
mu[b,c]  = (1/n) sum_i A[b,c,i]
rms[b,c] = sqrt((1/n) sum_i A[b,c,i]^2)
z[b,c]   = mu[b,c] + (1/n) * (rms[b,c] - mu[b,c])
```

The single preregistered coefficient is therefore exactly `alpha = 1/64`; it is not
calibrated, learned, swept, or changed by phase.  The corresponding `train.py` change is:

```python
out = self.layer3(out)
avg = out.mean(dim=(2, 3))
rms = torch.linalg.vector_norm(out, dim=(2, 3)) / 8.0
out = torch.lerp(avg, rms, 1.0 / 64.0)
return self.fc(out)
```

Preflight must prove the accepted final shape is exactly `8x8`.  `vector_norm` is chosen
instead of `sqrt(mean(square))` because PyTorch defines a finite zero subgradient for an
all-zero vector; an explicit zero-map backward oracle is mandatory.  No epsilon, learned
scale, extra classifier, normalization, clipping, diagnostic hook, or coefficient rescue is
allowed in production.

## Representation rationale and distinction from prior pooling failures

Global average is exactly area sensitive, but can dilute a compact class-bearing response.
RMS gives a smooth energy statistic: all positive locations contribute, with larger
responses weighted more strongly.  For an idealized response of height `a` on `k` of 64
sites and zero elsewhere, `rms/mu=sqrt(64/k)`, so the proposal increases the descriptor by
10.94% for one-site evidence, 4.69% for four-site evidence, 1.56% for sixteen-site evidence,
and 0% for a spatially constant feature.  It therefore mildly preserves localized CutMix
features while retaining almost all area-proportional evidence.

This is materially distinct from EXP014 and EXP031.  EXP014 added an independent raw-max
classifier whose first gradient was 4.10x the average path and whose weights reached 3.96x
the accepted classifier norm, collapsing accuracy to chance.  EXP031 used a hard-max
residual calibrated to 10% aggregate RMS only at initialization; sparse examples later
reached a 4.34 perturbation ratio, 1.58x updates, and class collapse.  The present method has
no argmax, new parameters, optimizer state, initialization calibration, or sparse
independent logits.  Its descriptor and pooling Jacobian are bounded for every nonnegative
example at every training step, not merely on a calibration corpus.

The external support is directional rather than exact: AISTATS 2016 mixed-pooling results
support responsive combinations of spatial statistics, and CutMix supports preserving
localized class-bearing evidence.  Neither establishes the `1/64` RMS point on this
300-second width-2 ResNet-20 recipe.

## Scale and gradient bounds

For every nonnegative channel map, norm inequalities give

```text
mu <= rms <= sqrt(n) * mu = 8 * mu.
```

Consequently the bound is componentwise, per example, and lifetime invariant:

```text
mu <= z <= (71/64) * mu
0 <= z - mu <= (7/64) * mu
||z - mu||_2 <= (7/64) * ||mu||_2.
```

Thus the maximum descriptor perturbation is 10.9375%, including the one-hot spatial worst
case; no small denominator can make it dominant as in EXP031.  For `rms>0`, the pooling
Jacobian coefficient at site `i`, relative to average pooling's `1/n`, is

```text
J_i / (1/n) = 63/64 + (1/64) * A_i/rms,
```

and `0 <= A_i/rms <= 8`, hence every coefficient lies in `[63/64, 71/64]` or
`[0.984375, 1.109375]`.  Every site retains a dense gradient floor; there is no hard-max
single-index backward.  At an all-zero map, `vector_norm` contributes PyTorch's zero
subgradient and the coefficient is `63/64`.

The oracle must compare FP64 implementation output with the formula on zero, constant,
one-hot, random nonnegative, and adversarial sparse maps; require the componentwise and L2
bounds above within dtype tolerance.  It must run VJPs with signed random upstream vectors
and compare against the analytic Jacobian, check the exact zero-map subgradient, and prove
finite hard/CutMix loss and gradients.  Constant positive maps must be output-identical to
average pooling and have the identical pooling Jacobian.  Linear-classifier logits are not
given a relative bound because cancellation can make their denominator zero; report instead
the denominator-safe absolute logit change and the operator-norm ceiling
`||W(z-mu)|| <= ||W||_2*(7/64)||mu||_2`.

## Expected effect and risks

The hypothesis is that this small, dense, scale-safe salience bias improves final
representation quality without weakening EXP010's 89.73% switch fit and raises seed-42
`best_test_acc` from 94.15% to at least 94.25%.  Point expectation is approximately
94.25-94.30%, with unchanged parameter count and nearly unchanged optimizer exposure.

The main scientific risk is that `1/64` is safe but too weak to matter; the final 8x8 maps
may already encode localized evidence into channel means.  RMS can still favor high-energy
RandAugment artifacts and is not exactly CutMix-area linear.  The added norm reduction and
distributed backward may cost useful fixed-budget steps, while the +0.10 gate is only ten
test examples and one seed provides weak causal evidence.  A valid miss rejects this exact
coefficient; it does not authorize `1/32`, GeM, a learned gate, or post-result tuning.

## Immutable-corpus safety and timing gates

Reuse, without regeneration, the registered EXP022 200-batch strong corpus (SHA-256
`e04dc2fe9d3994cef8bf192401bc36c63f306946fd3b9a2339b9f64040318946`) and EXP028 64-batch
weak corpus (SHA-256
`ffefe980241d9719c8d7f2b44fe81c1b3f94e35003b0a645d3fea5999a745032`).  Verify hashes and
schemas before and after.  First qualify two accepted/accepted controls, then replay
independent accepted/candidate models from identical seed-42 state over all 264 batches.
Require exact 264 BN updates, finite parameters/momentum/logits/loss/gradients, positive BN
variances, and no candidate-only persistent >95% predicted-class concentration.  Controls
must qualify every ratio gate with denominator-safe absolute-plus-relative floors.

At initialization and every 20 strong/16 weak steps, recompute the algebraic descriptor and
Jacobian bounds on each replay batch; any violation is a semantic failure.  As catastrophic
trajectory gates, require candidate terminal strong/weak loss EMA no more than 1.5x the
control envelope, logit RMS and whole-gradient/update RMS no more than 5x qualified control,
maximum candidate update below 25% of parameter norm, and no persistent class collapse.
Record switch-like fit, logit cosine, classifier gradients, whole updates, per-example
descriptor ratios, and class histograms as diagnostics; generic long-horizon model
divergence is not a mechanism-survival gate because the exact nonzero bounded RMS mechanism
is established directly by the oracle.

After one unscored conditioning process, use seven counterbalanced fresh-process
control/candidate pairs on one idle H20.  Each arm restores identical state and byte-identical
hard/CutMix/weak batches, warms 100 steps, then times at least 1,000 synchronized complete
steps with H2D, forward, CE, backward, ordinary SGD, and synchronization included at the
registered 40/40/20 weighting.  Persist means, medians, p95s, CVs, stage times, and memory
before assertions.  Require aggregate candidate/control mean `<=1.05`, every pair `<=1.08`,
trial/rate CV `<3%`, peak allocation `<650 MiB`, and conservative total wall `<540s`.
Ordinary overhead is priced by the real 300-second run; these gates reject only unstable or
catastrophically infeasible execution.

If all gates pass, run exactly once at seed 42 with the unchanged evaluator and at most 19
unique evaluations.  Require only `train.py` changed, 1,073,962 parameters, 300 counted
seconds, total below 600 seconds, one 80% loader transition, accepted CutMix/weak-target
semantics, finite summary, and `best_test_acc >=94.25%` for improvement.  No reroll or
coefficient rescue is permitted.

## Evidence-strength verdict

**Moderate mechanistic evidence, weak direct accuracy evidence; strong enough to retain as
an exploratory finalist, but not a high-confidence lead.**  The intrinsic per-example and
gradient bounds directly cure the specific safety defects of EXP014/031 and make the test
well identified.  Conversely, no local result shows that RMS information is useful, and the
necessarily small safe coefficient may have sub-threshold effect.  It should advance only
if the broader EXP038 review values a clean representation probe over candidates with
stronger empirical upside.

## Sources

- `goals/maximize-cifar10-best-test-accuracy/01-definition.md`
- `goals/maximize-cifar10-best-test-accuracy/02-system-understanding.md`
- `goals/maximize-cifar10-best-test-accuracy/03-experiment-learnings.md`
- `goals/maximize-cifar10-best-test-accuracy/04-results.tsv`
- `goals/maximize-cifar10-best-test-accuracy/knowledge/papers/mixed-pooling.md`
- `goals/maximize-cifar10-best-test-accuracy/knowledge/papers/cutmix.md`
- `goals/maximize-cifar10-best-test-accuracy/experiments/014/04-analysis.md`
- `goals/maximize-cifar10-best-test-accuracy/experiments/031/04-analysis.md`
