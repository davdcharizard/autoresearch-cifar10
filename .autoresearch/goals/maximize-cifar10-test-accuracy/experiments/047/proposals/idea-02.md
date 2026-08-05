# Proposal: Early Post-GAP Pooled-Feature Mixup Replacement

## Recommendation

Advance as a medium-low-confidence, one-score representation test against
accepted commit `a7c42dc`. During exactly the first `65%` of counted training,
replace input-pixel interpolation with one interpolation of the accepted
`128`-dimensional post-GAP vector immediately before the accepted pooled
residual MLP. Use the exact accepted one batch-shared `Beta(0.2,0.2)` draw,
ordinary permutation, paired target cross-entropies, and cutoff. Run one
forward per batch. The final `35%` and evaluation remain the exact ordinary
input/hard-label model path.

This is replacement, not composition. Early inputs receive the accepted
crop/flip/worker-safe RandAugment but are not convexly blended. Do not retain
input mixup, add an auxiliary loss, alternate locations, randomize placement,
or run a second forward. Preserve all parameters, initialization, GAP, pooled
MLP, scale `0.1`, classifier, optimizer, LR, weight decay, data order, batch
size, seed, temporal controls, evaluator, and budget.

EXP046 did not change the frontier: its mean-fill candidate stopped before
scoring on a prospective loader-service gate, so the baseline remains
`94.48%` and the accuracy effect of mean fill is unresolved. Its relevant
lesson is procedural. The present treatment changes no CPU transform or
worker behavior and therefore needs no new loader gate; qualification should
focus on GPU semantics and counterbalanced complete-step exposure. EXP046's
analysis also identifies fixed pooled-feature mixup as a distinct remaining
high-impact replacement idea, while keeping the prior cautious because it
removes accepted input mixup.

## Exact Placement and Bundled Mechanism

Fix the sole interpolation point after

```python
out = F.relu(self.bn(out))
out = F.adaptive_avg_pool2d(out, 1)
out = out.view(out.size(0), -1)
```

and immediately before

```python
out = out + POOLED_HEAD_SCALE * self.pooled_head(out)
```

This placement is prospective and final. Earlier stage boundaries would mix
large spatial tensors and send synthetic maps through downstream convolutions
and BatchNorms; after `layer3` but before final BN would still alter final BN
statistics on mixed maps. Post-GAP mixing touches only `128` values per image,
leaves all spatial operations and BatchNorms on ordinary augmented examples,
and retains the accepted nonlinear `128 -> 64 -> 128` MLP downstream. Moving
after that MLP would leave only an affine classifier and collapse much of the
intended nonlinear-between-feature test.

The treatment deliberately bundles two changes relative to accepted input
mixup:

1. every spatial BatchNorm sees individually augmented images rather than
   pixel-interpolated images; and
2. the accepted pooled MLP is trained on convex line segments between pooled
   representations rather than on pooled representations of convex pixels.

These effects cannot be separated by the sole score. A gain would support only
the complete clean-spatial-BN plus pre-MLP feature-interpolation replacement;
a loss could mean that accepted mixed-pixel invariance was essential, that the
late feature mixture was ineffective, or that their combination was harmful.
Do not describe the score as isolating manifold linearity or BatchNorm quality,
and do not launch a follow-up arm to deconfound them.

## Exact Forward and Label Semantics

For an early batch of independently transformed inputs `x`, labels `y`, scalar
`lambda`, and permutation `p`, define

```text
z_i       = GAP(ReLU(BN(layer3(layer2(layer1(conv1(x_i)))))))
z_mix_i   = lambda * z_i + (1 - lambda) * z_p(i)
r_mix_i   = z_mix_i + 0.1 * W2 ReLU(W1 z_mix_i)
logits_i  = fc(r_mix_i)

loss = lambda       * CE(logits, y)
     + (1 - lambda) * CE(logits, y[p])
```

The same scalar and same permutation must align each feature arm and label arm
in the same direction. Keep natural self-pairs and same-class pairs from
ordinary `torch.randperm`; do not derange, class-balance, reverse,
canonicalize `lambda`, symmetrize, or draw per-example coefficients. EXP015
showed that per-example draws lose useful batch coherence, while EXP004/020
and EXP005/035 bracket the accepted `65%` duration and alpha `0.2` strength.

At `progress >=0.65`, draw neither a coefficient nor a permutation and execute

```text
z_i      = accepted pooled feature for x_i
r_i      = z_i + 0.1 * W2 ReLU(W1 z_i)
logits_i = fc(r_i)
loss     = CE(logits, y)
```

For any fixed state and input, the hard/evaluation function and operation order
must be accepted-exact. The learned state will differ because of the early
treatment; that expected trajectory difference is not a tail-semantics failure.
Preserve the pre-step time predicate, one mixup-disable log, and accepted
exhausted-iterator RandAugment transition.

## Minimal Production Change

Keep all constants. Replace the input-blending helper with a pairing helper
that preserves draw order:

```python
def mixup_pairing(targets, distribution):
    mix = distribution.sample()
    permutation = torch.randperm(targets.size(0), device=targets.device)
    return targets, targets[permutation], mix, permutation
```

Add one default-`None` internal argument to `WideResNet.forward` and blend only
after accepted flattening:

```python
def forward(self, x, feature_mix=None):
    # accepted conv1/layer1/layer2/layer3/final BN/ReLU/GAP/flatten
    ...
    if feature_mix is not None:
        mix, permutation = feature_mix
        out = mix * out + (1.0 - mix) * out[permutation]
    out = out + POOLED_HEAD_SCALE * self.pooled_head(out)
    return self.fc(out)
```

The early branch is exactly

```python
targets_a, targets_b, mix, permutation = mixup_pairing(
    targets, mixup_distribution
)
outputs = model(inputs, feature_mix=(mix, permutation))
loss = mix * F.cross_entropy(outputs, targets_a) + (
    1.0 - mix
) * F.cross_entropy(outputs, targets_b)
```

The hard branch and evaluator keep the one-argument `model(inputs)` call. Do
not expose a public placement setting, return features, unroll the MLP, detach
an arm, blend in place, retain `mixed_inputs`, add production telemetry, or
change the transition text/cadence.

## State and RNG Contract

The model remains exactly `1,003,482` trainable parameters with identical
parameter/buffer names, order, shapes, dtypes, initialization bytes,
seed-36036 pooled-head construction, optimizer groups/options, and initial
CPU/CUDA RNG. No module, parameter, buffer, optimizer state, or persistent
configuration is added.

Every early step calls device `Beta(0.2,0.2).sample()` once and
`torch.randperm(256, device=...)` once, in accepted order before the RNG-free
forward. The candidate uses the permutation to gather `z`; accepted uses it to
gather `inputs`. With cloned pre-draw state, coefficient bytes, permutation
bytes, and post-draw/post-step CUDA RNG must match accepted because all changed
kernels are deterministic and consume no RNG. No CPU draw is added, so sampler
and worker trajectories remain accepted. The hard tail consumes no mixup RNG.

Candidate spatial BatchNorm running statistics intentionally match an ordinary
hard forward for the same initial state/input, not the accepted input-mix
forward. This is the clean-BN half of the bundled mechanism. Before the pooled
blend, candidate spatial inputs, activations, and BN updates must equal the
ordinary unblended path exactly; after it, logits, gradients, and learned state
are expected to differ.

## Evidence and Counter-Hypothesis

The local knowledge base supports convex example/label interpolation for CIFAR
generalization and early regularization followed by late clean refinement. It
contains no separate manifold-mixup source, and this refresh is offline; the
placement claim is therefore grounded in local code and results rather than a
new literature assertion.

- EXP002 established the strongest relevant positive: early input mixup plus
  the hard tail gained `0.69` points. This proposal preserves its coefficient,
  labels, pairing, and timing but risks losing its input-level mechanism.
- EXP015 explicitly left representation-level mixup open while showing that
  per-example coefficients regress at normal exposure. Keep one scalar.
- EXP027 established that early invariance composes with added stage-3
  capacity. EXP036 established that the post-GAP residual MLP improves both
  accuracy and loss at `130.304` passes. The fixed placement targets their
  interface without added spatial compute.
- EXP041 showed that another classifier invocation and auxiliary direct-path
  CE weaken the accepted head. Keep one refined path, one classifier call, and
  one paired objective.
- EXP046 produced no accuracy evidence, but its stable production-paced loader
  delivery and next-step analysis favor a GPU-local/post-pooling candidate
  rather than another worker-transform intervention.

The positive hypothesis is that convex pixels are a low-level proxy for the
desired decision-boundary linearity. Mixing after clean feature extraction may
avoid unnatural pixel composites and synthetic BN populations while forcing
the successful nonlinear head and the backbone gradients to support smooth
between-example decisions.

The counter-hypothesis is stronger than for an orthogonal addition. Input
mixup is proven locally, and its benefit may arise precisely because the whole
spatial backbone learns from mixed pixels. A post-GAP mixture may be too easy,
may create feature vectors the backbone never emits, and leaves only a small
MLP to process the line segment. Batch-shared lambda already makes accepted
mixed-input BN populations coherent, so clean BN statistics may add nothing.

## Semantic and Analytic Gate

Use an ignored evaluator-free harness with independent
`git show a7c42dc:train.py` reference. Block evaluator invocation and test-data
construction. Print all measurements before assertions and require:

1. Production scope changes only the pairing helper, default-`None` forward
   argument/post-GAP conditional, and early call. No constants, spatial
   operation, MLP/classifier, loss weights, optimizer, data transforms,
   workers, schedule, evaluator, cadence, or summary change.
2. Candidate/accepted construction from cloned seed-42 CPU/CUDA state yields
   identical model state bytes and post-construction RNG, exactly `1,003,482`
   parameters, identical optimizer membership/order/options, and no new state.
3. With `feature_mix=None`, fixed CPU/CUDA fixtures produce accepted-exact
   forward values, BN updates, losses, gradients, fresh/preseeded Nesterov
   updates, and RNG in training/evaluation. The hard source must call only this
   path and sample no mixup RNG.
4. On early fixtures, capture production `z`; independently reproduce
   `z_mix`, MLP input/output, refined vector, logits, and paired CE. Require
   identical feature/target permutation and coefficient direction and exactly
   one blend before `pooled_head[0]`.
5. Prove no pixel interpolation and one forward: `conv1` receives the original
   augmented `inputs`; all through-GAP activations and BN updates equal an
   ordinary hard forward; no input/post-MLP/logit blend or auxiliary invocation
   occurs. Use a nonlinear fixture that distinguishes pre-MLP interpolation
   from interpolation of MLP outputs.
6. For `m_i=lambda*z_i+(1-lambda)*z_p(i)` and arbitrary upstream `q`, verify
   the independent mixing Jacobian in FP64/FP32:

   ```python
   expected = lambda_ * q.clone()
   expected.index_add_(0, permutation, (1.0 - lambda_) * q)
   ```

   Require it to match `dL/dz`, then require finite nonzero complete
   backbone/MLP/classifier gradients and independently replayed optimizer
   updates for both early and hard paths.
7. From cloned pre-draw state, require accepted and candidate coefficient,
   permutation, post-draw RNG, and post-step RNG equality. Reconfirm strict
   `65%` predicate, one transition, accepted RandAugment boundary, batch
   `256`, `drop_last=True`, and finite-loss guard.
8. Check `lambda=1` and identity-permutation invariants within predeclared
   floating tolerances; retain natural self/same-class pairs and exact
   mixed-target equivalence.

Report only as non-gating diagnostics: pooled pair distance/cosine,
mixed/unmixed norm, self/same-class incidence, head Jensen gap, candidate versus
accepted-input-mix logits/loss, grouped gradient cosines/norm ratios, and clean
versus mixed-input BN-stat deltas. None may choose another location, detach,
normalize, alpha, cutoff, pairing, loss, or compound path.

## H20 Timing and Exposure Gate

The CPU transform, worker payload, and loader are source-identical, so do not
repeat EXP046's loader qualification or add a CPU feasibility condition.
Accepted input mixup blends `256x3x32x32` values without input-gradient
backward. Candidate removes that blend and adds a differentiable gather/blend
over `256x128`; its backward includes a small scatter. Convolution, parameter,
optimizer, and inference work remain unchanged. Near-neutral throughput is
expected but must be measured.

On one idle H20, compare complete production-equivalent accepted and candidate
steps for early input/feature mixup and hard-label regimes. Include pinned H2D,
LR writes, zeroing, active draws/interpolation, one forward, paired loss/finite
guard, backward, coupled Nesterov update, and synchronization. Use at least 20
disposable warmups and two local `A/C/C/A` cycles, yielding four retained
windows of at least 50 steps per arm/regime. Restore the same model, optimizer,
fixture, and pre-draw RNG state for each paired window; print every window
before gates.

Using the four-window medians, preregister

```text
retention =
  (0.65 / candidate_feature_mix_ms + 0.35 / candidate_hard_ms) /
  (0.65 / accepted_input_mix_ms   + 0.35 / accepted_hard_ms)

projected_passes = 130.304 * retention
```

Require every arm and paired-ratio population CV `<=5%`, candidate peak
allocation `<2,048 MiB`, `retention >=127/130.304 = 0.9746439096`, and
projected passes `>=127.0`. A stable miss ends the experiment before scoring.
Do not rerun timing, reorder aggregation, move placement, detach, retain input
mixing, compile only one arm, or relax the floor.

## Sole Score and Classification

After gates pass, reconfirm baseline `94.48%` at `a7c42dc`, one idle H20,
local training data, frozen `prepare.py`/evaluator, exact `train.py` scope, and
no stale `run.log`. Run once at seed 42:

```bash
timeout 600s uv run train.py > run.log 2>&1
```

Require exit zero, one finite summary, `300.0-300.1` counted seconds, wall
under 600 seconds, exactly `1,003,482` parameters, correct ordered transitions,
unique accepted-cadence evaluations, and no numerical/CUDA/worker/evaluator or
integrity fault. Compute passes as `num_steps*256/50000`.

A structurally valid completion remains the sole score regardless of exposure.
Success requires both `best_test_acc >=94.58%` and realized passes `>=127.0`.
A valid low-exposure completion is a recorded non-success, not an invalid run
and not grounds for rerun. Final accuracy/loss are descriptive only. Never use
intermediate or final test behavior to select a placement, restore input mixup,
compound paths, or launch another score.

## No-Rescue Closure

- A valid `>=127`-pass miss closes this exact post-GAP/pre-MLP replacement and
  immediate placement family. Do not try stage1/2/3, pre-BN, post-head, logit,
  or random-layer mixup; input plus feature mixing; alternating placement;
  another alpha/cutoff; per-example, deranged, class-aware, reversed, or
  symmetric pairing; detach; feature normalization/rescaling; auxiliary loss;
  seed change; or rerun.
- A score `>=94.58%` below 127 passes is not success and does not authorize a
  speed rescue or second score.
- A normal-exposure success supports only the complete bundled treatment. It
  does not establish clean BN statistics, post-GAP placement, or feature
  manifold linearity independently and does not authorize a sweep.
- A timing failure closes systems viability without an accuracy claim. An
  invalid score permits only repair of an independently demonstrated
  infrastructure/verifier defect with production semantics frozen.

## Falsifiable Hypothesis

If the accepted early convex-label prior is more useful when applied between
cleanly extracted decision representations than between raw pixels, then
replacing only early input interpolation with one batch-shared alpha-0.2
post-GAP/pre-MLP interpolation through exactly `65%` will retain at least
`127` passes and raise fixed-seed `best_test_acc` from `94.48%` to at least
`94.58%`. The honest prior is medium-low because the candidate is near-free and
precise but deletes the strongest locally validated input regularizer and
cannot separate its BN and placement effects.

## Local Sources

- `01-definition.md`, `02-system-understanding.md`,
  `03-experiment-learnings.md`, `04-results.tsv`, and accepted `train.py`.
- `knowledge/README.md`, `knowledge/papers/mixup.md`, and
  `knowledge/papers/time-matters-regularization.md`.
- `experiments/002/04-analysis.md`, `experiments/004/04-analysis.md`,
  `experiments/005/04-analysis.md`, `experiments/015/04-analysis.md`,
  `experiments/020/04-analysis.md`, and `experiments/035/04-analysis.md`.
- `experiments/027/04-analysis.md`, `experiments/036/04-analysis.md`, and
  `experiments/041/04-analysis.md`.
- `experiments/046/01-idea-review.md`, `03-execute.md`, and `04-analysis.md`.
