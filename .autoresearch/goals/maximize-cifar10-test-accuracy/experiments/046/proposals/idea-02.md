# Proposal: Early Pooled-Feature Manifold Mixup Before the Accepted MLP

## Recommendation

Advance as a medium-low-confidence, one-score replacement test. During exactly
the accepted first `65%` of counted training time, stop interpolating input
pixels. Instead, run each independently augmented image through the complete
accepted spatial backbone, final BatchNorm, ReLU, and global average pooling;
then mix the resulting `128`-dimensional pooled vectors with the accepted one
batch-shared `Beta(0.2, 0.2)` coefficient and ordinary batch permutation. Feed
that mixed vector through the unchanged accepted scale-`0.1` residual MLP and
classifier, and use the same coefficient and permutation for the two target
cross-entropies. The final `35%` remains the exact clean hard-label path.

This is a relocation of the accepted mixup mechanism, not an additional mixup
loss. Do not mix both inputs and features, make a second forward pass, retain an
auxiliary input-mix loss, or alternate placements. Preserve the accepted
architecture, parameters, initialization, optimizer, LR, weight decay,
RandAugment, crop/flip, batch size, seed, coefficient law, pairing law, cutoff,
evaluation, and budget.

The hypothesis is that pixel interpolation is an unnecessarily low-level way
to impose the already successful early convex-label prior. Pooled-feature
mixing retains individually meaningful augmented inputs and clean spatial
BatchNorm statistics, while directly training the accepted nonlinear pooled
head on line segments between examples. The principal risk is equally clear:
accepted input mixup produced a large `+0.69`-point gain in EXP002, whereas
representation-level mixing is unvalidated locally and a very late placement
may be too weak to replace the invariances learned from mixed pixels.

## Fixed Placement Decision

Choose exactly the vector after final `F.relu(self.bn(out))` and
`F.adaptive_avg_pool2d(..., 1).view(...)`, immediately before

```python
out = out + POOLED_HEAD_SCALE * self.pooled_head(out)
```

This placement is selected prospectively from the narrow implementation space:

- After `layer1` or `layer2`, mixing would touch `32x32x32` or `64x16x16`
  values per example and send synthetic maps through several spatial BatchNorm
  and convolution blocks. It offers more nonlinear depth but adds spatial
  gather/blend backward work and an arbitrary downstream-BN confound.
- After `layer3` but before final BN/ReLU, mixing still touches `128x8x8`
  values and changes final BatchNorm statistics on synthetic maps.
- After GAP and before the pooled MLP, mixing touches only `128` values, leaves
  every spatial operation and BatchNorm on real augmented examples, yet retains
  the accepted nonlinear `128 -> 64 -> 128` remapping downstream. EXP036 is
  direct local evidence that this decision-level nonlinear placement is useful.
- After the pooled MLP, only the affine classifier remains. Because affine
  classification commutes with feature interpolation, that placement would be
  close to mixing logits and would not test nonlinear behavior between pooled
  representations.

Do not randomize the layer per batch or example. A random placement would add
an RNG stream, combine mechanisms with very different BatchNorm and compute
semantics, and make a one-score result uninterpretable. Do not move to another
stage if the fixed pooled placement misses.

## Exact Early and Late Semantics

For an early batch of independently crop/flip/RandAugment-transformed inputs
`x`, labels `y`, one scalar `lambda`, and permutation `p`, define

```text
z_i       = GAP(ReLU(BN(layer3(layer2(layer1(conv1(x_i)))))))
z_mix_i   = lambda * z_i + (1 - lambda) * z_p(i)
r_mix_i   = z_mix_i + 0.1 * W2 ReLU(W1 z_mix_i)
logits_i  = fc(r_mix_i)

loss = lambda       * CE(logits, y)
     + (1 - lambda) * CE(logits, y[p])
```

`W1`, `W2`, and `fc` are the accepted tensors and are invoked once. The same
scalar and same permutation must align features and targets in the same
direction. Preserve ordinary `torch.randperm`, including natural same-class
and self-pairs; do not derange, class-balance, canonicalize `lambda`, reverse
pairs, or sample per-example coefficients. EXP015 showed that destroying
batch-shared coefficient coherence costs accuracy, while EXP005/EXP035 and
EXP004/EXP020 bracket the accepted alpha and duration.

At `progress >= 0.65`, sample no coefficient or permutation and execute

```text
z_i      = accepted pooled feature for x_i
r_i      = z_i + 0.1 * W2 ReLU(W1 z_i)
logits_i = fc(r_i)
loss     = CE(logits, y)
```

The code path and function for a fixed state/input must be exactly accepted in
the hard tail and during evaluation. Learned parameter and BN states will of
course reflect the earlier treatment; no claim of trajectory identity is
possible. Preserve the strict pre-step time predicate, the single existing
mixup transition log, and the separately exhausted-iterator RandAugment
transition. RandAugment remains active only under its accepted worker-safe
policy and is applied to each real image before the backbone.

## Minimal Production Implementation

Keep all constants unchanged and add no state. Replace the input-mixing helper
with a pairing-only helper that draws in accepted order:

```python
def mixup_pairing(targets, distribution):
    mix = distribution.sample()
    permutation = torch.randperm(targets.size(0), device=targets.device)
    return targets, targets[permutation], mix, permutation
```

Give `WideResNet.forward` one default-`None` training-only argument. Perform
the optional blend only after the accepted flatten and before the pooled MLP:

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

The early training branch becomes

```python
targets_a, targets_b, mix, permutation = mixup_pairing(
    targets, mixup_distribution
)
outputs = model(inputs, feature_mix=(mix, permutation))
loss = mix * F.cross_entropy(outputs, targets_a) + (
    1.0 - mix
) * F.cross_entropy(outputs, targets_b)
```

The hard branch and evaluator continue to call `model(inputs)` with one
argument. Do not expose feature mixing as a model configuration, constant, or
evaluation mode. Do not unroll or alter the accepted MLP, return auxiliary
features, add hooks/production diagnostics, detach either feature arm, or make
the interpolation in place.

## Parameters, RNG, and State

The model remains exactly `1,003,482` trainable parameters with identical
named parameters/buffers, construction order, Kaiming bytes, pooled-head
seed-36036 isolation, optimizer groups, momentum state allocation, and CPU/CUDA
construction RNG. The treatment introduces no parameter, buffer, module, or
optimizer state.

During each early step, call the existing device `Beta(0.2,0.2).sample()` once
and `torch.randperm(256, device=...)` once, in that accepted order and before
the RNG-free model forward. Because alpha, shape, order, and device are
unchanged, cloned accepted and candidate runs must have byte-identical CUDA RNG
state immediately after pairing and after each step. The candidate merely
uses the sampled permutation to gather `z` instead of using it to gather
`inputs`. No CPU RNG call is added, so sampler and worker trajectories remain
accepted. The clean tail consumes no mixup RNG.

Spatial BatchNorm running states intentionally differ from accepted input
mixup: candidate BNs observe the real augmented `inputs`, exactly as the
accepted hard path would for the same state and batch. This is part of the
mechanism, not a state leak or something to repair by mixing before a BN. On a
cloned initial state, the candidate backbone activations and BN updates before
the optional pooled blend must equal an ordinary hard-path forward.

## Mechanism and Local Evidence

The local knowledge base establishes that convex input/label interpolation can
improve CIFAR generalization and that early regularization followed by clean
refinement is useful. It contains no separate manifold-mixup reference, and
the offline constraint forbids adding one now; the feature-placement rationale
is therefore based on the accepted system and experiment history rather than a
new literature claim.

- EXP002 is the strongest positive evidence: early alpha-0.2 input mixup plus a
  35% hard tail gained `0.69` points. Keep its target prior, coefficient law,
  pairing, and temporal split rather than treating them as tunable.
- EXP004/EXP020 bracket duration at `65%`, and EXP005/EXP035 bracket strength at
  alpha `0.2`. This proposal changes placement only.
- EXP015's per-example coefficients scored `93.79%` at normal throughput;
  retain one coherent batch-shared coefficient and ordinary pairing.
- EXP027 showed early image invariance can compose with added stage-3 capacity,
  while EXP036 showed a cheap nonlinear pooled remapping improves the accepted
  representation. Pooled feature mixup asks whether the same early target
  prior is more effective at their interface.
- EXP041's auxiliary direct-path CE regressed despite normal exposure. Use one
  sole refined-path loss and one classifier call; feature mixup must replace,
  not supplement, the accepted objective path.

The proposed benefit has two inseparable parts. It avoids forwarding
low-level convex pixel composites and lets all BNs estimate statistics from
individual augmented images, while the nonlinear pooled MLP is trained to map
feature line segments consistently to mixed targets. A positive result cannot
identify which part caused the gain.

The counter-hypothesis is that accepted input mixup's low-level invariance is
precisely what helps. A post-GAP convex combination may be too easy, may create
feature vectors unsupported by the backbone, and gives no downstream spatial
network in which to smooth hidden representations. Clean BN statistics may
also be irrelevant because accepted batch-shared lambda already makes each
mixed-input batch statistically coherent. Replacing a proven mechanism makes
the prior lower than for an orthogonal addition, even though the implementation
is clean and cheap.

## Semantic and Analytic Preflight

Use an ignored evaluator-free harness with an independent
`git show a7c42dc:train.py` oracle. Block evaluator invocation and CIFAR-10 test
construction before module import. Print diagnostics before assertions and
require:

1. Only `train.py` production scope changes: the default-`None` forward
   argument and one post-GAP conditional, the pairing helper, and the early
   branch call. No constant, model state, spatial operation, head/classifier,
   optimizer, schedule, augmentation, worker, evaluator, cadence, or summary
   change is allowed.
2. Candidate and accepted construction from cloned seed-42 CPU/CUDA states has
   byte-identical named parameters/buffers and post-construction RNG, the same
   `1,003,482` parameters, identical optimizer group membership/order/options,
   and no new persistent state.
3. With `feature_mix=None`, fixed CPU/CUDA inputs produce accepted-exact logits,
   BN updates, losses, gradients, Nesterov updates, and RNG in training and
   evaluation modes. Require the hard-tail source to call this default path and
   to sample no Beta value or permutation.
4. On fixed early fixtures, hook the actual production pooled vector `z` and
   match the production blend, MLP input/output, refined vector, logits, and
   mixed-target CE against the independent equations above. Require the target
   permutation to be exactly the feature permutation and the scalar to apply
   in the same direction.
5. Prove this is feature-only replacement: candidate spatial inputs and every
   pre-pooling spatial activation equal an ordinary hard forward on `inputs`;
   no interpolated image enters `conv1`; exactly one pooled blend occurs before
   `pooled_head[0]`; and no pre-head, post-head, or logit blend is present.
   Use a nonlinear fixture that distinguishes pre-MLP mixing from mixing MLP
   outputs.
6. Independently verify the blend Jacobian. For upstream pooled-mix gradient
   `q` and `m_i=lambda*z_i+(1-lambda)*z_p(i)`, require

   ```text
   dL/dz = lambda*q + scatter_rows(p, (1-lambda)*q)
   ```

   against autograd in FP64 and FP32. Then require finite nonzero complete
   backbone/head/classifier gradients and exact fresh/preseeded Nesterov
   updates for early and hard paths.
7. From cloned pre-draw CUDA state, require candidate and accepted input-mix
   references to sample identical coefficient bytes, permutation bytes, and
   post-draw/post-step CUDA RNG. Reconfirm CPU sampler/worker RNG isolation and
   the exact one-time `65%` mixup and exhausted-epoch RandAugment transitions.
8. Check invariants: `lambda=1` equals the default hard feature path;
   identity permutation leaves features and targets unchanged; same-class and
   self-pairs remain valid; batch size/drop-last remain `256/True`; and all
   early/hard losses pass the existing finite guard.

Report but never gate or tune from: pooled pair distance and cosine, mixed-to-
unmixed norm ratio, same-class/self-pair counts, head Jensen gap, candidate
versus accepted-input-mix logit/loss deltas, grouped gradient cosines/norm
ratios, and initial BN running-stat deltas between clean and mixed inputs. These
measure whether the intended mechanism is active; they cannot select a layer,
alpha, cutoff, pairing rule, loss weight, or compound variant.

## Throughput and Exposure Gate

Accepted input mixup blends `256x3x32x32` values before a forward. The candidate
removes that blend, runs the same backbone tensor shapes, and adds a differentiable
gather/blend over only `256x128` pooled values. Candidate backward must scatter
through that small blend, so speedup is not guaranteed, but the treatment
should be computationally near-free and may be slightly cheaper. Parameter
count, convolution work, optimizer work, and inference cost are unchanged.

On one idle H20, compare accepted input mixup and candidate pooled-feature
mixup with complete production-equivalent early steps, plus accepted and
candidate hard steps. Include pinned H2D, LR writes, zeroing, coefficient and
permutation draws where active, interpolation, full forward/loss/finite guard,
backward, coupled Nesterov update, and synchronization. Use at least 20
disposable warmups and two counterbalanced `A/C/C/A` cycles, yielding four
retained windows of at least 50 steps per arm and regime from restored fixtures.
Print all windows before assertions.

Using four-window medians, compute

```text
retention =
  (0.65 / candidate_feature_mix_ms + 0.35 / candidate_hard_ms) /
  (0.65 / accepted_input_mix_ms   + 0.35 / accepted_hard_ms)

projected_passes = 130.304 * retention
```

Require all population CVs and paired-ratio CVs `<=5%`, candidate peak
allocation `<2,048 MiB`, `retention >= 127/130.304 = 0.9746439096`, and
projected passes `>=127.0`. A stable miss ends the proposal before scoring.
Do not move the mix earlier, detach the blend, reuse mixed inputs, compile one
arm, alter the pairing helper, or relax the floor as a throughput rescue.

## Sole Score and Decision Contract

After semantic and timing gates pass, reconfirm baseline `94.48%` at
`a7c42dc`, one idle H20, local CIFAR-10, frozen `prepare.py`, exact source
scope, and no stale `run.log`. Run exactly once at seed 42:

```bash
timeout 600s uv run train.py > run.log 2>&1
```

Require exit zero, one finite summary, `300.0-300.1` counted seconds, wall time
under 600 seconds, exactly `1,003,482` parameters, correct ordered transitions,
unique accepted-cadence evaluations, and no numerical, CUDA, worker, evaluator,
or integrity fault. Realized exposure is `num_steps * 256 / 50000`.

Success requires both `best_test_acc >=94.58%` and realized exposure
`>=127.0` passes. Final accuracy and loss are descriptive only. A valid score
below 127 passes remains the sole score and cannot be rerun, but it is not a
successful protected-exposure result. Never use intermediate or final test
behavior to select a placement, restore input mixup, compound mechanisms, or
launch another valid score.

## No-Rescue Closure

- A valid `>=127`-pass miss closes this exact post-GAP/pre-MLP feature-mix
  replacement and its immediate placement neighborhood. Do not try layer1,
  layer2, layer3, pre-BN, post-head, logit, or random-layer mixing; input plus
  feature mixing; placement alternation; another alpha/cutoff; per-example,
  deranged, class-aware, or symmetric pairing; detach; auxiliary losses;
  feature normalization/rescaling; seed changes; or reruns.
- A score at or above `94.58%` but below 127 passes is not success and does not
  authorize a speed-oriented placement or implementation rescue.
- A normal-exposure success supports only the complete replacement treatment.
  It does not prove that clean BN statistics, manifold linearity, or pooled
  placement caused the gain and does not establish feature mixup generally.
- A timing failure closes systems viability without an accuracy claim. An
  invalid scored run permits only repair of an independently demonstrated
  infrastructure/verifier defect while keeping production semantics fixed.

## Falsifiable Hypothesis

If the accepted early convex-label prior is more useful at the decision
representation than at raw pixels, then replacing only early input interpolation
with batch-shared alpha-0.2 post-GAP/pre-MLP feature interpolation, while
preserving the `65%` cutoff and exact clean tail, will retain at least `127`
passes and raise fixed-seed `best_test_acc` from `94.48%` to at least `94.58%`.
The honest prior is medium-low because the treatment is near-free and precisely
isolated but removes a strongly validated input-level regularizer.

## Local Sources

- `01-definition.md`, `02-system-understanding.md`,
  `03-experiment-learnings.md`, `04-results.tsv`, and accepted `train.py`.
- `knowledge/README.md`, `knowledge/papers/mixup.md`, and
  `knowledge/papers/time-matters-regularization.md`.
- `experiments/002/04-analysis.md`: accepted early alpha-0.2 input mixup.
- `experiments/004/04-analysis.md` and `experiments/020/04-analysis.md`:
  duration bracket around `65%`.
- `experiments/005/04-analysis.md` and `experiments/035/04-analysis.md`:
  alpha bracket around `0.2`.
- `experiments/015/04-analysis.md`: failed per-example coefficient mixing and
  explicit representation-mixup avenue.
- `experiments/027/04-analysis.md`, `experiments/036/04-analysis.md`, and
  `experiments/041/04-analysis.md`: accepted invariance/capacity/head context.
