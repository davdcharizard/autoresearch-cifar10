# Proposal: Decay-Calibrated Class-Direction Separation

## Recommendation

Do **not** regularize the raw `fc.weight` rows toward pairwise orthogonality.
Softmax probabilities are unchanged if the same vector is added to all ten
classifier rows, but raw-row angles change under that gauge transformation.
Literal orthogonality can therefore reduce its objective by learning a common
weight component that has no decision effect and is opposed only by weight
decay. Hard orthogonalization also changes the inference map, row norms, and
optimizer parameterization at once.

The only defensible version is a gauge-invariant directional treatment:
center the ten class vectors, normalize their directions, and softly reduce
their deviation from the regular-simplex Gram matrix. Ten nonzero centered
vectors cannot be mutually orthogonal because they sum to zero; their
maximally symmetric angular geometry instead has pairwise cosine
`-1 / (10 - 1)`. Scale the training-only penalty prospectively so its initial
gradient Frobenius norm exactly equals the accepted coupled-decay gradient
norm on `fc.weight`. This retains accepted radial decay and adds an equally
sized, Frobenius-orthogonal angular force without a tunable coefficient.

This corrected treatment is worth at most one low-to-medium-confidence score.
It has a defensible scale, but there is no diagnosis that the learned class
directions are insufficiently separated and the accepted initialization is
already near random-simplex geometry.

## Exact Treatment

Add a pure helper to `train.py`:

```python
def classifier_direction_penalty(weight):
    num_classes = weight.size(0)
    centered = weight - weight.mean(dim=0, keepdim=True)
    directions = F.normalize(centered, dim=1)
    gram = directions @ directions.t()
    target_cosine = -1.0 / (num_classes - 1)
    pair_errors = torch.triu(
        (gram - target_cosine).square(), diagonal=1
    )
    return 2.0 * pair_errors.sum() / (num_classes * (num_classes - 1))
```

After accepted seed-42 model construction and `.to(device)`, but before
optimizer construction and before the counted training timer, derive one
fixed Python-float coefficient from the untouched initialized classifier:

```python
initial_penalty = classifier_direction_penalty(model.fc.weight)
initial_angular_grad = torch.autograd.grad(
    initial_penalty, model.fc.weight
)[0]
classifier_direction_scale = (
    WEIGHT_DECAY * model.fc.weight.detach().norm()
    / initial_angular_grad.detach().norm()
).item()
```

Fail if the penalty, gradient, or coefficient is nonfinite, or if the gradient
norm is zero. The computation must leave `model.fc.weight.grad is None`; do
not call `backward`, mutate a parameter, retain the graph, or derive the scale
again later. Print the fixed coefficient once for auditability.

For the accepted seed-42 model at `a7c42dc`, an evaluator-free CPU
reconstruction gives:

- `||fc.weight||_F = 4.500954151` and accepted decay-gradient norm
  `5e-4 * ||W||_F = 0.002250477`;
- centered-row norms from `1.217378` to `1.543766`, safely away from the
  normalization epsilon;
- mean pair cosine `-0.110607` versus simplex target `-0.111111`, with pair
  cosines spanning `[-0.277645, 0.062672]`;
- mean squared simplex error `0.009310339`, unscaled gradient norm
  `0.031956214`, and derived scale approximately `0.070423774`;
- scaled initial penalty value approximately `0.000655669`.

These values are preregistered semantic expectations, not tuning feedback.
The coefficient rule, rather than the rounded value, is authoritative.

Add the penalty continuously to both accepted classification losses:

```python
if use_mixup:
    classification_loss = mix * F.cross_entropy(outputs, targets_a) + (
        1.0 - mix
    ) * F.cross_entropy(outputs, targets_b)
else:
    classification_loss = F.cross_entropy(outputs, targets)
loss = classification_loss + (
    classifier_direction_scale
    * classifier_direction_penalty(model.fc.weight)
)
```

The displayed smoothed training loss may include the penalty. Frozen test loss
does not: evaluation still calls cross entropy on unchanged logits. Preserve
the exact classifier forward, bias, initialization, `5e-4` matrix decay,
optimizer groups, LR, Nesterov momentum, pooled residual MLP, model graph,
data path, augmentation windows, seed, and evaluation cadence. The treatment
adds no parameters, buffers, optimizer groups, forward-time normalization,
projection, learned gain, target margin, or schedule.

## Why This Scale Is Defensible

EXP037 and EXP038 bracketed only the **radial** classifier regularization:
zero and `1e-3` decay both lost at normal exposure versus accepted `5e-4`.
The proposal therefore leaves `5e-4` untouched. The normalized centered
simplex objective is invariant to both a common row shift and a positive
global rescaling. Its gradient has zero row mean and, up to numerical error,
zero Frobenius inner product with `W`; it acts in an angular direction while
coupled decay acts radially.

Matching `||scale * grad(R)||_F` to `||5e-4 * W||_F` at the accepted
initialization gives the new directional force the same initial optimizer
scale as the locally validated radial force. Both are multiplied by the same
LR and transformed by the same Nesterov rule. This is not evidence that equal
force norms are optimal, but it is a prospective, dimensionally direct
one-point allocation rule and is stronger than choosing `1e-2`, `0.1`, or a
schedule by convention.

Rejected alternatives:

- **Raw off-diagonal cosine penalty:** gauge-dependent and can reward a
  decision-irrelevant shared class-vector component.
- **Hard row orthogonalization in forward:** coefficient-free, but changes
  initial logits and inference semantics, equalizes or otherwise entangles row
  norms, makes accepted decay indirect, and introduces inverse-square-root or
  QR gradient risks.
- **Post-step orthogonal projection:** coefficient-free, but breaks the
  relation between Nesterov momentum buffers and parameters and is order
  sensitive under QR.
- **Arbitrary soft coefficient:** operationally simple but has no local scale
  rationale. If initial gradient matching is not accepted as sufficient, the
  entire idea should be rejected rather than swept.

## Evidence and Expected Mechanism

- EXP036 improved the frontier to 94.48% with a cheap nonlinear pooled head,
  showing that post-spatial decision geometry can matter while preserving the
  expensive backbone (`experiments/036/04-analysis.md`).
- EXP037/038 show that accepted classifier decay is important and locally
  bracketed. The proposed force is tangent to overall classifier scale and
  thus tests a distinct direction rather than reopening decay strength
  (`experiments/037/04-analysis.md`, `experiments/038/04-analysis.md`).
- The accepted run nearly interpolates its hard tail but finishes at 0.2456
  test loss, leaving boundary/generalization quality as the measured limiter
  (`02-system-understanding.md`).
- The penalty touches only a `10 x 128` tensor and has no inference cost. Its
  extra Gram and normalization arithmetic should be negligible relative to
  the convolutional forward/backward bottleneck.

The evidence is mechanistic, not diagnostic. Initial centered class directions
already have the expected mean simplex cosine and only their pairwise spread
is irregular. The plausible benefit is therefore modest: reduce accidental
anisotropy among class boundaries while leaving learned features, logits, and
class-vector radii free. It may instead erase useful semantic structure among
CIFAR classes.

## Semantic and Optimizer Gates

Use an ignored evaluator-free verifier with an independent
`git show a7c42dc:train.py` accepted module. Guard `prepare.Eval` and all test
dataset access before imports. Abort before timing unless all checks pass:

1. Production changes are limited to the helper, one pre-optimizer fixed-scale
   derivation/log line, classification-loss naming, and one additive penalty.
   `prepare.py`, evaluator behavior, and every accepted constant remain exact.
2. Accepted and candidate construction from cloned CPU/CUDA RNG states yields
   byte-identical parameters and buffers, identical post-construction RNG, the
   same 1,003,482 parameters, and identical optimizer membership/options.
3. Independently recompute the penalty and coefficient from the seed-42
   initial tensor. Require finite values near the preregistered diagnostics,
   exact agreement with the production formula, no populated `.grad`, and no
   parameter, buffer, RNG, or allocator-persistent model-state mutation.
4. Prove `R(W + 1 v^T) == R(W)`, `R(aW) == R(W)` for positive finite `a`, and
   class-permutation invariance within tight FP32 tolerances. Require the
   penalty gradient's row mean and Frobenius inner product with `W` to be near
   zero, and its norm after scaling to equal `5e-4 * ||W||_F` within `1e-5`
   relative error.
5. On cloned synthetic early-mixup and hard-label fixtures, require exact
   inputs, lambda/permutation, logits, classification CE, BN buffers, and RNG.
   Before stepping, every non-`fc.weight` gradient must equal accepted; the
   candidate classifier gradient must equal accepted classifier gradient plus
   the independently calculated scaled angular gradient.
6. Verify fresh and deterministic-preseeded Nesterov updates against an
   independent coupled-decay oracle. Existing `5e-4 * W` must remain present;
   the only direct cross-arm parameter difference after one common pre-step is
   `fc.weight`, and all gradients, buffers, parameters, and losses must be
   finite.
7. Verify the penalty executes in both temporal regimes, is never called by
   model evaluation, and does not alter mixup/RandAugment transitions,
   time-based LR, finite-loss guard, once-per-epoch evaluation, or summary.

No semantic failure may be repaired by changing centering, target angle,
normalization epsilon, coefficient rule, activation window, or decay.

## Timing and Scored Gates

Run balanced complete-body H20 timing only after semantics pass. Use at least
20 warmups and four counterbalanced windows of at least 50 updates for accepted
and candidate in both early-mixup and hard-label regimes. Include pinned H2D,
LR writes, zeroing, forward, exact loss including the candidate penalty,
backward, Nesterov step, and synchronization. Require every population CV at
most 5%, candidate peak below 2,048 MiB, and print all raw windows before
asserting.

Using accepted EXP036 exposure `130.304`, require:

```text
retention = (0.65 / c_mix + 0.35 / c_hard) \
          / (0.65 / a_mix + 0.35 / a_hard) >= 0.974644
projected_passes = 130.304 * retention >= 127.0
```

If timing passes, remove stale `run.log` and run exactly once:

```bash
timeout 600s uv run train.py > run.log 2>&1
```

Require one H20, seed 42, exit zero, finite summary, 300.0-300.1 counted
seconds, less than 600 wall seconds, unchanged parameter count, correct single
temporal transitions, and evaluation epochs equal every fifth epoch union the
final epoch with no duplicate. Record realized passes, peak VRAM, derived
coefficient, best/final accuracy, final test loss, and best-final gap. A
completed result below 127 passes is valid and nonrepeatable but cannot support
a mechanism-level conclusion.

## Falsifiable Hypothesis and Closure

If irregular angular spacing of centered class vectors limits the accepted
pooled representation, then continuous centered-simplex regularization at the
prospectively decay-matched scale will retain at least 127 projected and
realized passes and raise fixed-seed `best_test_acc` from 94.48% to at least
94.58%. Preregister `final_test_acc >= 94.45%` and
`final_test_loss <= 0.2456` as corroboration; neither can rescue a primary
metric miss or veto a valid primary success.

A valid normal-exposure score below 94.58% rejects this continuous
decay-norm-matched class-direction mechanism. Do not rescue it with half/double
coefficients, raw orthogonality, hard QR/polar projection, per-step adaptive
normalization, early-only or late-only schedules, another seed, classifier
decay changes, or pooled-head tuning. The miss would not close unrelated
feature normalization or a future geometry intervention derived from a new
training diagnosis. A timing failure closes only this exact implementation;
an invalid run is diagnosed but never rerolled.

## Risks

- The accepted class directions already have mean cosine almost exactly equal
  to the simplex target, so remaining angular variance may be benign or useful
  and the effect may be below the 0.10-point acceptance margin.
- CIFAR classes have nonuniform semantic relationships; equal angular spacing
  can discard useful shared structure and worsen top-1 boundaries even if the
  penalty decreases cleanly.
- Initial gradient-norm matching gives a defensible one-shot coefficient, not
  an optimality theorem. The fixed coefficient's relative force changes as
  classifier norms and the Gram matrix evolve.
- Normalization is stable at initialization but centered row collapse later
  would amplify gradients. The scored finite-loss guard catches numerical
  failure; no epsilon or clipping rescue is allowed after preflight.
- Added angular pressure stacks with mixup, RandAugment, matrix decay, and the
  pooled MLP. Local history shows additive regularization often harms this
  already strong recipe.
