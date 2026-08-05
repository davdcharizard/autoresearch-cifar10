# Mixup-Window Gradient Centralization for Matrix Weights

## Thesis

Centralize each convolutional and linear **data-loss gradient** over all of its
non-output dimensions after `loss.backward()` and before the existing SGD
step, but only while the accepted alpha-0.2 mixup path is active. At the 65%
counted-time transition, stop centralizing and use the accepted hard-label SGD
rule for the remaining 35% of training.

This is a projection of early update geometry, not another augmentation, a
stronger soft target, a new optimizer, or a throughput strategy. It preserves
the accepted WRN-16-2, input and target distributions, LR schedule, selective
weight decay, momentum, Nesterov rule, parameter count, evaluation path, and
late hard-label refinement. The hypothesis is that removing per-output-filter
common-mode components from early loss gradients will reduce redundant feature
drift and produce representations whose final ordinary-SGD refinement crosses
more test examples' top-1 boundaries.

## Choice of Activation Window

Choose **mixup-window-only activation**, not full-run centralization.

EXP-002 shows that the final 35% hard-label tail is useful: accuracy continued
to improve after the 65% transition and finished at the best value. EXP-004
lost 0.16 points by shortening the mixup window, EXP-007 lost 0.33 by changing
late weight decay, and EXP-008 lost 0.27 by suppressing late update amplitude.
Those results make the late optimizer trajectory a locally validated part of
the recipe. Full-run centralization would constrain precisely that clean-label
margin-fitting phase without direct evidence that it is poorly conditioned.

Early-only centralization instead follows the validated division of labor:
alter representation learning during mixed-target training, then return to the
exact accepted loss-gradient/SGD operator for clean-label fitting. The parameter
state and momentum history at 65% will of course differ from the baseline, so
the tail cannot be bitwise identical; its per-step rule, hyperparameters, data
path, and random-number consumption are unchanged. There is no second cutoff,
ramp, coefficient, or result-conditioned fallback to full-run activation.

## Exact Tensor Semantics

For a trainable parameter `p` with data-loss gradient `g = p.grad`, apply

```text
P(g) = g - mean(g, dims=(1, ..., g.ndim - 1), keepdim=True)
```

if and only if `use_mixup` is true, `g` exists, and `p.ndim > 1`. Perform the
subtraction in place under `torch.no_grad()` after the finite-loss check and
`loss.backward()`, immediately before `optimizer.step()`.

For every convolution weight `[C_out, C_in, K_h, K_w]`, reduce over
`(1, 2, 3)`. Each output filter's projected gradient therefore has zero mean
over its `C_in * K_h * K_w` coefficients. For the classifier weight
`[10, 128]`, reduce over `(1,)`, giving a zero-mean 128-element gradient row for
each output class. Do not reduce over dimension 0 and do not centralize across
different output channels/classes.

The accepted network contains exactly 17 affected tensors:

- stem `conv1.weight`: `[16, 3, 3, 3]`;
- stage 1: `[32, 16, 3, 3]`, `[32, 32, 3, 3]`,
  `[32, 16, 1, 1]`, `[32, 32, 3, 3]`, `[32, 32, 3, 3]`;
- stage 2: `[64, 32, 3, 3]`, `[64, 64, 3, 3]`,
  `[64, 32, 1, 1]`, `[64, 64, 3, 3]`, `[64, 64, 3, 3]`;
- stage 3: `[128, 64, 3, 3]`, `[128, 128, 3, 3]`,
  `[128, 64, 1, 1]`, `[128, 128, 3, 3]`,
  `[128, 128, 3, 3]`; and
- classifier `fc.weight`: `[10, 128]`.

All 13 BatchNorm scale tensors, all 13 BatchNorm biases, and `fc.bias` are
one-dimensional and remain untouched. There are no convolution biases. The
helper must use the parameter's actual gradient dtype/device and create no
persistent buffers, optimizer state, model state, random draw, or parameter
hook.

## Update Ordering and Weight Decay

The ordering is deliberately:

1. zero gradients with `set_to_none=True`;
2. execute the accepted mixup or hard-label forward and loss;
3. check that loss is finite;
4. call `loss.backward()`;
5. if `use_mixup`, replace every eligible `p.grad` by `P(p.grad)` in place;
6. call the existing `optimizer.step()` unchanged.

PyTorch SGD then adds the existing coupled decay term internally for every
`p.ndim >= 2`, updates the momentum buffer, applies Nesterov, and updates the
parameter. In symbols during the treatment window, for an eligible weight,

```text
d_t = P(grad_loss_t) + 5e-4 * theta_t
b_t = 0.9 * b_(t-1) + d_t
u_t = d_t + 0.9 * b_t
theta_(t+1) = theta_t - lr_t * u_t
```

with PyTorch's normal first-buffer initialization semantics. Thus the loss
gradient is centralized, but weight decay is **not** projected away. Momentum
and Nesterov are not reimplemented, reset, or separately centralized. At and
after the first `use_mixup == False` step, `d_t` uses the raw hard-label loss
gradient plus the same decay term. Momentum accumulated before the boundary is
retained normally.

Do not manually add weight decay before centralization, set optimizer decay to
zero, centralize momentum buffers, centralize parameters, exclude the
classifier, or centralize only 3x3 convolutions. Each would be a distinct
treatment and would weaken attribution.

## Mechanistic Rationale and Evidence

The accepted model reaches near-zero late training loss yet remains at 94.07%,
so raw fitting ability is not the evident limiter. EXP-009 gained 12.1% more
passes with BF16 but regressed 0.26 points, showing that denser updates alone do
not solve the boundary-generalization problem. EXP-010/011 added low-resolution
capacity and moved accuracy only +0.04/+0.08; EXP-013 reduced evaluation loss
with EMA but moved accuracy only +0.03. The missing effect appears to be a
useful change in learned decision geometry, not simply lower loss, more
parameters, or more exposure.

Gradient centralization removes, independently for each output unit, the
component of the loss gradient parallel to an all-ones vector in that unit's
weight coordinates. In the many convolutional layers whose outputs are
subsequently normalized by BatchNorm, a shared shift of all coefficients in a
filter is plausibly less useful than its contrastive direction. Projecting that
component may reduce correlated filter drift and condition high-LR early
updates without changing activations or targets. Applying it during mixup is
also coherent with mixup's goal: both act while broad class transitions are
being learned, whereas the raw hard-label tail remains free to fit fine class
margins.

This rationale is mechanistic rather than locally demonstrated. The project
knowledge base contains no direct gradient-centralization result, and an
all-ones coefficient direction is not an exact symmetry of convolution or the
linear classifier. In particular, centralizing each classifier row can remove
a genuinely useful direction in the pooled feature basis. This uncertainty is
why the treatment is confined to the first 65% and why exact gradient and
ordering checks are required before scoring.

## Comparison With Prior Failures

- Unlike alpha-0.4 mixup, CutMix, or residual dropout (EXP-003/005/006), this
  adds no target softness, spatial corruption, or stochastic feature masking.
- Unlike removal of late weight decay or cosine-to-zero (EXP-007/008), it keeps
  decay and the entire hard-label LR schedule active and stops before the
  proven late-refinement phase.
- Unlike BF16 (EXP-009), all forwards, losses, gradients, parameters, momentum
  buffers, and updates remain FP32; any exposure loss is small overhead rather
  than a numerical trade.
- Unlike extra width/depth or the rank-64 bottleneck (EXP-010/011/012), it adds
  no parameters or forward MACs and targets optimization geometry directly.
- Unlike EMA (EXP-013), it acts on the trajectory before each early update
  rather than averaging a late trajectory that already lags top-1 refinement.
- Unlike zero residual endpoints (EXP-014), it preserves accepted Kaiming
  initialization and changes all eligible early gradients rather than the
  network's starting function.

These distinctions make it orthogonal enough to test, but the repeated
regularization regressions warn that even an early projection may overconstrain
the model. A valid negative closes this exact projection/window, not every
gradient-conditioning method.

## Required Preflight Discriminators

All semantic and timing preflights must be evaluator-free. Replace
`prepare.Eval` with a fail-closed stub before importing candidate code; do not
load or inspect test data or accuracy.

1. **Enumeration and shape test.** Instantiate WRN-16-2 and require 691,674
   parameters, exactly 17 `ndim > 1` trainable tensors, the exact shapes listed
   above, and exactly 27 untouched one-dimensional trainable tensors. Fail on
   any missing or additional affected parameter.
2. **Projection oracle.** Assign deterministic synthetic FP32 gradients with a
   nonzero mean to every parameter. Run the helper once. For each eligible
   tensor, compare bitwise or within a tight FP32 reduction tolerance against
   an out-of-place reference `g - g.mean(dims, keepdim=True)` and require each
   post-projection row/filter mean to be at most `1e-6` in absolute value.
   Require every one-dimensional gradient to remain bitwise unchanged.
3. **Idempotence and finite behavior.** Apply the helper a second time and
   require numerical equality within `1e-6`; exercise noncontiguous-looking
   real gradient layouts, `grad is None`, and finite zero/constant gradients.
   Any NaN, dtype cast, new `.grad` object requirement, or persistent state is
   a failure.
4. **Real-backward test.** From identical model state, fixed batch, and saved
   RNG state, run accepted and candidate mixup forward/backward paths. Require
   bitwise-equal inputs, pairings, coefficient, logits, loss, and raw gradients
   before projection. Then centralize only the candidate and require all 17
   eligible gradients to match the projection oracle while all 27 vector
   gradients remain unchanged. This proves the intervention begins only after
   backward and consumes no RNG.
5. **Optimizer-ordering oracle.** Clone a model, gradients, and optimizer state
   with nonzero momentum. Manually compute `P(g) + wd * p`, the momentum update,
   Nesterov direction, and parameter update for representative 3x3 convolution,
   1x1 shortcut, classifier, and BatchNorm vector tensors. Require the actual
   one-step result and momentum buffers to match PyTorch's unchanged optimizer
   semantics. In particular, require the decay contribution's mean to remain;
   a zero-mean total update indicates that decay was wrongly centralized.
6. **Boundary test.** Instrument the helper in an evaluator-free miniature
   loop around progress values immediately below, equal to, and above 0.65.
   Require invocation only when the same `use_mixup` Boolean selects the mixup
   loss. At exactly 0.65 and thereafter, require raw gradients to reach SGD,
   with no momentum reset and no second transition.
7. **Hard-label equivalence.** Starting from identical candidate state and
   optimizer state at a synthetic post-boundary step, execute one hard-label
   update with the candidate and accepted update code. Require bitwise-equal
   loss, gradients, parameters, and optimizer state. This isolates any helper
   call accidentally left outside the `use_mixup` branch.

Any failed semantic discriminator rejects the implementation before timing or
scoring. Do not repair failure by changing the affected layer set, using a
coefficient smaller than one, or moving the cutoff.

## Throughput Preflight and Gates

Centralization adds reductions and in-place subtractions for 17 small-to-medium
weight tensors on mixup steps only. It changes no forward graph and allocates
no persistent GPU state. Expected aggregate throughput retention is 97-100%,
with zero treatment overhead during the final 35%; expected peak allocation
remains approximately the accepted 1,094 MiB. The plausible scored exposure is
about 138-142 data-equivalent passes versus accepted 141.9.

Benchmark the complete timed production body on one H20 with cloned accepted
and candidate models, identical resident batches, matched RNG streams, the
actual optimizer groups, forward, loss, backward, candidate helper, optimizer
step, and synchronization. Measure mixup and hard-label regimes separately in
multiple warm, interleaved, order-balanced windows. Report median step time and
population CV for each path, then compute a 65/35 weighted aggregate. The
hard-label candidate path should be statistically identical to accepted; a
material difference there is an implementation or benchmark fault.

Proceed to the scored run only if every timing CV is at most 0.05, weighted
throughput retention is at least 0.97, hard-label retention is at least 0.99,
and the weighted projection from 141.9 accepted passes is at least 137.6
passes. These gates distinguish a lightweight update projection from a hidden
fixed-budget exposure trade. Do not weaken them after measurement.

## Predicted Impact and Falsifiable Decision Rule

The expected accuracy effect is modest but potentially threshold-relevant:
approximately **+0.10 to +0.30 percentage points**, with low-to-medium
confidence because there is no local gradient-centralization precedent. The
most credible positive signature is normal exposure, stable finite training,
and `best_test_acc >= 94.17%`; lower final test loss is supportive but not
required and cannot replace top-1 accuracy.

Run exactly one fixed-seed scored experiment on one H20 with the unchanged
300-second counted budget and external 600-second timeout. Accept only if all
integrity conditions pass and `best_test_acc >= 94.17%`, at least +0.10 points
over the accepted 94.07%. A result from 94.07% through 94.16% is a formal
no-improvement despite being near-flat. A result below 94.07% at normal
exposure indicates that removing early common-mode gradient components harms
useful representation learning or compounds mixup regularization.

Do not retry a valid result with full-run activation, convolution-only
centralization, a partial layer set, a scale coefficient, a different cutoff,
or another seed. Those are separate hypotheses. If the preflight exposure gate
fails, do not consume the scored run; classify this exact 17-tensor Python-loop
implementation as too expensive and return to brainstorming.

## Failure Modes

- **Overconstraint with mixup.** Although not stochastic regularization, the
  projection may remove useful directions while soft targets already reduce
  gradient sharpness, reproducing the additive-regularization failures.
- **Classifier damage.** Per-class row centralization is not justified by
  BatchNorm and can remove a useful pooled-feature direction. Excluding `fc`
  after observing results is forbidden.
- **Momentum persistence.** Stopping at 65% does not erase centralized early
  history; old momentum continues into the tail and may need time to wash out.
  Resetting it would be a confounded treatment.
- **Weight-decay ordering bug.** Centralizing after manually adding decay would
  partly project away the continuously beneficial regularizer, invalidating
  attribution to loss-gradient geometry.
- **Kernel-launch overhead.** Seventeen reductions plus subtractions can cost a
  few percent despite low FLOPs. The matched production preflight must measure,
  not assume, fixed-budget neutrality.
- **Boundary drift.** Gating with epoch or step count instead of the existing
  time-derived `use_mixup` can desynchronize treatment from target semantics.
- **Single-run noise.** The +0.10 acceptance margin is near the likely noise
  scale. The protocol still requires the fixed seed and forbids rerolls; a
  marginal pass should be recorded as low-confidence even though it satisfies
  the formal rule.

## Scope, Overhead, and Effort

Implementation effort is low: one stateless helper and one guarded call between
backward and step, plus an auditable transition/log field if the plan requires
it. Runtime risk is low-to-medium because reductions launch per tensor; memory
risk is negligible. The treatment requires no dependency, package, network,
remote operation, model-state addition, extra forward/backward pass, evaluator
change, extra validation, or seed change.

Only `train.py` may change. Keep seed 42, one H20, local CIFAR-10, batch 256,
FP32, alpha 0.2, the 65% mixup cutoff, continuous `5e-4` selective decay,
momentum 0.9, Nesterov, the accepted LR schedule and 0.002 floor, evaluation at
most once per epoch, and the normal final summary. Redirect the one scored run
to `run.log` and remove it before the next experiment.

## Evidence Base

- `.autoresearch/goals/maximize-cifar10-test-accuracy/01-definition.md`
- `.autoresearch/goals/maximize-cifar10-test-accuracy/04-results.tsv`
- `.autoresearch/goals/maximize-cifar10-test-accuracy/03-experiment-learnings.md`
- `.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/002/04-analysis.md`
- `.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/008/04-analysis.md`
- `.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/009/04-analysis.md`
- `.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/013/04-analysis.md`
- `.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/015/01-brainstorm.md`
- `train.py` (accepted model, parameter grouping, mixup gate, backward/step
  ordering, and time-based LR schedule)
