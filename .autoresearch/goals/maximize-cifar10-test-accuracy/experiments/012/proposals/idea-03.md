# Zero-Initialized Residual Endpoints, Adapted Safely for Pre-Activation WRN

## Thesis

Start every residual correction at exactly zero while preserving the accepted
WRN-16-2 topology, optimizer, schedule, alpha-0.2 mixup through 65%, hard-label
tail, data path, evaluation cadence, and seed. This should make the initial
network follow only its skip/projection paths and require each residual branch
to earn its contribution gradually. The intended mechanism is improved early
conditioning and implicit regularization, not added capacity or throughput.

The common instruction to zero the last BatchNorm scale cannot be copied
literally into this repository. `PreActBlock.bn2` is followed by `ReLU` and then
`conv2`; it is not a post-convolution endpoint BatchNorm. Setting
`bn2.weight = 0` while its bias is zero makes the ReLU input exactly zero.
PyTorch's ReLU derivative at zero is zero, so `bn2.weight`, `conv1`, and the
branch activations receive no gradient and the branch remains permanently
dead. The faithful trainable equivalent for this exact pre-activation block is
to zero the final convolution weight, `conv2.weight`, after normal model
initialization.

## Exact Intervention

After `self.apply(self._weights_init)` has initialized the whole model exactly
as accepted, iterate over the model's modules and apply `init.zeros_` to
`conv2.weight` for each of the six `PreActBlock` instances. Do not skip the
normal Kaiming draw for those tensors: initialize first and overwrite second.
The overwrite consumes no random values, so model construction advances the
CPU RNG by exactly the accepted amount and leaves subsequent DataLoader
shuffling and CUDA mixup draws on the accepted fixed-seed stream.

No BatchNorm scale changes:

- all six `PreActBlock.bn1.weight` tensors remain initialized to one;
- all six `PreActBlock.bn2.weight` tensors remain initialized to one; and
- the final `WideResNet.bn.weight` remains initialized to one.

All BatchNorm biases remain zero, and their running-statistic behavior is
unchanged. There is no new BatchNorm after `conv2`, no residual multiplier, and
no special per-stage policy. The only changed values are the six existing
bias-free `conv2.weight` tensors at initialization.

The parameter count remains exactly 691,674 and forward/backward operations,
tensor shapes, state-dict keys, optimizer groups, and MAC count remain
unchanged. Weight decay continues to apply to every `conv2.weight`; because the
weights begin at zero, the first update is driven only by the data gradient.

## Shortcut and Projection Semantics

Shortcut behavior must remain byte-for-byte structurally unchanged. At
initialization, the residual output of every block is exactly zero, so:

- an identity block returns its raw input `x` exactly; and
- a channel-changing or strided block returns the existing
  `shortcut(F.relu(bn1(x)))`, not raw `x` and not a separately normalized path.

The three learned projections at the first block of each stage keep their
accepted Kaiming initialization and train from the first backward pass. There
is no new projection, identity substitution, detach, or scaling. Consequently,
the initial network is still input-dependent through the stem, transition
projections, final BatchNorm, and classifier; only residual corrections start
closed.

## Mechanism

The accepted model already reaches near-zero late training loss, while EXP-010
and EXP-011 added low-resolution capacity for only +0.04 and +0.08 accuracy
points and the deeper model worsened test loss to 0.2782. The remaining limiter
looks more like generalization and confidence than raw fitting ability.

Zero residual endpoints make the initial stack close to its skip-path function
and prevent six independently random final convolutions from perturbing every
stage at once. On the first backward pass, each zero `conv2` receives a
nonzero gradient because its input activation is nonzero. Gradients into
`conv1` and the branch BatchNorms are zero on that one pass because the
backpropagated signal is multiplied by the zero `conv2` weight. After the first
SGD update, `conv2` is nonzero and the upstream branch parameters begin
learning. Thus this is a one-update staged opening, not a long warmup or a
permanently gated model. Over roughly 27,000 accepted-budget steps, the
one-update delay is negligible, while the changed early geometry may bias SGD
toward smoother residual corrections and reduce overconfident specialization.

This is an initialization-only experiment. Do not combine it with the
EXP-011 extra block, a bottleneck, EMA, label smoothing, altered learning rate,
or a nonzero residual scale.

## Required Semantic Tests

Run evaluator-free local tests before timing or scoring. Replace `prepare.Eval`
with a fail-closed stub before importing `train.py`, so no test-set information
is constructed or inspected.

1. Construct accepted and candidate WRN-16-2 models from identical saved RNG
   states. Confirm identical topology, 691,674 parameters, state-dict keys,
   shortcut locations/strides, and all non-`conv2` initialized tensors. Confirm
   that the construction leaves the RNG state identical in both cases.
2. Confirm there are exactly six `PreActBlock` modules; every candidate
   `conv2.weight` is exactly zero; every block `bn1.weight` and `bn2.weight` and
   the final BN scale are exactly one; and `conv1`, projections, and classifier
   weights are finite and nonzero.
3. For fixed finite inputs in training mode, expose each block's residual path
   and assert it is exactly zero. Assert each identity block output equals `x`,
   while each transition block output equals its existing projected
   preactivation shortcut. Repeat in evaluation mode to rule out an accidental
   running-statistic dependency.
4. On a fixed minibatch and cross-entropy loss, run the first backward pass.
   Require finite, nonzero gradients on all six `conv2.weight` tensors and on
   the classifier and projection shortcuts. Confirm upstream residual-branch
   gradients are zero on this first pass, as predicted by the zero endpoint.
5. Take one accepted SGD update, verify all six `conv2.weight` tensors have
   become finite and nonzero, then run a second forward/backward on a fixed
   minibatch. Require finite, nonzero gradients for every residual `conv1` and
   `bn2.weight`, proving all branches have opened. Require finite logits and
   loss throughout.
6. As a diagnostic guard only, demonstrate on an isolated copy that literal
   zero `bn2.weight` plus zero bias produces zero branch output and zero
   `bn2.weight` gradient. This rejected variant must never enter `train.py`.

Any failed count, equality, gradient, or RNG-state assertion rejects the
implementation before a scored run; do not patch around it with a residual
multiplier or positive BN bias.

## Throughput Preflight and Gates

The candidate executes exactly the same kernels and parameter shapes as the
accepted model, so expected steady-state throughput retention is approximately
99-101%, with the same roughly 141.9-pass exposure and approximately 1.1 GiB
peak allocation. Initialization adds only six one-time zero fills before the
counted training timer.

Use a warm, order-balanced, evaluator-free production-path benchmark of
accepted initialization versus candidate initialization on the H20. Exercise
both the alpha-0.2 mixup path and the hard-label path separately with fixed
resident batches, matched RNG states, the real optimizer grouping, backward,
optimizer step, synchronization, and the candidate's actual initialization.
Use multiple interleaved repetitions and report per-regime medians and
coefficients of variation, then weight the regimes 65/35 to match the scored
run.

Proceed to one scored run only if all timing CVs are at most 5%, aggregate
candidate throughput is at least 97% of the matched accepted path, and projected
exposure from the accepted 141.9 passes is at least 135 passes. A lower result
would indicate a benchmark or implementation anomaly because the computation
graph is unchanged; reject it before scoring rather than relaxing the gate.
The preflight must not evaluate accuracy or load the evaluator/test set.

## Falsifiable Hypothesis and Decision Rule

With one fixed-seed H20 run and the unchanged 300-second counted budget, safe
zero initialization of all six residual endpoints will achieve at least
**94.17% `best_test_acc`**, the required +0.10 percentage-point improvement over
94.07%, while retaining at least 135 data-equivalent passes and completing in
under 600 seconds total.

Accept only if all integrity constraints pass and `best_test_acc >= 94.17%`.
A valid run below 94.17% is `no-improvement`, even if test loss improves. A
supporting mechanistic signature is normal exposure with lower final test loss
than the accepted 0.2432, consistent with less overconfident specialization;
it is informative but cannot override the primary threshold. A normal-exposure
regression would falsify the claim that identity-biased initialization helps
this already shallow WRN under the accepted schedule.

If realized exposure falls below 135 passes despite the unchanged graph,
classify the implementation/timing anomaly before drawing a clean mechanistic
conclusion, but do not rerun or substitute literal BN-scale zeroing. A stable
negative closes exactly the all-six-`conv2` zero initialization treatment; it
does not test small nonzero endpoint scales or selective-stage initialization,
and neither is an in-experiment fallback.

## Fixed-Budget Risks

- The accepted LR and warmup were tuned around random residual endpoints;
  identity-biased initialization may make early updates too correlated or
  temporarily reduce useful representation diversity.
- Although branches open after one update, their internal `conv1` and BN
  parameters do not learn on the first update. The delay is tiny in step count,
  but the first high-LR trajectory can still land in a different basin.
- Starting all final branch convolutions at zero removes their random feature
  mixing. Data-dependent gradients should break this immediately, but the
  semantic test must verify every block rather than assuming it.
- This technique may improve optimization stability without improving the
  already strong generalization margin; expected gains may be smaller than the
  required 0.10 point.
- Literal zeroing of `bn2` is a fatal implementation trap in this architecture,
  not an alternate interpretation worth scoring.

## Constraint Compliance and Effort

Only `train.py` changes, with no package installation, network access, remote
operation, evaluator change, seed reroll, extra validation, or additional
training pass. Keep seed 42, one H20, the existing evaluation cadence, exact
300-second counted training budget, and the external 600-second timeout. Run
the scored command once with output redirected to `run.log`, analyze its final
summary, and remove the log before the next experiment.

Implementation effort is low and experimental risk is medium. The code change
is a deterministic post-initialization overwrite plus an auditable log field;
the substantive uncertainty is whether a generalization-oriented initialization
can improve accuracy within this fixed schedule.

## Evidence Base

- `.autoresearch/goals/maximize-cifar10-test-accuracy/01-definition.md`
- `.autoresearch/goals/maximize-cifar10-test-accuracy/03-experiment-learnings.md`
- `.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/002/04-analysis.md`
- `.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/011/04-analysis.md`
- `.autoresearch/goals/maximize-cifar10-test-accuracy/knowledge/papers/wide-residual-networks.md`
- `train.py` (`PreActBlock.forward` and `WideResNet._weights_init`)
