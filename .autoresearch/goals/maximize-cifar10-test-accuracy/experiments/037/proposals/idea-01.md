# Proposal: Exclude Only the Terminal Classifier Weight From Decay

## Summary

Preserve the accepted `a7c42dc` model and training trajectory except for one
optimizer-allocation decision: move the 1,280-element `fc.weight` tensor from
the `5e-4` matrix-decay group into the existing zero-decay group for the full
run. Keep `5e-4` decay on all other two-or-more-dimensional tensors, including
all convolution weights and both matrices in the accepted pooled residual
head. The classifier bias, BatchNorm affine tensors, and every other
one-dimensional parameter remain at zero decay as accepted.

This is not a general reduction of regularization and not a pooled-head tuning
experiment. It asks whether the terminal affine boundary should be free to fit
the already regularized nonlinear pooled representation while the backbone and
new residual head retain the continuous decay that has local support. The
production graph, initialization, parameter count, RNG streams, loss, schedule,
and amount of optimizer work remain effectively unchanged.

The testable hypothesis is that exempting only `fc.weight` improves class
boundary fit enough to raise fixed-seed `best_test_acc` from 94.48% to at least
94.58%, while retaining at least 127 realized data passes. Accepted
`final_test_acc=94.45%` and `final_test_loss=0.2456` are preregistered
corroboration references, not alternate success criteria.

## Evidence and Distinction From Prior Work

- EXP036 established the new accepted frontier at `a7c42dc`: its fixed
  bias-free `128 -> 64 -> 128` ReLU residual head at scale 0.1 scored 94.48%
  best, 94.45% final, and 0.2456 final loss over 130.304 passes. This proposal
  preserves that exact successful head, including its seed, initialization,
  forward formula, ordinary LR, and `5e-4` decay on both head matrices.
- The system diagnosis identifies generalization and boundary quality as the
  remaining limiter. The hard tail nearly interpolates training, while test
  loss remains 0.2456. A classifier-only allocation change acts directly at
  the final boundary without adding spatial compute or withholding any
  backbone gradient.
- EXP007 removed decay from **every matrix** only after the 65% transition and
  regressed to 93.74% with test loss 0.3244. That is strong evidence to retain
  continuous `5e-4` decay on the 983,472 convolution elements and 16,384
  pooled-head elements. It does not isolate the 1,280-element terminal
  classifier, and it changes the late optimization of the whole feature
  extractor rather than the full-run allocation of one tensor.
- The intervention is exceptionally narrow: `fc.weight` is about 0.128% of all
  1,003,482 trainable parameters. Before the optimizer step, accepted and
  candidate logits, losses, gradients, model state, and RNG state can be exact.
  Only the coupled L2 term on this named tensor differs during the step.
- The direction is uncertain. Classifier decay may beneficially control class
  vector norms and confidence; removing it can inflate logits, worsen loss, or
  change feature gradients without improving argmax boundaries. EXP007 and the
  current train/test gap make this a low-ceiling, adverse-prior experiment whose
  value is clean closure, not a claim of strong expected gain.

No network, remote source, or new literature search is used. The proposal is
grounded in `02-system-understanding.md`, `03-experiment-learnings.md`, the
EXP007 and EXP036 reports, and accepted source at `a7c42dc`.

## Exact Production Change

Keep the accepted two optimizer groups and derive them from named parameters:

```python
decay_params = [
    p
    for name, p in model.named_parameters()
    if p.requires_grad and p.ndim >= 2 and name != "fc.weight"
]
no_decay_params = [
    p
    for name, p in model.named_parameters()
    if p.requires_grad and (p.ndim < 2 or name == "fc.weight")
]
```

The existing SGD construction remains otherwise byte-for-byte: group zero has
`weight_decay=WEIGHT_DECAY`, group one has `weight_decay=0.0`, and both share
the accepted time-written LR, momentum 0.9, and Nesterov behavior. The expected
membership is exactly:

- decay group: 999,856 elements, every trainable tensor with `ndim >= 2`
  except `fc.weight`, including `pooled_head.0.weight` and
  `pooled_head.2.weight`;
- zero-decay group: 3,626 elements, exactly all accepted `ndim < 2` tensors
  plus the 1,280-element `fc.weight`;
- union: each of the 1,003,482 trainable elements exactly once, with no empty,
  duplicate, omitted, frozen, or newly created tensor.

Do not create a third group, change parameter registration order, alter group
LRs, introduce decoupled decay, schedule decay, log norms in production, or
special-case either pooled-head matrix. Keep the exact accepted `(2,2,3)` WRN,
pooled-head width 64/scale 0.1/seed 36036, batch 256, FP32, seed 42,
alpha-0.2 batch-shared mixup through 65%, worker-private N1/M5 RandAugment
through the first exhausted iterator after 65%, `0.2 -> 0.002` time-cosine LR,
loader, finite-loss guard, evaluator cadence, and 300-second counted budget.

## Preregistered Semantic Preflight

Use an ignored, evaluator-free verifier with an independent
`git show a7c42dc:train.py` oracle. It may not load test data, invoke `Eval`,
write `run.log`, or alter tracked production. Fail closed before timing on any
of the following checks:

1. Require the tracked production diff to modify only the two optimizer
   parameter comprehensions above. Require `prepare.py`, evaluator source,
   dependencies, model construction/forward, all constants, and every training
   or reporting statement to be exact relative to `a7c42dc`.
2. From cloned seed-42 CPU/CUDA states, construct accepted and candidate models
   and require every parameter and buffer byte, name, shape, dtype, device, and
   registration order to match, including the isolated pooled-head matrices.
   Require identical post-construction CPU/CUDA RNG and exactly 1,003,482
   trainable parameters.
3. Enumerate optimizer membership by identity and name. Require exactly two
   groups; group zero must be precisely all trainable `ndim >= 2` parameters
   except `fc.weight`, total 999,856 elements, at `5e-4`; group one must be
   precisely all `ndim < 2` parameters plus only `fc.weight`, total 3,626
   elements, at zero. Require each tensor once and exact accepted LR, momentum,
   Nesterov, dampening, maximize, differentiable, and foreach/fused defaults.
4. Explicitly require both pooled-head matrices and every convolution matrix
   to remain in group zero, while `fc.bias` remains in group one. Reject name
   prefixes, rank-wide exceptions, a third group, module-wide head exemption,
   scheduled switching, or any decay value other than accepted `5e-4`/zero.
5. On deterministic finite synthetic FP32 batches, start accepted and candidate
   models from byte-identical state and restore identical CPU/CUDA RNG before
   each arm. Before `optimizer.step`, require byte-identical inputs, optional
   batch-shared mixup lambda/permutation, logits, losses, every gradient, and
   RNG state in both the early-mixup and hard-label regimes.
6. Execute the first Nesterov step at an exact sampled production LR. Require
   all non-`fc.weight` parameter bytes and momentum buffers to remain
   byte-identical across arms. Require the candidate `fc.weight` update and
   momentum buffer to match an independent single-tensor zero-decay SGD oracle,
   the accepted version to match a `5e-4` oracle, and their finite difference
   to be nonzero and attributable only to the omitted coupled L2 term. Repeat
   a second deterministic step to cover nonempty momentum state.
7. Prove optimizer construction and updates consume no CPU/CUDA RNG. Require
   exact accepted mixup cutoff, RandAugment state transition, LR samples,
   model forward, loss branches, time accounting, once-per-epoch evaluation,
   and final summary source. Guard evaluator and test-data access throughout.

A semantic failure closes only this exact two-group implementation. It must not
be repaired by changing classifier decay to an intermediate value, exempting
the pooled head, changing group structure, or modifying the successful head.

## Preregistered H20 Timing Gate

Although skipping one decay multiply should be cost-neutral, the group
membership change must pass a direct disposable timing gate before consuming
the score. On one idle H20, compare independent accepted and candidate modules
from cloned state with production-equivalent complete update bodies separately
for early mixup and hard labels: H2D from fixed pinned batches, LR writes,
zeroing, optional Beta/permutation/interpolation, forward, exact loss,
finite-loss guard, backward, Nesterov SGD step, and synchronization. Exclude
the byte-identical loader and evaluator.

- Use at least 20 disposable warmups per arm and regime.
- Measure at least three counterbalanced windows of at least 50 steps per arm
  and regime, alternating accepted/candidate then candidate/accepted.
- Start paired arms from cloned model, BN-buffer, optimizer, input/target, and
  RNG state; print all synchronized per-step window values before assertions.
- Require every population CV at most 5%, all outputs/losses/gradients/states
  finite, and candidate peak allocation below 2,048 MiB.

For median update times `a_mix`, `a_hard`, `c_mix`, and `c_hard`, compute:

```text
retention = (0.65 / c_mix + 0.35 / c_hard) \
          / (0.65 / a_mix + 0.35 / a_hard)
projected_passes = 130.304 * retention
```

Score only if `retention >= 0.974644` and `projected_passes >= 127.0`. The
127-pass floor protects the approximate operating regime requested by the
updated system understanding while allowing only 2.54% slowdown from the
accepted realized exposure. The treatment removes work and should clear this
comfortably; a stable miss indicates a harness or grouping pathology and is
not rerun. No loader timing is needed because data source, transform, batch,
consumer shape, and worker policy are exact.

## Sole Scored Experiment and Verdict

If and only if all preflight gates pass, require one idle H20 and local CIFAR,
remove stale `run.log`, then execute exactly once:

```bash
timeout 600s uv run train.py > run.log 2>&1
```

Use seed 42. Do not rerun a valid completion, inspect test examples outside the
frozen evaluator, change evaluation cadence, tune decay from the result, or
modify the accepted pooled head. Audit exit zero, one finite summary, 300.0 to
300.1 counted seconds, less than 600 total seconds, 1,003,482 parameters, one
mixup transition, the worker-safe exhausted-iterator RandAugment transition,
and unique every-fifth-plus-final evaluations with no more than one per epoch.

- **Improvement:** `best_test_acc >= 94.58%`, the required +0.10 points over
  accepted 94.48%. This is the sole success condition.
- **Corroboration only:** `final_test_acc >= 94.45%` and
  `final_test_loss <= 0.2456`. Report both, but neither rescues a primary miss
  nor vetoes a valid primary success.
- **No improvement:** any structurally valid completed score below 94.58%,
  even if final accuracy or loss improves.
- **Exposure interpretation:** record `num_steps * 256 / 50000`. A valid
  completion below 127 passes remains the one nonrepeatable goal score and is
  classified by the primary metric; it makes the decay mechanism
  operationally inconclusive rather than authorizing a rerun.
- **Crash/invalid:** timeout, nonzero exit, missing/duplicate/malformed summary,
  nonfinite value, parameter-count/source/evaluator violation, repeated
  temporal transition, duplicate evaluation epoch, CUDA/OOM/worker error, or
  counted/wall-budget violation. Diagnose only; do not vary the intervention.

## Risks, Interpretation, and Closure

- The change affects only 1,280 parameters and adds no capacity, so its plausible
  top-1 effect may be smaller than the required ten-example margin.
- Coupled classifier decay may be part of why EXP036 improved test loss and
  retained a stable 0.03-point best/final gap. Removing it can enlarge class
  vector/logit norms and harm confidence even if training loss falls.
- Exact pre-step equality gives unusually strong attribution, but the first
  optimizer step intentionally diverges `fc.weight`; subsequent backbone and
  pooled-head gradients can therefore diverge through the changed logits. Do
  not misdescribe later common-parameter differences as direct decay effects.
- One fixed-seed score cannot estimate variance. Seed 42 is retained to avoid
  rerolling, and no result-conditioned repeat is permitted.

A valid score at or above 127 passes but below 94.58 closes the immediate
classifier-under-decay allocation family on the accepted pooled-head baseline:
do not follow with `fc.weight` decay `1e-4`, `2.5e-4`, or `3e-4`; delayed or
early-only classifier-decay schedules; decoupled classifier decay; classifier
LR/momentum compensation; exempting `fc.bias` (already zero decay); or moving
either pooled-head matrix out of decay. It also forbids width, scale,
activation, bias, initialization-seed, or other tuning of the successful pooled
head as a rescue. It does not close increased classifier decay, normalized-logit
geometry, or independently motivated representation changes. A pre-score
semantic/timing failure, or a completed score below 127 passes, closes only the
exact zero-decay allocation tested here. A valid success may be accepted as the
new baseline without any post-hoc decay or head adjustment.
