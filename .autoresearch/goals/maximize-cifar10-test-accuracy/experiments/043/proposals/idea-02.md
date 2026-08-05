# Proposal: One-Time Full Nesterov Momentum Reset at the Hard-Label Boundary

## Recommendation

This is a technically clean but low-upside candidate. If EXP043 must spend a
score on an optimizer-state intervention, clear every live SGD
`momentum_buffer` exactly once inside the existing one-way mixup transition,
before the first hard-label forward/backward/update. Preserve the accepted
global cosine, `0.002` floor, matrix-only `5e-4` coupled decay, momentum `0.9`,
Nesterov setting, data path, model, and all RNG streams.

The treatment asks whether velocity accumulated under mixed images and paired
soft targets is briefly harmful after the objective changes to hard-label CE.
It does not restart the learning rate, change the momentum coefficient, reset
parameters or BatchNorm state, or force RandAugment off early. The hypothesis
is easy to attribute and almost free in compute, but its prior is low: the
inherited component naturally falls below 1% in about 44 updates, less than
0.5% of the accepted hard-label tail, and there is no measured transition
instability in the accepted run.

This candidate should rank below an orthogonal mechanism with evidence of a
persistent generalization bottleneck. Its most likely result is a score near
the accepted stochastic trajectory, not the required `+0.10` point gain.

## Exact Production Intervention

Keep `learning_rate()` byte-identical to accepted `a7c42dc`. Reuse the existing
`mixup_enabled` one-way state. In the branch that already detects the first
pre-step `progress >= MIXUP_END_FRACTION`, remove each parameter's momentum
buffer before `optimizer.zero_grad`, the hard-label forward, `loss.backward`,
and `optimizer.step`:

```python
if mixup_enabled and not use_mixup:
    momentum_buffers_reset = 0
    for state in optimizer.state.values():
        if state.pop("momentum_buffer", None) is not None:
            momentum_buffers_reset += 1
    mixup_enabled = False
    print(
        f"\nMixup disabled at ep {epoch} step {step} | "
        f"training_seconds={total_training_time:.1f} "
        f"({100 * progress:.1f}%) | lr={lr:.4f} | "
        f"momentum_buffers_reset={momentum_buffers_reset}"
    )
```

The accepted model has 52 trainable parameter tensors and all participate in
every training step, so a qualified transition must report exactly 52 removed
buffers. The semantic verifier, rather than a new production constant, should
establish that count from `model.parameters()` and prove that every optimizer
state entry held exactly one live `momentum_buffer` immediately beforehand.
The printed count is execution evidence; it must not affect training control.

Deleting the buffer key is the faithful PyTorch "fresh momentum" operation.
Zeroing every buffer would happen to produce the same next update under this
optimizer's `dampening=0`, but it is not the requested absent-buffer state and
must not be substituted. Do not reconstruct the optimizer, call
`optimizer.state.clear()`, replace buffers with zero tensors, set group
momentum to zero, or reset only selected layers. Those operations either erase
state less selectively or define different interventions.

The location is exact. Inputs for the transition batch have already been
delivered by the accepted loader and may still contain worker-side
RandAugment. The reset aligns with the first hard-label loss, not with the later
exhausted-iterator RandAugment transition. Do not move it to an epoch boundary,
the later augmentation cutoff, exactly `195.0` wall seconds after a step, or
the first clean crop/flip-only batch.

Everything else remains accepted: the `(2,2,3)` WRN at widths
`[32,64,128]`, bias-free scale-0.1 `128 -> 64 -> 128` pooled residual head and
seed 36036, ordinary affine classifier, FP32, batch 256, alpha-0.2
batch-shared mixup through 65%, worker-safe N1/M5 RandAugment through its
exhausted epoch, crop/flip, seed 42, persistent loader, evaluator cadence,
300-second counted budget, and 1,003,482 parameters.

## Exact PyTorch SGD/Nesterov Semantics

For parameter `theta`, raw loss gradient `grad_t`, group weight decay
`lambda`, learning rate `eta_t`, momentum `mu=0.9`, and prior buffer
`b_(t-1)`, define the coupled gradient

```text
d_t = grad_t + lambda * theta_(t-1)
```

where `lambda=5e-4` for rank-at-least-two tensors and zero otherwise. With
PyTorch SGD, `dampening=0`, and an existing buffer, the accepted update is

```text
b_t                 = mu * b_(t-1) + d_t
nesterov_direction  = d_t + mu * b_t
theta_t             = theta_(t-1) - eta_t * nesterov_direction
```

After deleting the buffer immediately before the first hard-label update,
PyTorch initializes the fresh buffer directly from the current coupled
gradient:

```text
b_t(reset)                = d_t
nesterov_direction(reset) = d_t + mu * d_t = (1 + mu) * d_t
theta_t(reset)            = theta_(t-1) - eta_t * (1 + mu) * d_t
```

Starting from identical parameters, hard batch, gradient, and RNG, the reset
therefore removes exactly

```text
nesterov_direction(accepted) - nesterov_direction(reset)
    = mu^2 * b_(t-1)
    = 0.81 * b_(t-1)
```

from the first hard update direction. Equivalently,
`theta_t(reset) - theta_t(accepted) = eta_t * mu^2 * b_(t-1)` before later
trajectory divergence. This is a buffer-state intervention, not a plain-SGD
step: the current hard gradient still receives the fresh-state Nesterov
multiplier `1 + mu = 1.9`, and coupled decay remains active inside `d_t`.

For a frozen-gradient counterfactual, after `j` post-boundary updates the
difference in buffers caused only by the inherited pre-boundary state is
`-mu^j * b_(t-1)`, and its contribution to the `j`th Nesterov direction is
`-mu^(j+1) * b_(t-1)`. Real candidate and accepted gradients diverge after the
first parameter update, so these formulas isolate inherited-memory decay; they
are not a claim that the complete trajectories differ only by that term.

At `mu=0.9`, `0.9^44 = 0.0096977`, so inherited buffer memory is below 1%
within 44 recurrences. EXP036 observed 9,114 hard-tail updates
(`25,450 - 16,336`) and a 44-step gap between its mixup and RandAugment
transitions. Thus the direct transient occupies about `44 / 9,114 = 0.483%`
of the hard-label updates and expires around the later augmentation boundary in
that accepted score. This brevity is the proposal's central weakness.

## Mechanistic Rationale and Local Evidence

The positive argument is narrow. Before 65%, the accepted optimizer integrates
gradients from convexly mixed images and paired soft targets. At 65%, the loss
switches discontinuously to hard-label CE while the existing velocity passes
through unchanged. A full reset prevents the first hard updates from inheriting
a direction estimated for the prior stochastic objective. In a nonconvex
trajectory, even a short early-tail displacement can select a different basin,
so a 44-step direct memory does not mathematically imply zero final effect.

Several local facts keep expectations modest:

- EXP036 already achieved 94.48% best / 94.45% final and 0.2456 loss without a
  visible transition fault. Its best occurred near the end, not immediately
  after the 65% switch.
- EXP039 increased hard-tail LR area by 39.46% yet regressed to 93.98% and
  0.2661 loss at 131.215 passes. The accepted global cosine and nonzero floor
  are locally protected, so this candidate preserves both exactly.
- EXP041's always-on 90/10 direct-path auxiliary CE produced strongly aligned
  sampled gradients but still regressed to 94.26% and 0.2529 loss at 128.538
  passes. It supports preserving the sole refined-path CE exactly; it does not
  supply evidence that accepted mixed-objective velocity is antagonistic.
- The accepted tail nearly interpolates training while test loss remains
  0.2456. That diagnosis points to persistent generalization/boundary quality,
  whereas a one-time reset directly changes only a brief optimization
  transient.
- Mixup does not necessarily produce an antagonistic velocity. Mixed-target
  gradients still train the same classes, model, classifier, and pooled head;
  no local gradient measurement shows that the inherited buffer opposes the
  first hard gradients.
- RandAugment remains active for the already-prefetched transition batch and
  until the current iterator exhausts. Resetting at the mixup boundary therefore
  does not create a fully clean second phase.

The expected throughput impact is effectively zero and the expected accuracy
effect is centered close to zero. A roughly few-hundredths-point movement is
more plausible than the required +0.10, while basin sensitivity permits a
larger positive or negative fixed-seed result. That uncertainty justifies one
falsifiable run, not a reset schedule sweep.

## Semantic and Transition Preflight

Use an ignored evaluator-free harness with an independently compiled exact
`git show a7c42dc:train.py` oracle. Before timing or scoring, require:

1. The production diff is confined to removal/counting of momentum buffers in
   the existing mixup transition and the transition log suffix. `prepare.py`,
   evaluator behavior, model, losses, schedule, data, constants, RNG, cadence,
   and summary remain accepted.
2. Candidate and accepted construction produce byte-identical parameters,
   buffers, optimizer groups/options, post-construction CPU/CUDA RNG states,
   and exactly 1,003,482 trainable parameters across 52 tensors.
3. After deterministic pre-boundary warmup updates, prove every trainable
   tensor has exactly one finite momentum buffer of matching shape/device/dtype
   and no other per-parameter SGD state. Capture all parameters, buffers, BN
   state, inputs, and CPU/CUDA RNG before the transition.
4. Probe immediately below 65% and prove the candidate is bitwise accepted:
   no buffer is removed, `mixup_enabled` remains true, mixed inputs/targets,
   logits, loss, gradients, updates, buffers, and RNG all match.
5. Probe the first pre-step progress at or above 65%. Require `use_mixup=False`,
   remove exactly 52 buffer keys exactly once before the hard forward, and
   prove parameter bytes, BN buffers, gradients, LR, weight-decay groups,
   inputs, targets, and CPU/CUDA RNG are unchanged at removal.
6. For every tensor, independently reproduce both accepted and reset first
   hard updates using the formulas above. Require the accepted direction minus
   reset direction to equal `mu^2*b_prev` and require production parameters and
   newly created buffers to match the reset oracle within preregistered FP32
   tolerances. Check rank-based coupled decay explicitly.
7. From cloned synthetic float64 states, verify the recurrence for at least 50
   post-boundary updates under a shared prescribed gradient sequence. Require
   inherited-buffer contribution ratios `mu^j`, including
   `mu^44=0.0096977`, and analytic/autograd-free parameter recurrences.
8. On the second and later hard steps, prove the one-way flag prevents another
   removal. Buffer identities may be newly allocated, but keys, values, and
   updates must persist normally. Restoring complete state/RNG must replay the
   transition exactly.
9. Prove RandAugment retains accepted exhausted-iterator semantics and is not
   disabled by the reset. Audit unique every-fifth plus final evaluation and
   the unchanged finite-loss/timeout/summary controls.

Print pre-reset buffer count/norm statistics, first-step accepted/reset update
norms and cosines, `mu^2*b_prev` oracle errors, per-step inherited-memory
ratios, and replay errors before assertions. These are implementation evidence,
not tuning signals. Do not use observed buffer-gradient alignment to choose a
subset, scale, timing, or combined restart.

## Timing and Exposure Gate

Steady early and hard steps are source- and operation-identical after the one
reset, but measure rather than assume. On one idle H20, run at least 20 warmups
and two complete counterbalanced `A/C/C/A` cycles of windows with at least 50
production-equivalent steps per arm and regime. Include H2D, LR writes,
zeroing, mixup where active, full forward/loss/backward, coupled Nesterov,
finite guard, and synchronization. Require all population CVs at most 5%.

The recurring-step gate uses the accepted 130.304-pass reference:

```text
retention =
    (0.65 / candidate_early_ms + 0.35 / candidate_hard_ms) /
    (0.65 / accepted_early_ms  + 0.35 / accepted_hard_ms)
projected_passes = 130.304 * retention
```

Require `retention >= 127 / 130.304 = 0.9746439096` and projected passes at
least 127. Candidate peak allocation must remain below 2,048 MiB.

Also benchmark the one-time transition separately with at least 12
counterbalanced cloned accepted/reset pairs. Each pair must start with all 52
preseeded buffers and include transition control, buffer deletion where
applicable, one complete hard update that recreates fresh buffers, and
synchronization. Print raw paired deltas, median, CV, and maximum. Conservatively
use the maximum stable positive transition penalty `p_s` when reporting
`adjusted_projected_passes = projected_passes * (300 - p_s) / 300`; it must
still be at least 127 passes. This isolated benchmark prevents the only changed
operation from being diluted out of steady 50-step windows.

A stable timing failure closes the exact full reset. Do not replace deletion
with in-place zeroing, reduce the reset subset, relax the floor, or rerun until
a favorable timing sample appears.

## Sole Score and Decision Contract

After all gates pass, reconfirm baseline 94.48% at accepted `a7c42dc`, threshold
94.58%, exact `train.py`-only production scope, frozen `prepare.py`, one idle
NVIDIA H20, local CIFAR-10, and no stale `run.log`. Execute exactly once:

```bash
timeout 600s uv run train.py > run.log 2>&1
```

Require exit zero, one finite summary, 300.0-300.1 counted seconds, total wall
below 600 seconds, 1,003,482 parameters, exactly one mixup transition reporting
52 buffers removed, a later exhausted-iterator RandAugment transition, unique
every-fifth-epoch evaluations plus the final partial epoch, and no traceback,
OOM, worker, evaluator, or non-finite failure. Record realized passes as
`num_steps * 256 / 50000`, transition step/time/LR and RandAugment lag, peak
VRAM, best/final accuracy, loss, and best-final gap.

Primary success is only `best_test_acc >= 94.58%`, exactly 0.10 points above
the accepted 94.48%. Preregister `final_test_acc >= 94.45%` and
`final_test_loss <= 0.2456` as non-decisive corroboration. They neither rescue
a primary miss nor veto a valid primary success. At least 127 realized passes
is required for normal-exposure attribution. A lower-exposure completed score
still consumes the sole run and is operationally inconclusive; it may not be
rerun.

## Falsifiable Hypothesis and Closure

If momentum inherited from mixed-image, paired-target training impedes the
initial hard-label trajectory enough to affect the final basin, then deleting
all 52 momentum buffers exactly before the first hard-label update will retain
at least 127 projected and realized passes and raise fixed-seed
`best_test_acc` from 94.48% to at least 94.58%, without changing accepted final
loss corroboration.

A valid normal-exposure score below 94.58% rejects this exact one-time full
buffer reset at the mixup boundary. Close immediate result-conditioned rescues:
in-place zeroing, partial/weighted resets, classifier-only or head-only resets,
moving the reset to the exhausted RandAugment boundary, resetting twice,
briefly disabling momentum, and tuning a buffer scale. EXP039 already rejected
the isolated LR rephase; combining that failed schedule with a failed reset is
an unsupported interaction rescue and should also be declined absent a new
diagnosis. The result does not reject fundamentally different optimizers,
continuous momentum schedules justified prospectively, or state mechanisms
with a persistent rather than 44-step effect.
