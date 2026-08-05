# Proposal: Classical Momentum at the Accepted 0.9 Operating Point

## Recommendation and Exact Treatment

This is a valid one-point optimizer experiment, but it has a lower prior than
representation-side candidates and should advance only if the idea review judges
that closing an optimizer choice bundled into EXP-001 is worth the sole fixed-seed
score.

Preserve the accepted `a7c42dc` learner byte-for-byte except for one optimizer
keyword:

```python
optimizer = optim.SGD(
    [
        {"params": decay_params, "weight_decay": WEIGHT_DECAY},
        {"params": no_decay_params, "weight_decay": 0.0},
    ],
    lr=MIN_LR,
    momentum=MOMENTUM,
-   nesterov=True,
+   nesterov=False,
)
```

`MOMENTUM` remains exactly `0.9`. Do not change peak/floor LR, warmup, cosine
schedule, weight decay, parameter groups, batch size, mixup, RandAugment, pooled
head, initialization, seed, or evaluation. Do not replace `SGD` with another
optimizer and do not add dampening. This tests classical PyTorch momentum at one
fully specified operating point; it is not a momentum or optimizer sweep.

## Mechanism and Evidence

The accepted optimizer has been Nesterov SGD since EXP-001, where it was bundled
with the WRN architecture, batch size, decay allocation, and time-aligned LR. The
37-run history contains no isolated classical-versus-Nesterov comparison. The new
accepted `a7c42dc` baseline adds a nonzero scale-0.1 pooled residual MLP and scores
94.48% at 130.304 data passes. This head changes optimization geometry without
changing the optimizer chosen for the earlier affine-head model, so the bundled
Nesterov choice is not directly validated for the current learner.

For PyTorch SGD with coupled decay, let `d_t` be the current gradient after adding
group weight decay and let `b_t = 0.9 b_(t-1) + d_t` (with the first buffer set to
`d_t`). The accepted Nesterov direction is
`d_t + 0.9 b_t`; the proposed classical direction is only `b_t`. Thus both arms
retain exactly the same velocity memory and LR schedule, while the candidate
removes Nesterov's extra current-gradient component. On the first step the
accepted direction is `1.9 d_0`, whereas the candidate direction is `d_0`.

The upside hypothesis is narrow: removing the extra current-gradient term may
reduce stepwise boundary jitter or overshoot in the newly nonlinear pooled
representation while accumulated momentum continues to carry the long-run
direction. It has effectively zero compute or memory cost and preserves the
accepted exposure regime. The counterargument is strong: LR 0.2 with 0.9
Nesterov was part of the successful optimization recipe from EXP-001 onward,
EXP-036's best and final accuracies differ by only 0.03 point, and classical
momentum materially reduces early first-step displacement without an LR
compensation. Therefore this should not outrank an orthogonal, better-evidenced
candidate merely because it is cheap.

No external search is used in this offline/local session. The optimizer equations
above come from the installed PyTorch `torch.optim.sgd._single_tensor_sgd`
implementation and must be verified against the live environment in preflight.

## Semantic Preflight

Use a disposable, evaluator-free experiment harness and an independent
`git show a7c42dc:train.py` accepted oracle. Require all of the following before
timing or scoring:

- The production diff is exactly the single `nesterov` Boolean change above;
  `prepare.py` is byte-identical and no other production file changes.
- Accepted and candidate model topology, named parameter/buffer bytes, parameter
  count `1,003,482`, post-construction CPU/CUDA RNG states, loader construction,
  transforms, constants, LR function, loss paths, phase cutoffs, and evaluator
  cadence are exact.
- Both optimizers contain the same parameter objects exactly once in the same
  order and the same two groups, with identical live values for LR, momentum
  `0.9`, dampening `0`, weight decay `[5e-4, 0.0]`, maximize, differentiable, and
  foreach/fused settings. Only `nesterov` may differ (`True` versus `False`).
- Inspect the installed PyTorch SGD implementation and fail closed unless its
  live update semantics match the equations preregistered above.
- From cloned full-model state and identical fixed CPU/CUDA RNG, execute one
  early batch-shared-mixup forward/backward in both arms without stepping.
  Require bitwise-identical input, targets, coefficient, permutation, logits,
  loss, every gradient, BatchNorm state, and post-backward RNG. Repeat for the
  hard-label path.
- For each path, step cloned arms from the same parameter values and empty
  momentum state. For every trainable parameter, independently form `d_0`
  including the correct group's coupled decay. Require both momentum buffers to
  equal `d_0`, candidate parameters to match `p_0 - lr*d_0`, and accepted
  parameters to match `p_0 - lr*1.9*d_0`, within `rtol=1e-6, atol=1e-7`; require
  all states finite and at least one update nonzero.
- Separately seed every parameter with a deterministic finite nonzero prior
  momentum buffer, then perform a hard-label step from common state. Require the
  two arms' post-step buffers to be bitwise equal to each other and to
  `0.9*b_prev + d_t`; require candidate and accepted updates to match
  `-lr*b_t` and `-lr*(d_t + 0.9*b_t)` within the same fixed tolerances. This
  prevents a first-step-only check from missing persistent-state semantics.
- Restoring the complete candidate model, optimizer, CPU/CUDA RNG, mode, and
  fixed-input snapshot must reproduce coefficient/permutation when applicable,
  loss, gradients, buffers, parameters, and terminal RNG bitwise for both an
  early and a hard step.

Abort on any failed check rather than repairing it by changing optimizer flags,
tolerances, LR, momentum, model, or data behavior. No loader timing is required:
the production data source and all CPU transforms are source-identical.

## Throughput and Exposure Gate

Although classical momentum removes rather than adds one per-parameter axpy,
measure the exact complete production step body to protect the new 130-pass
operating regime. On one idle NVIDIA H20, construct accepted and candidate models
from identical snapshots and time early-mixup and hard-label steps separately.
Each window must include pinned-host transfer, LR/group assignment, zeroing,
mixup and permutation when active, full FP32 forward/loss/backward, the respective
SGD step, and final CUDA synchronization.

Use at least 20 untimed warmups per arm/regime followed by three counterbalanced
windows of at least 50 measured steps, reversing arm order in the middle window.
Use private restored RNG/state per arm so timing order cannot change the work.
Print all raw window means, medians, population CVs, parameter/state audits, and
peak VRAM before asserting. Require every arm/regime CV at most 5% and finite
outputs/state.

With medians in seconds per step, compute the time-fraction-weighted throughput
retention exactly as

```text
retention = (0.65 / candidate_early + 0.35 / candidate_hard) /
            (0.65 / accepted_early  + 0.35 / accepted_hard)
projected_passes = 130.304 * retention
```

Require `retention >= 0.9746439096` and `projected_passes >= 127.000` before the
sole score. This is the current system-understanding floor for preserving a
roughly normal-exposure optimizer comparison. A stable gate miss closes only this
exact implementation without scoring; do not rerun a stable miss or relax the
floor. An unstable CV may be repeated once only after confirming the GPU was not
idle, and the original failed output remains recorded.

## Sole Score, Verdict, and Closure

After all gates pass, remove stale `run.log`, confirm exactly one local NVIDIA H20,
and run the fixed seed exactly once:

```bash
timeout 600s uv run train.py > run.log 2>&1
```

Require exit zero, finite summary, `training_seconds` in `[300.0, 300.1]`,
`total_seconds < 600`, parameter count `1,003,482`, at most one evaluation per
epoch, the accepted first-exhausted-epoch RandAugment transition, and no frozen
file/evaluator changes. Record realized steps and passes; a completed run below
127 passes is still a valid fixed-seed result and must not be rerun, but exposure
loss weakens optimizer attribution.

The sole success condition is `best_test_acc >= 94.58%`, exactly 0.10 point above
the moving 94.48% baseline. `final_test_acc >= 94.45%` and
`final_test_loss <= 0.2456` are directional corroboration only and cannot rescue
or veto the primary verdict.

A valid miss closes classical momentum with coefficient 0.9 under the exact
accepted LR/decay/model/data schedule. Do not rescue it with momentum 0.85, 0.95,
or 0.99; LR scaling; dampening; late-only or head-only Nesterov; separate pooled-
head optimizer groups; head scale/width changes; decay changes; schedule/cutoff
changes; another seed; checkpoint selection; or a score rerun. Those are distinct
multi-variable tuning programs without support from this one-point result. A
pre-score semantic or feasibility failure closes only this exact implementation.

## Falsifiable Hypothesis

Changing only PyTorch SGD from Nesterov to classical momentum at coefficient 0.9
will preserve at least 127 projected data passes and raise fixed-seed CIFAR-10
`best_test_acc` from 94.48% to at least 94.58% by reducing the extra current-
gradient contribution at the new pooled nonlinear decision head. Failure at
normal exposure falsifies this optimizer choice and should leave the accepted
Nesterov configuration unchanged.
