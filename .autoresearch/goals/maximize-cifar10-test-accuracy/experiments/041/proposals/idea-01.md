# Proposal: Learn Only the Accepted Pooled-Residual Gain

## Recommendation

Test one direct, unconstrained scalar gain only if EXP041 needs a low-cost
candidate. Replace the accepted Python constant in
`out + 0.1 * pooled_head(out)` with one scalar `nn.Parameter` initialized to
the exactly representable FP32 value produced by `torch.tensor(0.1)`. Leave the
accepted pooled branch, classifier, backbone, recipe, and all existing
optimizer options unchanged. The new scalar should enter the existing
zero-decay parameter group because it is dimensionless and rank zero; it should
receive the accepted time-varying LR, momentum 0.9, and Nesterov update.

This is a clean one-score question but a weakly supported improvement
hypothesis. EXP036 established that the fixed scale-0.1 branch is useful, not
that 0.1 is miscalibrated or that its useful amplitude should change during
training. Its own accepted closure cautioned against adjacent learned-scale
rescue/tuning. Learnability is nevertheless distinct from a fixed-value sweep:
it lets the training objective allocate the direct and nonlinear pooled paths
over time while preserving the accepted initial function. The treatment costs
one parameter and no spatial work. It deserves at most low-to-medium priority,
and should not outrank an orthogonal candidate with a diagnosed mechanism.

## Exact Production Treatment

Keep `POOLED_HEAD_SCALE = 0.1` as the sole initialization constant. After the
accepted isolated construction and initialization of `self.pooled_head`, append
one parameter without consuming RNG:

```python
with torch.random.fork_rng(devices=[]):
    torch.random.default_generator.manual_seed(POOLED_HEAD_INIT_SEED)
    self.pooled_head = nn.Sequential(
        nn.Linear(widths[2], POOLED_HEAD_WIDTH, bias=False),
        nn.ReLU(),
        nn.Linear(POOLED_HEAD_WIDTH, widths[2], bias=False),
    )
    init.kaiming_normal_(self.pooled_head[0].weight)
    init.kaiming_normal_(self.pooled_head[2].weight)
self.pooled_head_scale = nn.Parameter(torch.tensor(POOLED_HEAD_SCALE))
```

Then change only the gain operand in the accepted forward:

```python
out = out + self.pooled_head_scale * self.pooled_head(out)
```

The placement after the restoring fork is deliberate. `torch.tensor(0.1)`
does not draw RNG, and construction after the accepted branch leaves branch
initialization and the protected global CPU/CUDA RNG states unchanged. The
parameter is created on CPU in default FP32 and moves with `model.to(device)`.
It is registered after `pooled_head`, so every accepted state entry and
parameter remains byte-identical and in accepted order; only the final new
state key `pooled_head_scale` is appended. Total trainable parameters become
1,003,483.

Do not use `torch.empty`, an initializer, or any random draw for the scalar.
Do not put it inside the isolated branch construction before the two Kaiming
draws. Do not reinitialize the accepted branch, change seed 36036, or change
any existing parameter bytes.

The current generic optimizer comprehensions provide the intended allocation
without special cases:

```text
decay group:    all trainable tensors with ndim >= 2, unchanged, 5e-4 decay
no-decay group: all trainable tensors with ndim < 2 plus the new scalar, 0 decay
```

The scalar is dimensionless, so applying matrix shrinkage would silently test
gain decay as a second intervention. A dedicated LR or optimizer group would
also introduce an arbitrary new hyperparameter. Keep the accepted global LR,
momentum, dampening default, Nesterov setting, and parameter ordering. Add no
positivity transform, log parameterization, clamp, sigmoid, softplus, gain
schedule, gradient clipping, freeze window, or separate regularizer. Negative
values must remain mathematically possible; preventing them would be a
different hypothesis.

Printing the final scalar once in the final summary is acceptable and useful
for interpretation, but it must not be inspected during the run or used for
control flow. Do not add per-step synchronization or logging.

## Exact Initialization and Optimizer Semantics

At construction, the candidate function must equal the accepted function on
the same state and input. Python float `0.1` in the accepted FP32 tensor
multiplication and the new FP32 scalar both supply the same rounded FP32 gain.
Require bitwise-equal branch outputs, residual features, and logits in the
semantic verifier where the kernel path permits it; a fixed, preregistered
near-machine tolerance may be used only if scalar-tensor dispatch changes
operation ordering despite identical operands.

Let `s` be the scalar, `h` the accepted pooled branch output, `z` the pooled
direct feature, and `L` the accepted mixup or hard-label CE:

```text
u = z + s h
g_s = dL/ds = sum over batch and channels of (dL/du) * h
```

There is no decay term for `s`. With current PyTorch SGD semantics, accepted
LR `eta`, momentum `mu=0.9`, and Nesterov enabled:

```text
fresh state:       b_1 = g_s
                   s_1 = s_0 - eta * (g_s + mu * b_1)

existing buffer b: b_next = mu * b + g_s
                   s_next = s - eta * (g_s + mu * b_next)
```

The first formula's effective multiplier is therefore `1 + mu = 1.9`, not a
plain-SGD step. The verifier must use this exact oracle. At the initial gain,
all accepted parameter gradients should match the fixed-scale model because
the forward function and derivative with respect to every existing tensor are
the same; only the candidate has `g_s` and its new momentum state. After the
first optimizer step, later gradients may legitimately diverge because `s`
has learned.

No sign or magnitude expectation for `g_s` is preregistered. A finite,
nonzero gradient on deterministic early-mixup and hard-label fixtures is a
learnability check, not evidence for the idea. Print its value, the independent
dot-product oracle, the predicted first update, the resulting gain, and the
initial branch/direct feature-norm ratio before assertions. Do not use any of
those diagnostics to pick an LR, reparameterization, or alternative start.

## Evidence and Expected Mechanism

- EXP036 added the exact bias-free `128 -> 64 -> 128` ReLU branch at scale 0.1
  and improved best accuracy from 94.32% to 94.48%, final accuracy from 94.22%
  to 94.45%, and loss from 0.2523 to 0.2456 at 130.304 passes. This proves the
  nonlinear pooled correction is useful at one fixed amplitude.
- The initial scale-0.1 branch/direct feature-norm ratio was 0.120864 and its
  logit perturbation RMS was 0.069719. The branch is a correction rather than
  a dominant replacement path, making scalar allocation numerically plausible.
- EXP037-040 all retained normal exposure but worsened the accepted head by
  changing classifier decay, tail LR, or class-vector radii. They support
  preserving those mechanisms exactly; they provide no positive evidence for
  a movable head gain.
- System attribution places only about 1.4% of forward time in the entire head,
  while backpropagation is the main counted cost. One scalar multiply-gradient
  and SGD state should be negligible and should preserve at least 127 passes.
- The accepted model nearly interpolates the hard tail but still has 0.2456
  test loss. A learned gain could improve boundary quality if the relative
  usefulness of direct pooled features and their nonlinear correction changes
  as augmentation, labels, and LR evolve.

The counterargument is strong: the same CE that trains both branch matrices can
already alter the branch's effective amplitude and direction. A free scalar is
partly redundant with scaling the second head matrix, while changing the
factorization changes SGD conditioning and weight-decay behavior rather than
adding representational capacity. The scalar has zero decay while the second
matrix has `5e-4` decay, so training may inflate `s` and shrink the matrix, an
implicit reallocation not backed by local evidence. Direct full-LR Nesterov
updates may also move a one-dimensional gain too aggressively. These are
reasons for low confidence, not licenses to add a scalar-specific LR or decay.

## Semantic and RNG Gate

Use an ignored evaluator-free verifier that independently loads accepted
`a7c42dc:train.py`, blocks evaluator/test access, and fails before timing unless
all of the following hold:

1. Production diff is limited to the one scalar parameter, the forward operand,
   and at most one final-summary diagnostic. All accepted constants, model
   tensors, data, temporal transitions, loss, schedule, seed, evaluator, and
   cadence remain unchanged.
2. From cloned CPU/CUDA RNG states, accepted and candidate construction yields
   byte-identical values for every common parameter and buffer, identical
   post-construction CPU/CUDA RNG states, exactly one appended FP32 scalar equal
   to FP32 0.1, and total parameter count 1,003,483.
3. Independently reconstruct the seed-36036 pooled branch and prove its two
   matrices are exact. Prove creating the scalar alone consumes no RNG and does
   not alter allocator-persistent model state other than the new key.
4. On fixed CPU and CUDA inputs, prove initial pooled direct features, branch
   outputs, residual features, logits, loss, BN updates, and RNG equal accepted.
   Verify both early mixup (same lambda and permutation) and hard-label paths.
5. Independently compute `g_s` as the contraction of upstream feature gradient
   with branch output in float64 and a separately coded FP32 oracle. Require the
   production scalar gradient to be finite, nonzero, and within a fixed
   numerical tolerance; print all values before asserting.
6. Prove optimizer groups contain every accepted tensor exactly as before and
   contain the scalar exactly once in the zero-decay group. Options and ordering
   remain accepted. No accepted tensor changes merely from optimizer creation.
7. Verify fresh and deterministic preseeded-momentum scalar updates against the
   equations above. On the common first fixture, existing parameter gradients,
   momentum buffers, and updates match accepted, while the only extra state and
   direct update are the scalar and its buffer. Everything must remain finite.
8. Restore model/optimizer/input/RNG and replay each regime exactly. Verify the
   new parameter does not alter mixup or worker RNG, the one-way 65% transitions,
   time-based LR, finite-loss guard, evaluation frequency, or final summary.

A semantic failure closes this exact implementation. Do not repair it by
changing parameterization, initial value, placement, optimizer allocation, or
numeric precision.

## Timing and Exposure Gate

After semantics pass, compare complete accepted and candidate training steps on
the idle H20 for both early-mixup and hard-label regimes. Include H2D, LR group
writes, zeroing, mixup when active, full forward, CE, finite guard, backward,
coupled Nesterov step, and final synchronization. Use at least 20 warmups and
four counterbalanced windows of at least 50 steps per arm/regime with fresh,
deterministic fixtures. Print raw windows before all assertions.

Require finite measurements, population CV no greater than 5% for every
arm/regime, peak candidate allocation below 2,048 MiB, and:

```text
retention =
    (0.65 / candidate_early_ms + 0.35 / candidate_hard_ms) /
    (0.65 / accepted_early_ms  + 0.35 / accepted_hard_ms)
projected_passes = 130.304 * retention

retention >= 127 / 130.304 = 0.9746439096
projected_passes >= 127
```

One scalar should pass comfortably, but measured complete-body behavior is
authoritative. A stable miss closes the direct learned-gain implementation;
do not rerun timing, detach the scalar, fuse it into a matrix, or relax the
floor.

## Sole Score and Decision Contract

If both gates pass, reconfirm baseline 94.48% at `a7c42dc`, exact `train.py`
scope, one idle NVIDIA H20, local CIFAR-10, frozen `prepare.py`, and no stale
log. Execute exactly once at seed 42:

```bash
timeout 600s uv run train.py > run.log 2>&1
```

Require exit zero, one finite final summary, 300.0-300.1 counted seconds, wall
time below 600 seconds, 1,003,483 parameters, correct ordered mixup and
RandAugment transitions, unique every-fifth-epoch evaluations plus the final
partial epoch, and no traceback, OOM, worker, evaluator, or non-finite error.
Record realized passes as `num_steps * 256 / 50000`, peak VRAM, final scalar,
best/final accuracy, loss, and best-final gap. The final scalar is diagnostic
only and must never select a rescue.

Primary success is only `best_test_acc >= 94.58%`, exactly 0.10 points above
the accepted 94.48%. Preregister `final_test_acc >= 94.45%` and
`final_test_loss <= 0.2456` as non-decisive corroboration. Neither rescues a
primary miss or vetoes a valid primary success. At least 127 realized passes is
required to attribute the outcome to the learned-gain mechanism at normal
exposure. A valid lower-exposure result still counts as the sole score and may
not be rerun, but is operationally inconclusive.

## Falsifiable Hypothesis and Closure

If the useful relative amplitude of the accepted nonlinear pooled correction
changes during training, then allowing only its scale to follow the accepted
zero-decay SGD/Nesterov dynamics from an exactly function-preserving 0.1 start
will retain at least 127 projected and realized passes and raise fixed-seed
`best_test_acc` from 94.48% to at least 94.58%.

A normal-exposure score below 94.58% rejects this direct full-LR, zero-decay,
unconstrained learned scalar. Close immediate rescues: scalar LR or decay
tuning, log/sigmoid/softplus parameterization, positivity clamps, alternate
initial scales, freeze/unfreeze schedules, another seed, a rerun, or changes to
the head matrices. It does not prove fixed scale 0.1 was accidental, nor close
a future head interaction derived from independent evidence. A timing failure
closes systems viability only; an invalid scored run is diagnosed and never
rerolled.

## Principal Risks

- The proposal is adjacent to the sole recent success and lacks evidence that
  its fixed gain is limiting; expected improvement may be below 0.10 points.
- Scale is redundant with the second head matrix in function space but not in
  optimization space. Zero-decay scalar plus decayed matrix may create harmful
  norm reallocation rather than useful adaptation.
- Accepted full-LR Nesterov semantics may move the scalar sharply, cross zero,
  or amplify logits. Those outcomes are part of the test and cannot be repaired
  after observing diagnostics.
- A learned gain adds no capacity. It can only alter conditioning and temporal
  allocation between paths, while EXP036 already trains both branch matrices
  end to end.
- Fixed-seed top-1 is noisy at the 0.10-point margin. Endpoint accuracy and loss
  provide corroboration but cannot substitute for the preregistered threshold.
