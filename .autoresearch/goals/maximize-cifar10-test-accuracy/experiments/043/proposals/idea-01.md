# Proposal: Convolution-Only Gradient Centralization

## Recommendation

Test one globally consistent, convolution-only gradient-centralization rule as
an orthogonal optimizer intervention. Immediately after the accepted backward
pass and before the accepted `optimizer.step()`, replace each convolution
weight's data gradient by its per-output-filter zero-mean projection:

```python
centralized_params = [
    module.weight for module in model.modules() if isinstance(module, nn.Conv2d)
]

# After loss.backward(), before optimizer.step().
for parameter in centralized_params:
    gradient = parameter.grad
    if gradient is not None:
        gradient.sub_(gradient.mean(dim=(1, 2, 3), keepdim=True))
```

The fixed rule applies from the first through the final optimizer update to all
18 `Conv2d.weight` tensors: the stem, both convolutions in every residual block,
and the three projection shortcuts. It centralizes over input-channel, height,
and width axes independently for each output channel. Do not centralize BatchNorm
affines, the two accepted pooled-head matrices, classifier weight or bias, or
any activation gradient. Do not normalize or rescale the projected gradient.

This candidate is worth one tightly gated score, but confidence should be
moderate to low. Its appeal is that it changes optimization geometry without
changing the accepted model's function, state, loss, or augmentation recipe,
and it removes only one direction per convolution output channel. Its main
weakness is equally concrete: the deleted common-mode data-gradient component
may encode useful low-frequency/DC response, especially in the unnormalized
stem and positive post-ReLU residual inputs. There is no local result proving
that convolution-gradient conditioning, rather than representation or
regularization, is the remaining limiter.

## Exact Scope and Projection

For a convolution gradient `G` of shape `[C_out, C_in, K_h, K_w]`, let
`D = C_in * K_h * K_w` and flatten the last three axes. For each output filter
`o`, apply the orthogonal projector

```text
mean_o = (1 / D) * sum_j G[o, j]
P(G)[o, j] = G[o, j] - mean_o
```

Consequently, in exact arithmetic:

```text
sum_j P(G)[o, j] = 0
P(P(G)) = P(G)
||P(G)||_F <= ||G||_F
```

The accepted model has 983,472 convolution weights and 1,392 convolution
output filters across its 18 convolution tensors. The treatment therefore
removes only 1,392 scalar data-gradient directions, about 0.142% of the
convolution-weight degrees of freedom, while leaving all 1,003,482 trainable
parameters present and the forward function exactly accepted. It nonetheless
touches the update path of nearly all parameters by count, so this is not a
cosmetic change.

Use a precomputed Python list of convolution weights created after
`model.to(device)` and before optimizer construction. The list is not model or
optimizer state and must not be rebuilt by name on every step. The production
training-loop change is exactly the in-place projection above between
`loss.backward()` and `optimizer.step()`. Add no constant, threshold, start or
stop fraction, per-layer exception, epsilon, clipping, learned gain, logging,
or alternate gradient scale.

The rule deliberately excludes all 2D linear matrices. The accepted
`128 -> 64 -> 128` pooled residual head is the sole recent improvement, and
EXP040 showed that constraining classifier geometry can substantially worsen
both accuracy and loss. Preserving their exact gradients makes this a test of
convolution-filter optimization rather than an entangled retuning of the
successful readout. Applying the same formula to linear layers would be a
different experiment, not a permissible rescue.

Everything else remains accepted at `a7c42dc`: parameter and buffer bytes,
seed 42, `(2,2,3)` WRN, the isolated seed-36036 pooled head, FP32, batch 256,
early worker-private RandAugment N1/M5, batch-shared alpha-0.2 mixup through
65%, exhausted-iterator RandAugment cutoff, 0.2-to-0.002 time cosine, continuous
`5e-4` matrix decay, momentum 0.9, Nesterov, finite-loss guard, loader, budget,
and every-fifth-plus-final evaluation cadence.

## Coupled Decay, Momentum, and Nesterov Semantics

Centralization acts on the loss-derived gradient **before** PyTorch SGD adds
coupled weight decay. This ordering is mandatory. For one selected convolution
weight `W_t`, raw autograd gradient `G_t`, projector `P`, decay
`lambda = 5e-4`, momentum `mu = 0.9`, dampening zero, and current learning rate
`eta_t`, the intended update is:

```text
C_t = P(G_t)
D_t = C_t + lambda * W_t

fresh state:       B_t = D_t
existing buffer:   B_t = mu * B_(t-1) + D_t

Nesterov direction Q_t = D_t + mu * B_t
W_(t+1) = W_t - eta_t * Q_t
```

On the first update, `Q_t = 1.9 * D_t`. Centralization is not applied to
`D_t`, `B_t`, or `Q_t`; doing so would also project away decay and momentum
components and would define a different optimizer. Thus the data-gradient
contribution to each output-filter mean is zero, while the existing mean weight
continues to shrink under ordinary coupled decay and its momentum history. For
an output-filter mean in exact arithmetic:

```text
mean(D_t) = lambda * mean(W_t)
```

For every excluded tensor, `D_t = G_t + lambda * W_t` in the accepted matrix
decay group or `D_t = G_t` in the accepted no-decay group, followed by the
same momentum/Nesterov equations. No optimizer subclass, hook, second backward,
parameter perturbation, or custom momentum buffer is needed.

## Plausibility and Generalization Diagnosis

The system understanding places about 74% of an isolated step in backward and
shows that the accepted model nearly interpolates the hard tail while retaining
0.2456 test loss. The immediate limiter is generalization and boundary quality,
not data delivery, memory, or convergence time. Gradient centralization could
help only if removing the common component of each filter's stochastic data
gradient improves conditioning or reduces poorly generalizing common-mode
motion while retaining useful feature learning. Since `P` is an orthogonal
projection and not a norm rescaling, it cannot amplify the full gradient norm;
it may reduce correlated drift across a filter's coefficients.

There are reasons this geometry may fit the architecture. Most convolutions
consume BatchNorm/ReLU-conditioned residual features, and residual/shortcut
paths provide other routes for low-frequency information. Removing one shared
direction per output filter is much less restrictive than forcing every
input-channel kernel to have zero spatial sum. It also leaves BN scale/shift,
the nonlinear pooled correction, and the affine classifier free to adapt.

The counterarguments are stronger than generic claims about smoother loss
surfaces. ReLU inputs have positive, nonzero means, so a uniform filter-weight
direction is not redundant. The stem sees normalized images but no preceding
BN, and the 1x1 shortcuts have no spatial redundancy at all: their projection
removes the all-input-channel direction. Coupled decay then steadily erodes the
initial weight component in every deleted direction because data gradients can
never replenish it. The fixed 300-second run also leaves no opportunity for a
separate convergence study. A normal-exposure miss should close this exact
global convolution rule without layer exclusions, schedule tuning, or partial
centralization rescues.

No external or network evidence is used. The proposal is grounded in the
accepted source, current system attribution, and local experiment record.

## Distinction From Prior Interventions

- **Not SAM (EXP021/022):** SAM perturbs parameters and obtains an additional
  forward/backward gradient at a neighboring point. This candidate uses one
  accepted forward/backward and applies a deterministic linear projection to
  that gradient. It targets common-mode filter updates, not neighborhood
  sharpness, and should be far cheaper than the failed final-window SAM paths.
- **Not weight-decay tuning (EXP007/037/038):** decay acts on current parameter
  values. Centralization acts only on the data gradient before the unchanged
  coupled `5e-4` term is added. The decay strength, duration, and membership
  remain accepted, including for the classifier and pooled head.
- **Not classifier geometry (EXP040):** equal-row normalization changed the
  effective forward classifier and its gradients continuously. This candidate
  leaves all forward logits exactly accepted at initialization and excludes
  classifier and pooled-head gradients from projection.
- **Not gradient clipping or normalization:** there is no threshold and no
  division by a norm. Projection may reduce norm as an algebraic consequence,
  but its fixed operation is removal of one mean direction per output filter.
- **Not a momentum reset:** accepted buffers are created and accumulated
  normally from the projected-data-plus-decay directions for the whole run.

## Semantic and RNG Qualification

Use an ignored, evaluator-free preflight that independently loads the accepted
`a7c42dc:train.py`, blocks test-data construction/evaluation, and prints all
diagnostics before assertions. Fail closed before timing unless every condition
passes:

1. Diff production against `git show a7c42dc:train.py`. Require changes only
   for the precomputed convolution-weight list and post-backward projection.
   Hash `prepare.py`; prove all accepted constants, data, model, initialization,
   loss, transitions, optimizer construction, time accounting, and evaluator
   cadence are unchanged.
2. From cloned CPU and CUDA RNG states, instantiate accepted and candidate
   models and optimizers. Require identical named parameters, buffers, bytes,
   group ordering, options, and total count 1,003,482; identical post-setup RNG
   states; and exactly 18 distinct selected tensors, all and only
   `Conv2d.weight` objects, totaling 983,472 elements and 1,392 output filters.
3. On cloned deterministic early-mixup and hard-label fixtures, run accepted
   forward/loss/backward once and clone every raw gradient. Before projection,
   require candidate logits, loss, BN updates, gradients, and RNG to match the
   accepted path. Apply production centralization only afterward.
4. Independently compute the FP64 and separately expressed FP32 projection for
   each selected raw gradient. Require production gradients to match within
   fixed FP32 reduction tolerance, every per-output mean to be numerically zero,
   projection to be idempotent, and projected norm not to exceed raw norm beyond
   tolerance. Print raw/projected norms, removed fraction, and maximum residual
   mean; require a finite nonzero intervention but do not tune from its size.
5. Require every excluded parameter gradient, especially both pooled-head
   matrices and classifier weight/bias, to remain byte-identical to its cloned
   raw gradient after the centralization loop. Prove the selected list has no
   duplicate identities and no BN or linear tensor.
6. Verify fresh and deterministic preseeded-momentum updates against the exact
   equations above for a 3x3 body convolution and a 1x1 shortcut. Independently
   verify an excluded pooled-head matrix and classifier update remain accepted.
   For selected tensors, require the data-gradient mean to vanish while the
   effective SGD direction retains `5e-4 * W`; never compare against a projector
   applied after decay.
7. Restore model, optimizer, fixture, and RNG and replay both regimes exactly.
   Require finite loss/gradients/parameters/buffers, no autograd version error,
   and exact replay. Prove the projection consumes no CPU/CUDA RNG and introduces
   no parameter, buffer, optimizer option, or persistent model state.
8. Statically and dynamically re-prove batch-shared alpha-0.2 mixup, the strict
   65% mixup cutoff, exhausted-epoch RandAugment transition, time-based LR
   samples, finite-loss guard, seed 42, single-H20 use, and every-fifth-plus-final
   unique evaluation behavior.

A semantic failure closes this exact implementation before scoring. Repair only
a demonstrable verifier or implementation defect while preserving the selected
tensor set, axes, full-run timing, before-decay ordering, and lack of rescaling.
Do not exclude the stem/shortcuts, add linear layers, change axes, centralize
after decay, schedule the projection, or relax tolerances based on favorable
diagnostics.

## H20 Timing and Exposure Gate

Each training step adds 18 reductions and 18 broadcast subtractions over a
combined 983,472 FP32 values. Arithmetic and memory traffic are small relative
to convolution backward, but roughly 36 sequential small CUDA launches can be
latency-bound on the H20. A prospective 1.5-3.5% step penalty is plausible;
the high end would violate the protected 127-pass regime. Static arithmetic is
therefore not a feasibility verdict.

On one otherwise idle NVIDIA H20, compare independent accepted and candidate
models with identical weights, optimizer states, fixtures, and private RNG.
Measure complete production-equivalent early-mixup and hard-label steps,
including H2D, LR writes, zeroing, mixup where active, forward, CE, finite-loss
check, backward, the candidate's actual centralization loop, coupled Nesterov
step, and final synchronization. Use at least 20 warmups and four
counterbalanced windows of at least 50 steps per arm and regime. Wall-clock
timing bracketed by CUDA synchronization is authoritative because host launch
overhead is part of the scored step. Print every raw window before assertions.

Require finite measurements, population CV no greater than 5% for every
arm/regime, candidate peak allocation below 2,048 MiB, and:

```text
retention =
    (0.65 / candidate_early_ms + 0.35 / candidate_hard_ms) /
    (0.65 / accepted_early_ms  + 0.35 / accepted_hard_ms)
projected_passes = 130.304 * retention

retention >= 127 / 130.304 = 0.9746439096
projected_passes >= 127.0
```

The accepted 130.304-pass scored run is the preregistered exposure reference.
A stable timing miss closes systems viability; do not rerun favorable windows,
fuse with an unverified private API, remove selected layers, centralize less
often, or relax the floor.

## Sole Score and Decision Contract

If semantics and timing pass, reconfirm the 94.48% baseline at `a7c42dc`, the
94.58% success threshold, one idle NVIDIA H20, frozen `prepare.py`, local data,
exact `train.py` scope, and no stale log. Run exactly once at seed 42:

```bash
timeout 600s uv run train.py > run.log 2>&1
```

Require exit zero, one finite final summary, 300.0-300.1 counted seconds, wall
time under 600 seconds, 1,003,482 parameters, correct ordered one-way mixup and
RandAugment transitions, unique every-fifth plus final evaluations, and no
traceback, OOM, worker, evaluator, or non-finite error. Record `num_steps`,
realized passes as `num_steps * 256 / 50000`, peak VRAM, best/final accuracy,
final loss, and best-final gap.

Primary success is only `best_test_acc >= 94.58%`, at least 0.10 points above
94.48%. Preregister `final_test_acc >= 94.45%` and `final_test_loss <= 0.2456`
as non-decisive corroboration. They neither rescue a primary miss nor veto a
valid primary success. At least 127 realized passes is required for a
normal-exposure mechanism attribution. A valid lower-exposure score still
counts as the sole score and cannot be rerun, but is operationally inconclusive.

## Falsifiable Hypothesis and Closure

If common-mode convolution-filter data gradients are a harmful source of
conditioning or generalization error in the accepted pooled-head learner, then
projecting all convolution gradients per output filter before unchanged coupled
decay will retain at least 127 projected and realized passes and raise fixed-seed
`best_test_acc` from 94.48% to at least 94.58%.

A normal-exposure score below 94.58% rejects this exact all-convolution,
all-training, axes-`(1,2,3)`, before-decay projection. Close immediate rescues:
excluding stem or shortcuts, spatial-only axes, adding linear matrices, partial
projection coefficients, late/early windows, alternating steps, gradient
renormalization, decay-order changes, another seed, or a rerun. It does not
prove every constrained optimizer geometry is useless; it shows this clean
standardized projection is not sufficient on the accepted recipe. A timing
failure closes H20 feasibility only, while a semantic failure diagnoses the
implementation and is not score evidence.

## Principal Risks

- Uniform filter directions are not redundant under ReLU inputs. Their removal
  may impair useful DC and cross-channel response, particularly in the stem and
  1x1 projection shortcuts.
- Coupled decay continues shrinking weight components along directions that no
  data gradient can replenish, so the long-run treatment is more restrictive
  than a one-step projection description suggests.
- Thirty-six small kernels per step may consume the entire 2.54% exposure
  allowance despite touching less than 4 MiB of gradient data.
- The proposal has no local positive signal and affects most trainable weights;
  fixed-seed top-1 variance could exceed the expected gain even when optimization
  diagnostics look benign.
