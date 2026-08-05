# Proposal: Identity-Initialized Spatial-Standard-Deviation Residual

## Recommendation

Test one fixed post-spatial representation change: retain the accepted global
mean feature exactly, compute the per-channel population standard deviation of
the same final `128 x 8 x 8` post-BN/ReLU feature map, and add a fixed-scale
learned projection of that statistic before the accepted pooled residual MLP
and classifier. Use an identity-initialized, bias-free `128 -> 128` projection
and scale its output by exactly `0.1`:

```text
x       = ReLU(final_bn(layer3_output))             # [N, 128, 8, 8]
mu      = adaptive_avg_pool2d(x, 1).flatten(1)      # accepted mean path
sigma   = sqrt(var(x, dim=(-2, -1), correction=0) + 1e-5)
u       = mu + 0.1 * spatial_std_projection(sigma)
refined = u + 0.1 * pooled_head(u)                  # accepted head, unchanged
logits  = fc(refined)                               # accepted classifier
```

This is a coherent but exploratory candidate, not a high-confidence extension
of EXP036. Its attractive property is that it exposes information global mean
pooling provably discards while adding no learned operation on the spatial
grid. Its main weakness is that neither the scale nor identity initialization
is selected by direct local evidence about the final feature distribution.
The accepted `0.1` residual-head scale offers a defensible fixed convention,
but reusing it here is still an arbitrary one-point choice. The idea should
advance only if its semantic and measured `>=127`-pass feasibility gates pass
and if review judges this orthogonal representation bet stronger than the
other EXP042 candidates.

## Exact Production Treatment

Add constants, without a sweep or adaptive fallback:

```python
SPATIAL_STD_EPS = 1e-5
SPATIAL_STD_SCALE = 0.1
```

Construct the new module only after the complete accepted model, classifier,
and seed-36036 pooled head have been initialized. Use a restoring CPU RNG fork
because `nn.Linear`'s constructor draws a disposable random matrix even though
the matrix is immediately overwritten:

```python
with torch.random.fork_rng(devices=[]):
    self.spatial_std_projection = nn.Linear(
        widths[2], widths[2], bias=False
    )
    init.eye_(self.spatial_std_projection.weight)
```

Do not call `manual_seed` in this fork: the retained initialization is the
deterministic identity, so a new arbitrary seed would carry no information.
The fork must restore the global CPU generator after the discarded constructor
draw, and CPU-only construction must leave CUDA RNG unchanged. Registering the
module after `pooled_head` preserves every accepted parameter/buffer value and
key order, appending only `spatial_std_projection.weight`.

In `forward`, preserve the accepted final activation and exact
`adaptive_avg_pool2d(..., 1)` mean path. Compute `sigma` independently with
`torch.var(out, dim=(-2, -1), correction=0)` and add `SPATIAL_STD_EPS` before
the square root. Population variance is required: the 64 spatial positions
are the complete feature map being summarized, not a sample needing Bessel's
correction. `1e-5` is fixed prospectively because it matches the model's
existing BatchNorm numerical floor and makes the value and gradient finite for
a spatially constant channel. Do not use `torch.std`, `unbiased=True`,
`var_mean`, `E[x^2]-E[x]^2`, a clamp, or a data-derived epsilon; those alter
either reduction semantics, numerical behavior, or the protected mean path.

The exact forward edit is:

```python
out = F.relu(self.bn(out))
pooled = F.adaptive_avg_pool2d(out, 1).flatten(1)
spatial_std = torch.sqrt(
    torch.var(out, dim=(-2, -1), correction=0) + SPATIAL_STD_EPS
)
out = pooled + SPATIAL_STD_SCALE * self.spatial_std_projection(spatial_std)
out = out + POOLED_HEAD_SCALE * self.pooled_head(out)
return self.fc(out)
```

The identity start makes the initial new residual exactly `0.1 * sigma`.
Unlike a zero-finalized projection, it makes the statistic active in the first
forward pass and opens gradients to both diagonal and off-diagonal projection
entries immediately. It also avoids an arbitrary random projection seed. The
cost is intentional non-identity relative to the accepted function; initial
accepted logits are not expected to match.

The projection adds exactly `128 * 128 = 16,384` FP32 trainable parameters, so
the count rises from `1,003,482` to `1,019,866`. It adds 16,384 dense MACs per
image plus one population-variance reduction and square root over each of 128
channels. The existing generic grouping places its rank-2 matrix exactly once
in the continuous `5e-4` decay group. It receives the accepted time-varying LR,
momentum `0.9`, Nesterov semantics, and no special schedule. Do not add a bias,
learned gain, no-decay exception, projection-specific LR, normalization,
activation, dropout, cutoff, or telemetry in production.

## Mechanism and Local Evidence

Global mean pooling retains channel presence but erases spatial dispersion.
Two final maps can have the same per-channel mean while one is concentrated on
a few positions and the other is diffuse. Their population standard deviations
differ, so `sigma` can encode confidence, extent, and part-versus-background
heterogeneity unavailable to the accepted pooled vector. The identity start
first exposes each channel's own dispersion; training can then learn
cross-channel combinations through the full projection. Feeding the augmented
vector through the already successful pooled MLP allows nonlinear interaction
between means and dispersions without adding another nonlinear branch.

Local evidence is suggestive but indirect:

- EXP036 is the strongest support for placement. Its bias-free
  `128 -> 64 -> 128` ReLU correction after pooling added 16,384 parameters,
  retained 130.304 passes, and improved best/final/loss to
  `94.48/94.45/0.2456`. Cheap global channel interaction can therefore improve
  the accepted representation without spatial convolution.
- EXP012 is negative evidence against processing the whole 8x8 grid with a
  new learned bottleneck. Its `128 -> 64 -> 64 -> 128` spatial residual cost
  3.41M MACs/image, retained 135.49 passes, but scored 93.74 and 0.2873. This
  proposal performs only a fixed statistical reduction on the grid and moves
  all learned work after pooling; it neither retries that bottleneck nor
  assumes compressed spatial convolution preserves dense capacity.
- EXP017-025 close the tested stage-3 gating family. Full SE's signal required
  two input-dependent residual gates and dense cross-channel mixing, while
  final-only, static, and diagonal variants failed; diagnostic-free full SE
  still projected only 136.90 passes under the older base. The proposed branch
  does not gate either residual block, multiply an 8x8 activation, or reuse
  mean-conditioned SE. It summarizes the final nonnegative feature map's
  centered second moment once and adds it after all spatial blocks. Thus it is
  a distinct representation statistic, not a cheaper SE rescue.
- EXP041 showed that an auxiliary CE on the raw mean path was strongly aligned
  with the accepted objective yet worsened accuracy and loss. This candidate
  preserves exactly one CE on the accepted final logits; it changes the
  representation rather than diluting the successful head gradient.
- The system attribution says backpropagation consumes about 74% of step time,
  the accepted head only about 1.4% of forward time, and extra spatial compute
  directly competes with the 130-pass frontier. A reduction over 8,192 values
  per image and a 16k-MAC pooled projection are plausibly affordable, but
  variance backward and small-kernel launch overhead make timing measurement
  mandatory.

## Semantic and RNG Qualification

Use an ignored evaluator-free preflight that independently loads accepted
`a7c42dc:train.py`, blocks test-data construction/evaluation, and prints all
measurements before assertions. Before timing, require all of the following:

1. The tracked production diff changes only `train.py` and contains only the
   two constants, deterministic projection construction, spatial-statistic
   forward path, and at most a final-summary diagnostic. `prepare.py`, data,
   loss, schedules, augmentations, seed, and evaluator cadence are unchanged.
2. From cloned CPU/CUDA RNG states, accepted and candidate construction yields
   byte-identical values for every common named parameter and buffer and
   bitwise-equal post-construction global CPU/CUDA RNG states. The sole appended
   state is an FP32 `[128,128]` bias-free matrix exactly equal to `eye(128)`;
   total trainable parameters are exactly `1,019,866`.
3. Independently reconstruct the accepted seed-36036 pooled head and require
   both matrices to remain byte-identical. Construct the new linear repeatedly
   from different pre-fork CPU states and prove the retained identity bytes and
   post-fork states are exact, while CUDA state never changes.
4. Hook the final BN/ReLU activation and classifier input on deterministic CPU
   and CUDA inputs. Independently compute the accepted pooled mean, population
   variance, `sqrt(var + 1e-5)`, identity projection, augmented feature, accepted
   pooled-head correction, and logits. Require production tensors to match
   fixed FP32 tolerances derived before looking at a score. Require the direct
   `pooled` tensor itself to equal the accepted model's pooled tensor, proving
   the established mean reduction was not replaced.
5. Verify zero-variance synthetic channels produce exactly finite
   `sqrt(1e-5)` statistics and finite gradients. On paired synthetic feature
   maps with equal per-channel means but deliberately different variances,
   require accepted mean-only subhead inputs to match and candidate augmented
   inputs/logits to differ. This is a semantic discrimination check, not an
   accuracy proxy.
6. Print, without gating or tuning from their magnitudes, the initial
   `||0.1 P(sigma)|| / ||mu||` ratio, cosine between `mu` and `sigma`, RMS/max
   classifier-input delta, and RMS/max logit delta on fixed synthetic and one
   local training batch. Require only finite values and a nonzero intervention.
   Do not use these diagnostics to change epsilon, scale, initialization, or
   projection width.
7. Enumerate optimizer groups by parameter identity and name. Require every
   trainable tensor exactly once, all accepted membership unchanged, and only
   the new matrix appended to the `5e-4` group. LR, momentum, dampening,
   Nesterov, and group ordering must remain accepted.
8. For both an early batch-shared-mixup fixture and a hard-label fixture,
   require finite logits/loss, finite nonzero new-matrix gradient and update,
   finite gradients for the backbone, accepted pooled head, and classifier,
   and exact replay after restoring model/optimizer/input/RNG. Verify a fresh
   and a deterministic preseeded-momentum new-matrix update against an
   independent coupled-SGD/Nesterov oracle. With `d = grad + 5e-4 * W`, require
   fresh `buffer=d`, update direction `d + 0.9*buffer`; with prior buffer `b`,
   require `buffer_next=0.9*b+d` and direction `d+0.9*buffer_next`.
9. Compare production autograd gradients through the statistic against an
   independently expressed population-variance formula on a small FP64 tensor,
   including a constant-channel case. Require finite agreement under a fixed
   numerical tolerance. Prove forward itself consumes no CPU or CUDA RNG.
10. Audit strict 65% mixup behavior, exhausted-epoch RandAugment transition,
    time-based LR samples, finite-loss guard, and every-fifth-plus-final unique
    evaluation semantics as accepted.

A semantic failure closes this exact implementation before scoring. Do not
repair it by switching variance formulas, changing epsilon/scale, zeroing or
randomizing the projection, adding normalization, or moving the branch.

## Throughput Estimate and Binding Timing Gate

The dense projection has the same 16,384 MAC count as the accepted pooled MLP,
but `torch.var` introduces a read/reduction/backward over the final 8x8 map.
The arithmetic is far below one percent of convolution MACs; nevertheless the
small reduction kernels and saved tensors can be latency-bound. A reasonable
prospective estimate is roughly 1-2.5% complete-step overhead, corresponding
to about 127.0-129.0 passes from the accepted 130.304 exposure. This range is
not evidence and must not replace direct measurement; the lower edge is
uncomfortably close to the protected floor.

Compare independent accepted and candidate modules on the idle H20 with equal
model, optimizer, fixture, and private RNG states. Time complete
production-equivalent early-mixup and hard-label steps including pinned H2D,
LR writes, zeroing, Beta/permutation/interpolation when active, full forward,
loss and finite guard, backward, accepted Nesterov step, and synchronization.
Use at least 20 disposable warmups per arm/regime and four counterbalanced
windows of at least 50 measured steps per arm/regime, with fresh deterministic
fixtures and cloned starting states per pair. Print every raw window before
assertions.

Require finite values, population CV no greater than 5% for every arm/regime,
candidate peak allocation below 2,048 MiB, and:

```text
retention =
    (0.65 / candidate_early_ms + 0.35 / candidate_hard_ms) /
    (0.65 / accepted_early_ms  + 0.35 / accepted_hard_ms)
projected_passes = 130.304 * retention

retention >= 127 / 130.304 = 0.9746439096
projected_passes >= 127.0
```

No DataLoader benchmark is needed because transforms, workers, batch shape,
and H2D volume are source-identical. A stable gate miss is final: do not rerun
timing, fuse or approximate variance, reduce projection width, change the
scale, or relax the floor. Such a result closes systems viability of this exact
treatment without spending the sole accuracy run.

## Sole Scored Run and Decision Contract

If semantic and timing gates pass, reconfirm baseline `94.48%` at `a7c42dc`,
success threshold `94.58%`, exactly one idle NVIDIA H20, local CIFAR-10, frozen
`prepare.py`, clean experiment scope, and no stale `run.log`. Run exactly once:

```bash
timeout 600s uv run train.py > run.log 2>&1
```

Require exit zero, one finite final summary, counted training time within the
accepted `300.0-300.1`-second envelope, wall time below 600 seconds, exactly
`1,019,866` parameters, correct ordered mixup and exhausted-epoch RandAugment
transitions, unique every-fifth-epoch evaluations plus the final partial epoch,
at most one evaluation per epoch, and no traceback, OOM, worker, evaluator, or
non-finite error. Record realized passes as `num_steps * 256 / 50000`, peak
VRAM, transition steps, best/final accuracy, loss, best-final gap, and all
preflight diagnostics. Never inspect interim accuracy for control flow and
never launch a second valid score.

Primary success is only `best_test_acc >= 94.58%`. Preregister
`final_test_acc >= 94.45%` and `final_test_loss <= 0.2456` as non-decisive
corroboration. Neither rescues a primary miss nor vetoes a primary success; a
success without corroboration must be described as fragile. At least 127
realized passes is required to attribute the result to the intended
normal-exposure mechanism. A valid lower-exposure result still counts as the
sole score and may not be rerun, but the representation hypothesis is
operationally inconclusive.

## Confounds, Risks, and Family Closure

- The branch changes the initial function and all common gradients immediately.
  A score cannot separate the usefulness of `sigma` from the conditioning
  effect of adding an identity-initialized matrix before the accepted head.
- There is no mean-only projection control. A gain could reflect generic extra
  pooled linear capacity rather than unique second-moment information, while a
  miss could reflect projection dynamics rather than useless spatial variance.
- For post-BN/ReLU activations, mean and standard deviation are correlated.
  `sigma` may be mostly redundant with activation magnitude or may encode
  nuisance spread rather than object geometry.
- The fixed scale `0.1` is borrowed from EXP036, and identity initialization is
  chosen for determinism and immediate signal, not because local data establish
  either as optimal. The initial correction could still be too large or too
  small. Diagnostic magnitudes cannot trigger a rescue.
- Continuous `5e-4` decay shrinks the identity diagonal while SGD learns both
  diagonal and cross-channel terms. This is the accepted matrix policy, but it
  entangles representation learning with decay-induced departure from identity.
- `torch.var` backward can cost more than its arithmetic count suggests and
  may reduce exposure below 127. The timing gate, not the MAC estimate, decides.
- One fixed seed and a ten-example acceptance margin cannot establish an
  average effect. No reroll is permitted.

If a valid run with at least 127 passes scores below `94.58%`, retain the
accepted mean-only input to the pooled head and close the immediate family:
do not try a different epsilon, scale, identity multiplier, zero/random
initialization, projection bias, low rank, activation, normalization,
mean-plus-variance concatenation, alternate placement, temporal cutoff,
another seed, or a rerun as a rescue. The result rejects this exact population-
standard-deviation residual, not all second-order pooling in principle; a
future second-order method would need an independently motivated formulation
and should be deprioritized after a normal-exposure negative. A timing failure
closes only the exact system design. An invalid scored run is diagnosed under
the loop protocol and never converted into a seed reroll.

## Falsifiable Hypothesis

If spatial dispersion in the final post-BN/ReLU feature map contains
class-boundary information discarded by global mean pooling, then adding the
fixed identity-initialized `0.1 * Linear_128(sigma)` residual before the
accepted pooled MLP will retain at least 127 projected and realized passes and
raise fixed-seed CIFAR-10 `best_test_acc` from 94.48% to at least 94.58%, with
final accuracy at least 94.45% and final loss no worse than 0.2456 as
non-decisive corroboration. A valid normal-exposure miss falsifies this one
fixed formulation and closes its immediate rescue neighborhood.

## Local Sources

- `experiments/012/04-analysis.md`: the half-width learned spatial bottleneck
  retained exposure but regressed accuracy and loss.
- `experiments/017/04-analysis.md` through `experiments/025/04-analysis.md`:
  full SE's weak signal required two dense conditional gates; static,
  final-only, and diagonal simplifications failed, and the diagnostic-free full
  treatment missed its exposure gate.
- `experiments/036/04-analysis.md`: the accepted cheap post-pooling residual
  MLP improved best/final accuracy and loss at 130.304 passes.
- `experiments/041/04-analysis.md`: direct-path auxiliary supervision retained
  normal exposure but weakened the accepted pooled-head learner.
- `02-system-understanding.md`: generalization and boundary quality, not fit,
  memory, I/O, or wall time, limit the goal; spatial backward work competes
  directly with the fixed exposure regime.
