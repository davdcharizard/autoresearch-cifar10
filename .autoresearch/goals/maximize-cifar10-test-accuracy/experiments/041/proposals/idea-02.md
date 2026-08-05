# Proposal: Training-Only Direct-Path Auxiliary Cross-Entropy

## Claim and Scope

Preserve the accepted `a7c42dc` model and its sole inference path, but during
training also classify the raw post-pooling feature through the already shared
classifier. With

```text
z          = GAP(ReLU(BN(backbone(x))))
h          = pooled_head(z)
z_refined  = z + 0.1 h
main       = fc(z_refined)
direct     = fc(z)
```

evaluation remains exactly `main`, while the training objective is a convex
blend of main-path and direct-path cross-entropy. No parameter, initialization,
data, RNG, schedule, temporal boundary, optimizer setting, evaluator call, or
inference operation changes.

The narrow hypothesis is that weakly preserving linear class usefulness of the
dominant direct feature `z` will regularize the accepted nonlinear correction
without removing it, improving fixed-seed boundary quality from 94.48% to at
least 94.58%. This is not based on a diagnosed collapse of `z`; it is an
exploratory deep-supervision test whose strongest local counterevidence is that
EXP036's nonlinear pooled correction is itself the accepted gain.

## Local Evidence and Novelty

- EXP036 added the scale-0.1 bias-free `128 -> 64 -> 128` pooled residual MLP
  and improved best/final accuracy from 94.32/94.22% to 94.48/94.45%, while
  improving final test loss from 0.2523 to 0.2456 at 130.304 data passes. The
  direct `z` path was retained in the forward representation, but it had no
  independent classification objective.
- The system diagnosis identifies generalization and boundary quality, rather
  than memory or input delivery, as the remaining limit. A second `128 -> 10`
  classifier application and CE occur after all spatial computation, so this
  tests representation supervision without adding high-resolution backward
  work or inference cost.
- EXP040 showed that forcibly equalizing classifier row radii harms the accepted
  boundary at normal exposure. This proposal leaves the affine classifier's
  class-specific radii completely free and changes only the training signal.
- Prior additive regularizers, masking, geometry constraints, and altered tail
  optimization often worsened accuracy. Therefore the auxiliary objective must
  be weak, fixed prospectively, and receive only one score. It is not a license
  for a coefficient, cutoff, detach, or separate-head sweep.
- EXP028 mentioned a discarded early stage-2 classifier, but did not execute
  it. This proposal is materially narrower: it adds no classifier parameters,
  does not supervise an intermediate spatial stage, does not require feature
  adapters, and keeps the accepted objective active for the entire run.

No network source is required or permitted. The proposal uses only the local
accepted source, system measurements, and completed experiment record.

## Exact Production Semantics

### Model output contract

Refactor only the accepted `WideResNet.forward` return section so training can
request direct logits explicitly:

```python
def forward(self, x, return_direct_logits=False):
    # Accepted backbone, final BN/ReLU, pooling, and flattening are unchanged.
    pooled = out
    refined = pooled + POOLED_HEAD_SCALE * self.pooled_head(pooled)
    main_logits = self.fc(refined)
    if return_direct_logits:
        return main_logits, self.fc(pooled)
    return main_logits
```

The exact argument name may follow local style, but its default must be false.
The main logits must be computed first with the same `self.fc(refined)` call as
accepted. Do not concatenate the features into one larger GEMM, because that
could change main-logit floating-point accumulation and weaken the accepted
identity oracle. Do not use `self.training` to change return type: evaluation
and any ordinary caller must always receive the accepted tensor unless the
auxiliary return is explicitly requested.

The scored evaluator continues calling `model(inputs)` and therefore executes
one classifier call, returns one tensor, and is bitwise equivalent to accepted
inference at common state. Only the training loop calls
`model(inputs, return_direct_logits=True)`. The auxiliary path is not computed
during evaluation and adds zero inference parameters or operations.

### Fixed auxiliary weight

Let `s = POOLED_HEAD_SCALE = 0.1`, and define for one target `y`:

```text
C_main(y)   = CE(main_logits, y)
C_direct(y) = CE(direct_logits, y)
L(y)        = (1 - s) C_main(y) + s C_direct(y)
```

Thus the fixed main/direct weights are exactly `0.9/0.1`. Reusing the accepted
scale is a defensible structural anchor: the accepted feature is the direct
path plus a 0.1-scaled nonlinear correction, and the objective gives the direct
representation a correspondingly subordinate independent constraint without
introducing a new coefficient. The convex form is important. Using
`C_main + 0.1 * C_direct` would raise the aggregate shared-loss scale by about
10% when the two paths agree, confounding supervision with effective LR and
decay strength. A `0.9/0.1` blend keeps the nominal CE scale at one.

This tie is not evidence that loss weight and feature amplitude are equivalent
units, nor that 0.1 is optimal. It defines one falsifiable operating point. Do
not derive the weight from test behavior, gradient diagnostics, or timing; do
not tune, warm up, anneal, disable, or adapt it.

### Batch-shared mixup

Keep the accepted `mixup_batch` call exactly once. Both logits must be computed
from the same `mixed_inputs`, and both objectives must use the same scalar
batch-shared `mix`, `targets_a`, and `targets_b` returned by that call:

```python
main_logits, direct_logits = model(
    mixed_inputs, return_direct_logits=True
)
main_loss = mix * F.cross_entropy(main_logits, targets_a) + (
    1.0 - mix
) * F.cross_entropy(main_logits, targets_b)
direct_loss = mix * F.cross_entropy(direct_logits, targets_a) + (
    1.0 - mix
) * F.cross_entropy(direct_logits, targets_b)
loss = (1.0 - POOLED_HEAD_SCALE) * main_loss + (
    POOLED_HEAD_SCALE * direct_loss
)
```

There must be no second permutation, Beta draw, image mixture, per-example
coefficient, target smoothing, or detached target/logit. Preserve operation
ordering within each accepted main CE expression. In particular, do not replace
the accepted pair of `F.cross_entropy` calls with a soft-target or gathered
log-softmax implementation merely to fuse work; that would confound the main
loss numerics.

### Hard-label tail

After the accepted 65% transition, request the same two logits on clean
crop/flip inputs and use the exact hard targets for both paths:

```python
main_logits, direct_logits = model(inputs, return_direct_logits=True)
main_loss = F.cross_entropy(main_logits, targets)
direct_loss = F.cross_entropy(direct_logits, targets)
loss = (1.0 - POOLED_HEAD_SCALE) * main_loss + (
    POOLED_HEAD_SCALE * direct_loss
)
```

The auxiliary objective remains active for the complete 300 counted seconds.
An early-only auxiliary cutoff would combine representation supervision with a
new schedule and could not be inferred from this result. Mixup and worker-safe
RandAugment still end under their accepted independent boundary semantics.

## Parameter, RNG, and Gradient Contract

The candidate has exactly the accepted 1,003,482 trainable parameters and
identical state keys, shapes, dtypes, devices, and initialized bytes. The
optimizer must retain exactly the accepted groups: every rank-at-least-two
matrix, including `fc.weight` and both pooled-head matrices, at `5e-4` decay;
all rank-below-two tensors at zero decay; LR, 0.9 Nesterov momentum, and state
construction unchanged. There is no auxiliary head, buffer, learned weight,
temperature, stop-gradient, extra zeroing, gradient clipping, or optimizer
state.

The intended gradient decomposition on a fixed batch is:

```text
g_head(L)     = 0.9 g_head(C_main)
g_fc(L)       = 0.9 g_fc(C_main) + 0.1 g_fc(C_direct)
g_backbone(L) = 0.9 g_backbone(C_main) + 0.1 g_backbone(C_direct)
```

`C_direct` must give no gradient to either pooled-head matrix, but it must give
finite nonzero gradients to `fc`, final BN, and the spatial backbone. Neither
`pooled` nor the direct logits may be detached. The shared classifier is
intentional: it prevents extra inference parameters and tests whether one
boundary can serve both raw and refined feature geometries. It also creates a
real compromise risk if those geometries prefer different class vectors.

The treatment changes three coupled effects that cannot be separated by this
single score:

1. it rewards linear usefulness of raw `z` throughout the backbone;
2. it reduces pooled-head supervision to 90% of accepted strength;
3. it changes `fc` and backbone updates according to the main/direct gradient
   agreement.

A gain supports the complete shared-classifier objective, not any one of these
effects alone. A miss likewise does not diagnose which effect failed.

## Semantic Preflight

Create an ignored, evaluator-free preflight that independently loads
`git show a7c42dc:train.py` with a stub `prepare.Eval`, instantiates accepted
and candidate models from controlled seed 42, and fails closed before timing.
It must print measurements before assertions and prove all of the following:

1. **Scope and state identity**: the only production diff is `train.py`; frozen
   files are untouched; parameter count remains 1,003,482; state key/shape/
   dtype sets and every initialized tensor byte match accepted; post-construction
   CPU and CUDA RNG states match; optimizer membership and hyperparameters are
   identical.
2. **Accepted inference identity**: on fixed finite FP32 CUDA inputs in eval
   mode, candidate `model(x)` is bitwise equal to accepted output. A forward
   hook proves exactly one `fc` invocation. The returned object is a single
   `[batch, 10]` tensor, including through the real evaluator-compatible default
   call surface.
3. **Dual-output formula**: in train and eval modes,
   `model(x, return_direct_logits=True)` returns exactly two `[batch, 10]`
   tensors. Its main tensor is bitwise equal to the default candidate call;
   an independent stage-by-stage reconstruction proves main equals
   `fc(z + 0.1 * pooled_head(z))` and direct equals `fc(z)`. A hook proves two
   and only two classifier calls in dual mode, in main-then-direct order.
4. **No state or RNG side effect**: default and dual-output forwards add no
   state keys, mutate no parameters or buffers beyond ordinary accepted BN
   behavior, and consume no CPU/CUDA RNG. Compare eval mode for exact absence of
   mutation and matched train-mode models for accepted BN-buffer evolution.
5. **Mixup objective oracle**: use a fixed synthetic batch, fixed scalar mix,
   and explicit fixed permutation. Independently calculate all four accepted
   CE terms and the exact nested `0.9/0.1` formula; match the production loss at
   tight FP32 tolerances. Prove one scalar mix and one target pair are shared by
   both paths, and that candidate main loss alone is bitwise equal to the
   accepted mixup loss at common logits.
6. **Hard objective oracle**: independently match the two-CE convex blend on
   one hard-label batch, while proving the main CE alone is bitwise equal to
   accepted hard CE.
7. **Gradient decomposition**: use cloned common state and identical fixtures
   to obtain main-only, direct-only, and combined gradients. Direct-only head
   gradients must be absent/zero; combined pooled-head gradients must equal
   `0.9 * main`; combined `fc`, final-BN, and representative early/late
   backbone gradients must equal `0.9 * main + 0.1 * direct` within stated FP32
   tolerances. All intended combined gradients must be finite and nonzero.
8. **Optimizer oracle**: independently reproduce one fresh-state and one
   preseeded-momentum Nesterov update for representative convolution,
   pooled-head, classifier weight/bias, and BN parameters using the combined
   loss. Match parameter and momentum-buffer values, proving no hidden
   parameter-group or effective-LR change.
9. **Temporal/source invariants**: seed 42, batch 256, FP32, `(2,2,3)` blocks,
   N1/M5 worker-private RandAugment, alpha-0.2 batch-shared mixup, both accepted
   65% transitions, LR curve/floor, 300-second accounting, `MAX_STEPS`, and
   at-most-once-per-epoch evaluation remain exact.

As non-tuning diagnostics only, print the main/direct CE ratio, main/direct
argmax agreement, and cosine/norm ratios between their gradients for `fc`, the
pooled representation, final BN, one early convolution, and one stage-3
convolution. These measurements may abort only for nonfinite/degenerate values
or a semantic mismatch. They must not select the coefficient, cutoff, or any
other configuration, and must not inspect CIFAR-10 test labels or evaluator
metrics.

## Throughput Gate

Benchmark accepted and candidate complete training steps on the single idle H20
with identical cloned state, fixed synthetic tensors, and no data loader,
printing all raw windows before assertions. Include forward, all CE calls,
backward, Nesterov update, zeroing, and synchronization. Measure early mixup and
hard-label regimes separately using counterbalanced `A/C/C/A` ordering, warmup,
at least four retained windows per implementation/regime, and CUDA-event or
explicitly synchronized wall timing consistent with prior local preflights.

Require every implementation/regime coefficient of variation to be at most 5%
and compute whole-run projected retention from the accepted 65% mixup / 35%
hard weighting. The mandatory pre-score gate is

```text
weighted candidate throughput / weighted accepted throughput
    >= 127 / 130.304
    = 0.974643909627
```

equivalently at least 127 projected data passes from EXP036's accepted 130.304.
Also require finite loss/gradients, no memory growth across retained windows,
and peak allocation comfortably below device capacity. Do not repeat a stable
failed gate, relax 127, shorten the benchmark after observing it, or recover
through fusion that changes accepted main-loss numerics. A timing failure makes
the mechanism accuracy-unmeasured and closes this exact implementation on the
protected operating regime.

## Sole Score, Verification, and Closure

Only after every semantic and timing gate passes, remove stale `run.log` and run
exactly once:

```bash
timeout 600s uv run train.py > run.log 2>&1
```

Use fixed seed 42 on one NVIDIA H20, with no network, package install, remote,
reroll, restart, or extra validation. Require exit zero, one complete finite
summary, exactly 300 counted training seconds, less than 600 wall seconds, no
more than one evaluation per epoch, correct 65% transition ordering, frozen
evaluator/source scope, and the unchanged parameter count.

Primary success is `best_test_acc >= 94.58%`, exactly 0.10 points above the
accepted 94.48%. The protected-mechanism hypothesis additionally requires at
least 127 realized passes. `final_test_acc >= 94.45%` and
`final_test_loss <= 0.2456` are preregistered corroboration against a sparse
best-epoch accident, not alternate acceptance metrics.

The scored run is never repeated. If it completes below 127 realized passes,
it remains the valid fixed-seed goal result and must be reported, but it cannot
support the normal-exposure mechanism claim even if timing had projected 127;
do not rerun or tune. A valid >=127-pass miss falsifies only the proposition
that this exact always-on shared-classifier 90/10 auxiliary objective improves
the accepted learner. It closes immediate auxiliary-weight, duration, ramp,
detach, separate-classifier, consistency/distillation, head-scale, head-width,
seed, LR, decay, and momentum rescues after seeing the score.

A miss does not prove raw `z` is not linearly separable, does not isolate direct
supervision from the 10% reduction in pooled-head gradient, and does not reject
auxiliary supervision at an intermediate spatial stage or an independently
motivated loss family. Conversely, a gain justifies accepting the complete
objective; it does not justify a coefficient sweep without a new prospective
rationale.

## Risks and Expected Value

- **Head suppression**: only 90% of the accepted main CE reaches the residual
  MLP, possibly weakening the exact mechanism that produced EXP036's gain.
- **Shared-boundary conflict**: direct and refined features may demand different
  classifier rows; their shared `fc` gradient can compromise both rather than
  regularize either.
- **Backbone conflict**: if main/direct gradients oppose, the blend changes the
  entire end-to-end trajectory even though the auxiliary graph is post-pooling.
- **Redundancy**: because `z_refined` already contains `z`, the direct CE may add
  little information while still consuming launches and gradient budget.
- **Throughput**: the extra tiny GEMM and one/two extra CE calls are cheap in
  FLOPs but may be launch-bound; only measured full-step retention determines
  feasibility.
- **Interpretability**: the fixed coefficient is disciplined but heuristic.
  This is a one-point test of a coherent objective, not a calibrated optimum.

Expected evidence strength is moderate-low and potential impact moderate. The
candidate is attractive mainly because it is parameter-free at inference,
orthogonal to the closed classifier-radius/decay and tail-schedule treatments,
and tests a representation-level interaction at near-zero spatial cost. Its
local downside is concrete enough that it should lose to any candidate with a
more directly diagnosed mechanism, but it is rigorous enough for one score if
selected.

## Sources

- `01-definition.md`
- `02-system-understanding.md`
- `03-experiment-learnings.md`
- `experiments/028/01-brainstorm.md`
- `experiments/036/03-execute.md`
- `experiments/036/04-analysis.md`
- `experiments/039/proposals/idea-03.md`
- `experiments/040/04-analysis.md`
