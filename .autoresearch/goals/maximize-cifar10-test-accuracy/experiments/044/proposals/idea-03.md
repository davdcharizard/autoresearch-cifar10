# Proposal: Exact-Neutral Spatial-Dispersion Input to the Accepted Pooled MLP

## Recommendation

Advance only as a low-to-medium-confidence, one-score representation test.
Preserve accepted commit `a7c42dc` in full and add one fixed statistic branch:
the per-channel population standard deviation of the final `128 x 8 x 8`
post-BN/ReLU map enters the hidden preactivation of the accepted pooled
`128 -> 64 -> 128` residual MLP through a zero-initialized, bias-free
`128 -> 64` adapter. Ordinary GAP remains the untouched direct path.

This is the exact-neutral repair of the unscored EXP042 standard-deviation
proposal. That earlier proposal actively added an identity-projected
`0.1 * sigma` to the mean, conflating dispersion with an arbitrary startup
perturbation and an extra 128-dimensional residual. Here the new adapter starts
at zero, the accepted function and common first-step signal are preserved, and
the statistic can open through its own data gradient. No network, test-set
inspection, literature retrieval, or parameter sweep informed this choice.

Confidence must remain restrained. After final BatchNorm and ReLU, per-channel
mean and standard deviation are likely strongly correlated; the accepted MLP
may already infer most useful activation magnitude or occupancy from the mean.
The proposal tests one plausible missing invariant statistic, not a diagnosed
CIFAR-10 error mode.

## Exact Treatment

For final activations `X in R^(B x 128 x 8 x 8)`, with `S=64`, define

```text
mu[b,c]    = (1/S) * sum_s X[b,c,s]
var[b,c]   = (1/S) * sum_s (X[b,c,s] - mu[b,c])^2
sigma[b,c] = sqrt(var[b,c] + 1e-5)

a          = W_mean mu + D sigma
h          = W_out ReLU(a)
z          = mu + 0.1 h
logits     = fc(z)
```

`W_mean`, `W_out`, `fc`, and the `0.1` scale are the accepted parameters and
formula. Only `D in R^(64 x 128)` is new. Production should preserve the exact
accepted mean reduction and express the branch as:

```python
features = F.relu(self.bn(out))
pooled = F.adaptive_avg_pool2d(features, 1).flatten(1)
spatial_std = torch.sqrt(
    torch.var(features, dim=(-2, -1), correction=0) + 1e-5
)
hidden = self.pooled_head[0](pooled) + self.dispersion_adapter(spatial_std)
refined = pooled + POOLED_HEAD_SCALE * self.pooled_head[2](F.relu(hidden))
return self.fc(refined)
```

Use population variance because all 64 sites are the summarized feature map.
The fixed `1e-5` matches the existing BatchNorm numerical floor and prevents
singular square-root gradients for constant maps; it is a numerical choice,
not evidence of an optimal semantic scale. Do not use unbiased variance,
`E[X^2]-E[X]^2`, clamping, normalization, concatenation, max pooling, or a
learned spatial operation.

The new adapter adds exactly `128 * 64 = 8,192` parameters, raising the count
from `1,003,482` to `1,011,674`. It adds 8,192 dense MACs per image after the
fixed reduction. There is no learned compute on the `8 x 8` grid, spatial
weighting, query, softmax, positional state, or replacement of uniform GAP.

## Initialization, Gradient, RNG, and Optimizer Semantics

Register `dispersion_adapter = nn.Linear(128, 64, bias=False)` after the
accepted pooled head. Construct it inside a restoring CPU-only
`torch.random.fork_rng(devices=[])` and overwrite its weight with exact zeros.
Do not assign a seed: zero is deterministic and a seed would carry no retained
information. The fork isolates the constructor's discarded random draw and
must restore global CPU state; CPU-only construction must leave CUDA RNG exact.
Every accepted parameter and buffer must remain byte-identical.

At `D=0`, `D sigma=0`, so pooled features, hidden activations, refined features,
logits, loss, and BN evolution equal accepted. The dispersion path contributes
zero gradient to the backbone initially, while

```text
dL/dD = (dL/da)^T sigma
```

is generally nonzero, allowing the branch to open on update one. This is not a
delayed branch with a blocked new endpoint: the complete accepted GAP/MLP path
and its backbone gradient remain active. Require common gradients to match an
independently loaded accepted model within predeclared tight FP32 tolerances;
do not demand bitwise equality where the added zero node changes reduction
ordering.

The generic optimizer rule places `D` exactly once at the end of the rank-2
decay group. It receives accepted coupled weight decay `5e-4`, time-varying LR,
momentum `0.9`, and Nesterov, with no special LR, gain, cutoff, or no-decay
exception. At the zero start its first decay contribution is zero. Verify both
fresh and preseeded momentum updates against the accepted SGD formula. Forward
and statistic computation consume no CPU or CUDA RNG.

## Mechanism and Local Evidence

GAP retains mean channel presence but discards whether evidence is diffuse or
concentrated. `sigma` can distinguish maps with identical means, and injecting
it into the accepted 64-unit hidden layer permits mean/dispersion interaction
without adding a second output head or learned spatial processing.

- EXP036 is the positive placement evidence: the accepted scale-0.1 pooled
  MLP improved `best_test_acc` to 94.48% and loss to 0.2456 at 130.304 passes.
  This proposal preserves that head and adds only new input weights to its
  existing bottleneck.
- EXP041 retained 128.54 passes but direct-path auxiliary CE scored 94.26%.
  Therefore keep the sole refined-path CE; this proposal changes no objective,
  classifier invocation, or relative head gradient by construction.
- EXP042 retained 127.93 passes but centered content-attention pooling scored
  93.80%/0.2787. That treatment learned competitive spatial weights and changed
  the pooled vector away from uniform averaging. This proposal instead keeps
  `mu` exactly as the direct residual path and supplies a permutation-invariant
  fixed dispersion statistic only to the subordinate MLP. It is distinct from
  attention pooling, not a query/temperature/scale rescue of it.
- The system record says generalization and boundary quality are limiting,
  backward is about 74% of step time, and spatial learned work threatens the
  protected regime. A fixed reduction plus an 8k-MAC pooled adapter is a
  plausible fit, but kernel-launch and variance-backward cost require timing.

The leading counter-hypothesis is redundancy. For approximately standardized
pre-ReLU channels, ReLU mean and standard deviation co-vary; object occupancy,
amplitude, and sparsity may already be encoded by `mu` and remapped by
`W_mean`. In that case `D` adds correlated capacity, optimizer noise, and
reduction cost without useful boundary information. A gain also supports only
the complete statistic-plus-adapter treatment, not proof that spatial
dispersion was independently causal.

## Semantic Preflight

Use an ignored evaluator-free harness with an independent
`git show a7c42dc:train.py` oracle. It must block evaluator and test-data
construction. Print measurements before assertions and require:

1. Only `train.py` production scope changes: one epsilon constant if desired,
   adapter construction, and the final pooling/head block. Data, augmentation,
   schedule, loss, seed, cadence, and summary stay accepted.
2. All common state bytes and post-construction CPU/CUDA RNG states match;
   only zero `[64,128]` `dispersion_adapter.weight` is appended; total parameters
   are exactly `1,011,674`.
3. On fixed CPU/CUDA fixtures, production `mu`, population `sigma`, hidden,
   refined vector, and logits match an independent formula. At zero `D`,
   accepted logits are exact and common gradients match within declared bounds.
4. Constant channels produce finite `sqrt(1e-5)` values and finite gradients;
   equal-mean/different-variance fixtures produce equal `mu` and different
   `sigma`.
5. In early mixup and hard-label fixtures, `D` has finite nonzero data gradient
   and update, the initial feature gradient through the dispersion term is zero,
   and complete updates replay from restored state/RNG.
6. Every parameter occurs once in accepted optimizer group order, with only
   `D` appended to matrix decay; fresh/preseeded Nesterov updates match an
   independent oracle.
7. Temporal augmentation transitions, LR, finite-loss guard, once-per-epoch
   evaluation, and frozen evaluator contract remain accepted.

Report, but never gate or tune from, `cos(mu,sigma)`, per-channel correlations,
`||dL/dD||`, first-update `||D||`, and the resulting statistic-to-mean hidden
RMS ratio. A semantic failure closes the implementation; it does not authorize
changing epsilon, statistic, initialization, width, scale, or placement.

## Timing and Exposure Gate

On one idle H20, compare accepted and candidate complete production-equivalent
steps in early mixup and hard-label regimes. Include pinned H2D, LR writes,
zeroing, accepted mixing when active, full forward/loss/finite guard/backward,
coupled Nesterov update, and synchronization. Use at least 20 disposable
warmups and two counterbalanced `A/C/C/A` cycles, giving four retained windows
of at least 50 steps per arm and regime from restored deterministic fixtures.
Print all windows before assertions.

Using four-window medians, compute

```text
retention =
  (0.65 / candidate_mix_ms + 0.35 / candidate_hard_ms) /
  (0.65 / accepted_mix_ms  + 0.35 / accepted_hard_ms)
projected_passes = 130.304 * retention
```

Require every population CV `<=5%`, candidate peak allocation `<2,048 MiB`,
`retention >= 127/130.304 = 0.9746439096`, and projected passes `>=127.0`.
A stable miss ends the proposal before scoring. Do not rerun timing, fuse or
approximate variance, drop the square root, narrow the adapter, or relax the
floor as a rescue.

## Sole Scored Run and Decision Contract

After both gates pass, reconfirm baseline 94.48% at `a7c42dc`, one idle H20,
local CIFAR-10, frozen `prepare.py`, clean scope, and no stale log. Run exactly
one seed-42 score through the required frozen evaluator:

```bash
timeout 600s uv run train.py > run.log 2>&1
```

Require exit zero, one finite summary, `300.0-300.1` counted seconds, wall time
under 600 seconds, exactly `1,011,674` parameters, correct ordered transitions,
unique every-fifth plus final evaluations, and no runtime/integrity fault.
Realized exposure is `num_steps * 256 / 50000`.

The sole accuracy threshold is `best_test_acc >=94.58%`; final accuracy and
loss are descriptive only and cannot rescue or veto it. Mechanism success also
requires realized exposure `>=127` passes. A valid lower-exposure score still
counts as the only score and may not be rerun, but is not a successful
normal-exposure result. Never inspect intermediate test accuracy for control
flow, choose a variant from test behavior, or launch a second valid score.

## No-Rescue Closure

- A valid `>=127`-pass miss closes this exact standard-deviation-to-hidden
  branch and its immediate neighborhood. Do not try variance/RMS/max, another
  epsilon, width, bias, scale, identity/random/nonzero initialization,
  normalization, concatenation, output placement, decay/LR exception, cutoff,
  seed, or rerun.
- A normal-exposure success supports only this complete fixed treatment. It
  does not prove mean/std independence, object extent recovery, or superiority
  of second-order pooling generally, and it does not authorize a sweep.
- A timing failure closes systems viability of this exact implementation
  without an accuracy claim. An invalid run permits repair only of an
  independently demonstrated infrastructure or verifier defect, never a
  result-conditioned production change.

## Falsifiable Hypothesis

If per-channel spatial dispersion after final BN/ReLU contains useful boundary
information not recoverable from the accepted mean, then the exact-zero
`128 -> 64` dispersion adapter into the accepted pooled MLP will preserve at
least 127 passes and raise fixed-seed `best_test_acc` from 94.48% to at least
94.58%. The honest prior is modest because post-BN/ReLU mean and standard
deviation may be largely redundant.

## Local Sources

- `01-definition.md`, `02-system-understanding.md`,
  `03-experiment-learnings.md`, and `04-results.tsv`.
- `experiments/036/04-analysis.md`: accepted pooled residual MLP.
- `experiments/041/04-analysis.md`: failed direct-path auxiliary CE.
- `experiments/042/04-analysis.md`: failed exact-neutral content attention.
- `experiments/042/proposals/idea-02.md` and `experiments/042/01-idea-review.md`:
  unscored active-start standard-deviation proposal and its confound critique.
