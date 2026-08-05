# Proposal: Exact-Neutral Learned Content-Attention Pooling

## Recommendation

Replace only the final global average pooling operator in accepted `a7c42dc`
with a single-query content-attention pool over the final `8 x 8` feature map.
Use one bias-free `1 x 1` scoring convolution with 128 weights, initialize it
to exact zero, and express attention as a centered correction to the unchanged
accepted average. Preserve the exact accepted `128 -> 64 -> 128` residual MLP
head, classifier, objective, optimizer, temporal augmentation recipe, seed,
budget, and evaluator.

This is a spatial-selection hypothesis, not a rescue of the closed stage-3 SE
family. The scorer uses all 128 channels to assign one scalar weight to each
spatial site, then softmax competition reallocates a fixed unit mass across the
64 sites. It never creates per-channel residual gates, never modifies a
shortcut or residual branch, and never adds a second loss. The treatment starts
at the exact accepted function yet receives a generally nonzero gradient on
the first backward pass.

## Exact Pooling Definition

Let the accepted post-`layer3`, post-final-BN/ReLU feature map be
`X in R^(B x C x H x W)`, with `C=128`, `H=W=8`, and `S=H*W=64`. Flatten the
spatial coordinates and write `x_{b,s} in R^C`. Let the new bias-free scorer
weight be `q in R^C`. Define

```text
l_{b,s}     = q^T x_{b,s}
a_{b,s}     = exp(l_{b,s}) / sum_t exp(l_{b,t})
u           = 1 / S
mu_b        = (1/S) * sum_s x_{b,s}
delta_b     = sum_s (a_{b,s} - u) * x_{b,s}
z_b         = mu_b + delta_b
```

In real arithmetic,

```text
z_b = sum_s a_{b,s} * x_{b,s},
```

because `sum_s u*x_{b,s} = mu_b`. Thus the centered expression is exactly a
content-attention pool, not a residual mixture of average and attention. Its
purpose is operational identity at initialization: when `q=0`, every score is
zero, softmax is uniform, every centered coefficient is zero, and `z_b=mu_b`.

The forward should preserve the accepted average-pooling kernel and add only
the centered correction:

```python
spatial_features = out.flatten(2)
mean_pooled = F.adaptive_avg_pool2d(out, 1).flatten(1)
score_logits = self.pool_score(out).flatten(1)
attention = F.softmax(score_logits, dim=1)
uniform = 1.0 / score_logits.size(1)
attention_delta = attention - uniform
pooled_correction = torch.bmm(
    spatial_features, attention_delta.unsqueeze(2)
).squeeze(2)
out = mean_pooled + pooled_correction
out = out + POOLED_HEAD_SCALE * self.pooled_head(out)
return self.fc(out)
```

For CIFAR's `S=64`, `1/64` is exactly representable in FP32. The semantic gate
must nevertheless prove on CPU and CUDA that zero scores produce exactly the
same coefficient as `uniform`, that `pooled_correction` is bitwise zero, and
that pooled features and final logits are bitwise accepted. A direct
`sum(attention * features)` implementation is disallowed because its reduction
order need not match accepted `adaptive_avg_pool2d`, even though the formulas
are algebraically equal.

Do not add a temperature, `sqrt(C)` divisor, attention residual scale, learned
gain, positional parameter, scorer bias, multihead split, entropy loss, or
average/attention interpolation coefficient. The norm of `q` already controls
softmax sharpness through training; any extra scalar would create an
unsupported operating point.

## Construction, Initialization, and RNG

Register exactly one `nn.Conv2d(128, 1, kernel_size=1, bias=False)` after the
accepted model, classifier, and pooled head have been constructed and
initialized. Its sole `[1,128,1,1]` weight must be exact zero. Because the
standard `Conv2d` constructor samples a default initialization before it can be
zeroed, construct and zero it inside a restoring CPU RNG fork:

```python
with torch.random.fork_rng(devices=[]):
    self.pool_score = nn.Conv2d(widths[2], 1, 1, bias=False)
    init.zeros_(self.pool_score.weight)
```

No new initialization seed is needed: the only retained new tensor is all
zeros. The restored fork prevents the temporary default constructor draw from
advancing global CPU RNG, and CPU construction does not touch CUDA RNG. This
ordering preserves every accepted common parameter/buffer byte, including the
seed-36036 pooled head, and both post-construction global RNG states. The model
parameter count rises by exactly 128, from 1,003,482 to 1,003,610.

The scorer weight has rank four, so the accepted optimizer grouping places it
once in the matrix-decay group at `weight_decay=5e-4`. It uses the same LR,
momentum `0.9`, and Nesterov semantics as every other matrix. Coupled decay is
zero at initialization because `q=0`; after the scorer opens, decay pulls it
toward uniform pooling. Adding a special no-decay group would be a different
treatment and is disallowed.

## Why Zero Initialization Is Gradient-Open

Zero initialization does not create the delayed-opening failure of a
zero-terminal two-layer branch. Let `g_b = dL/dz_b`. Softmax differentiation
gives

```text
d a_{b,s} / d l_{b,t} = a_{b,s} * (1[s=t] - a_{b,t}).
```

At `q=0`, `a_{b,s}=1/S` and `z_b=mu_b`. The scorer gradient for one example is

```text
dL/dq
  = sum_s (1/S) * x_{b,s} * ((x_{b,s} - mu_b)^T g_b)
  = [(1/S) * sum_s x_{b,s} x_{b,s}^T - mu_b mu_b^T] * g_b
  = Cov_spatial(X_b) * g_b.
```

The batch gradient is the sum of these covariance-vector products. It is
nonzero whenever the downstream pooled-feature gradient has a component in the
spatial covariance range and examples do not cancel exactly. Final WRN feature
maps are spatially nonconstant, while the accepted pooled head and classifier
supply nonzero downstream gradients, so a nonzero first-step scorer gradient
is expected and must be measured under both fixed early-mixup and hard-label
fixtures before timing.

At the same zero state, the feature-map derivative remains the accepted one:
the direct derivative of centered coefficients is zero because
`a_{b,s}-u=0`, while the derivative of attention with respect to a feature
contains the factor `q=0`. Therefore

```text
dL/dx_{b,s} = g_b / S
```

on the first backward pass, exactly the global-average-pooling derivative in
real arithmetic. The scorer can move on optimizer step one without initially
perturbing the backbone, pooled head, or classifier training signal. After the
first update, attention becomes content-dependent and all gradients are
allowed to diverge from accepted.

The semantic harness should verify this formula in float64 against autograd,
including the batch sum, and report spatial covariance rank/norm, downstream
gradient norm, analytic scorer-gradient norm, and maximum analytic/autograd
error. These are implementation diagnostics only, not tuning signals.

## Mechanistic Rationale and Distinction From Prior Gates

The accepted head shows that channel co-occurrences in the globally pooled
representation contain useful nonlinear structure: EXP036 improved best
accuracy from 94.32% to 94.48% and test loss to 0.2456 while adding only
post-pooling work. That does not prove average pooling is deficient, but it
makes the endpoint representation a more defensible intervention site than a
new high-resolution residual block.

Content pooling asks a different question from EXP017-025:

- EXP017/018/019/024/025 modified signed residual branches inside stage 3 and
  learned 128 separate channel scales at each selected block. This candidate
  leaves every residual branch and shortcut untouched.
- SE first averaged spatial content, then used a dense `128 -> 8 -> 128` MLP
  to emit channel gates and multiplied those gates across all positions. This
  scorer uses one dense 128-channel dot product at each position, emits one
  scalar per position, and preserves all channels inside the selected pooled
  vector.
- The candidate retains global cross-channel evidence in the scoring dot
  product and adds cross-position competition through softmax, but it does not
  attempt the dense channel-output interaction whose simplified SE substitutes
  failed. It is spatial routing rather than channel gating.
- The mechanism is applied once after the complete accepted backbone, before
  the successful pooled residual MLP. It cannot attenuate individual residual
  branches or reproduce the two-gate interaction observed in EXP017.
- Unlike the failed exact-neutral SE two-layer path, the single zero scorer has
  a direct first-step covariance gradient. There is no blocked upstream scorer
  layer and no bias-only opening phase.

Potential benefit is input-adaptive suppression of background or uninformative
sites while preserving translation equivariance: the same content scorer is
shared across all positions and has no positional embeddings. This may give
the accepted pooled head a cleaner summary of channel co-occurrences at very
low parameter cost.

The risks are equally concrete. CIFAR-10 objects occupy much of a `32 x 32`
image, and uniform averaging may be a beneficial invariance rather than a
bottleneck. A shared scalar weight cannot select different spatial regions for
different output channels. Softmax may become overly concentrated, discard
distributed evidence, or overfit augmentation artifacts. The scorer begins
with no preferred direction and its batch-summed covariance gradients can
partially cancel. Exact initial neutrality protects attribution but supplies no
local evidence that learned spatial selection is beneficial.

## Compute and Exposure Risk

The new scorer performs `C*S = 8,192` multiply-accumulates per image, followed
by a 64-way softmax and roughly another 8,192 weighted-correction operations.
It adds only 128 parameters and negligible memory relative to the 1.0M-parameter
backbone. Arithmetic is far below one 8x8 convolutional block and comparable
to the accepted 16,384-MAC pooled MLP.

However, the operation remains spatial and its backward pass materializes
score, softmax, and weighted-reduction work. Several small CUDA kernels can be
launch-bound, so arithmetic counts are not sufficient. Accepted exposure is
130.304 passes; the protected floor of 127 permits only
`1 - 127/130.304 = 2.5356%` throughput loss. Qualification must use complete
production-equivalent steps rather than an isolated forward estimate.

No loader timing is needed because data workers, transforms, batch size, and
augmentation are source-identical. Candidate peak allocation must remain below
2,048 MiB; accepted scored peak is about 1,096.4 MiB and memory is not expected
to bind.

## Semantic Preflight

Use an ignored evaluator-free harness with an independently compiled exact
`git show a7c42dc:train.py` oracle. Before timing or scoring, require:

- only tracked production `train.py` differs; `prepare.py`, evaluator behavior,
  root Python scope, and every other production file remain frozen;
- all accepted common state-dict keys, shapes, dtypes, bytes, registration,
  post-construction CPU/CUDA RNG, constants, schedule, transforms, temporal
  gates, losses, evaluation cadence, and summary contract remain exact;
- exactly one new `[1,128,1,1]` bias-free scorer weight exists, is exact zero,
  consumes no persistent buffer/RNG, and gives 1,003,610 total parameters;
- the scorer appears exactly once in the accepted rank-at-least-two decay group
  at `5e-4`, with unchanged group options/order for all common parameters;
- attention softmax is only over each example's flattened spatial axis, sums to
  one, never mixes examples, and has no channel or positional bias;
- at zero scorer, scores are bitwise zero, weights are bitwise `1/64`, centered
  weights and correction are bitwise zero, and pooled features, pooled-head
  output, logits, loss, BN-buffer evolution, CPU/CUDA RNG, and common parameter
  gradients match accepted on CPU/CUDA fixtures;
- on fixed nonzero scorer fixtures, production centered pooling matches an
  independent direct weighted sum within dtype-appropriate tolerance, remains
  finite, and changes outputs nontrivially;
- float64 autograd and the covariance formula above agree for scorer gradients;
  fixed early-mixup and hard-label full-model fixtures produce finite nonzero
  first-step scorer gradients without relying on decay;
- at zero scorer, feature-map and all common-parameter gradients match the
  accepted average-pooling path within declared FP32 reduction tolerances;
  after one independently computed update, scorer weights become finite and
  nonzero and attention becomes nonuniform on the fixed fixture;
- fresh and deterministic preseeded-momentum complete updates for every tensor
  and momentum buffer match an independent coupled-Nesterov oracle; scorer
  update uses `grad + 5e-4*q`, with the decay term exactly zero only on its
  first step;
- restored model/optimizer/input/RNG states replay both early-mixup and
  hard-label steps deterministically, with no additional data randomness or
  control-state mutation.

Print all invariant/gradient/update errors before assertions. A semantic
failure must not be repaired by nonzero scorer initialization, a temperature,
detachment, an attention residual scale, no-decay allocation, or direct
weighted pooling that loses exact accepted initialization. Those changes are
different treatments.

## Throughput and Exposure Gate

Run matched accepted/candidate production-equivalent complete steps on the
idle local H20 for both early mixup and hard-label regimes. Include pinned H2D,
LR calculation and group writes, zeroing, mixup when active, full
attention/pooling/head/classifier forward, loss and finite guard, backward,
coupled Nesterov update, and final synchronization. Use at least 20 disposable
warmups, then two complete `A/C/C/A` cycles with windows of at least 50 steps,
giving four accepted and four candidate measurements per regime from restored
equivalent fixtures.

Print every window, medians, population CVs, peak allocation, retention, and
projection before assertions. Require every CV no greater than 5%, candidate
peak below 2,048 MiB, and

```text
retention =
    (0.65 / candidate_early_ms + 0.35 / candidate_hard_ms) /
    (0.65 / accepted_early_ms  + 0.35 / accepted_hard_ms)

retention >= 127 / 130.304 = 0.9746439096
projected_passes = 130.304 * retention >= 127
```

A stable miss ends the experiment before scoring. Do not rerun timing, lower
the floor, remove exact-neutral correction, cache attention across batches,
or fuse it into a different operator after observing the result.

## Sole Scored Run and Decision Contract

After all gates pass, reconfirm baseline 94.48% at `a7c42dc`, threshold 94.58%,
one idle NVIDIA H20, local CIFAR-10, frozen evaluator, exact source scope, and
no stale `run.log`. Execute exactly once at fixed seed 42:

```bash
timeout 600s uv run train.py > run.log 2>&1
```

Require exit zero, one finite summary, 300.0-300.1 counted training seconds,
wall time below 600 seconds, 1,003,610 parameters, exactly one mixup transition
near 195 counted seconds, the later exhausted-iterator RandAugment transition,
unique every-fifth plus final-partial-epoch evaluations, and no traceback,
OOM, worker, evaluator, or non-finite signature. Record realized passes as
`num_steps * 256 / 50000`.

Primary success is only `best_test_acc >=94.58%`, exactly 0.10 points above the
accepted 94.48%. Pre-register `final_test_acc >=94.45%` and
`final_test_loss <=0.2456` as non-decisive corroboration. They cannot rescue a
primary miss, and a primary success without them should be reported as fragile.
Never rerun a valid score.

## Interpretation and Closure

**Normal-exposure success:** At least 127 realized passes and 94.58% supports
the complete claim that one zero-started, globally shared, bias-free content
query can improve the fixed-seed accepted endpoint by spatially reweighting its
final features before the pooled residual head. It does not prove that GAP is
generally inferior, that learned weights focus on objects, or that SE/channel
attention should be reopened. Preserve the exact candidate without a
temperature, head, scorer-init, entropy, or query-count sweep.

**Normal-exposure miss:** At least 127 passes below 94.58% falsifies this exact
single-query, always-on, temperature-one, zero-initialized centered-softmax
pool as a useful standalone refinement. Restore accepted GAP. Close immediate
result-conditioned rescues: nonzero scorer seeds, scorer bias, learned/fixed
temperature, attention residual scales, entropy penalties, cutoff schedules,
no-decay allocation, multiple queries, or moving the scorer into stage-3
branches. The result does not formally close independently motivated GeM,
second-order pooling, class-specific pooling, or other pooling families, but
they require new evidence rather than adjacency to this miss.

**Low exposure or pre-score failure:** A semantic/timing failure provides no
accuracy evidence and rejects only this exact implementation's systems or
identity contract. A valid score below 127 still counts and cannot be rerun,
but the spatial-selection mechanism is operationally inconclusive because it
left the protected exposure regime.

## Falsifiable Hypothesis

If uniform global averaging discards useful content-localization information
from the accepted final WRN feature map, then an exact-neutral single-query
content-attention pool whose zero scorer receives the first-step gradient
`sum_b Cov_spatial(X_b) * dL/dz_b` will retain at least 127 projected and
realized passes and raise fixed-seed `best_test_acc` from 94.48% to at least
94.58%, with final accuracy at least 94.45% and test loss no worse than 0.2456
as corroboration.

A valid normal-exposure miss falsifies only this exact centered-softmax spatial
pool and its immediate scalar/init/cutoff rescues, not all pooling mechanisms.

## Local Evidence

- `experiments/036/04-analysis.md`: the accepted `128 -> 64 -> 128` pooled
  residual head improved best/final accuracy and loss at 130.304 passes,
  supporting the endpoint representation as a compute-efficient intervention
  location while leaving the exact head protected.
- `experiments/017/04-analysis.md` through `experiments/025/04-analysis.md`:
  full stage-3 channel SE had a small signal but its gate-removal, static,
  diagonal, width-composed, and diagnostic-free follow-ups failed or missed
  feasibility. This closes that branch-gating family and motivates a formally
  distinct spatial selector, not another SE simplification.
- `experiments/040/04-analysis.md`: constraining classifier row radii lost 0.57
  points at normal exposure; preserve ordinary affine boundary freedom.
- `experiments/041/04-analysis.md`: auxiliary raw-path supervision lost 0.22
  points at normal exposure; preserve sole refined-path CE and change only the
  representation supplied to it.
- `02-system-understanding.md`: forward/backward compute and boundary quality
  are binding, memory and I/O are not, and new spatial work must protect the
  127-pass floor.
- `03-experiment-learnings.md`: preserve the accepted pooled head, classifier,
  decay, global cosine, temporal mixup/RandAugment recipe, and full backbone;
  prioritize a genuinely orthogonal mechanism over immediate rescue tuning.
