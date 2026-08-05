# Proposal: Default SiLU in the Accepted Pooled Residual MLP

## Recommendation

Advance one exact, low-confidence activation experiment: starting from accepted
commit `a7c42dc`, replace only the state-free `nn.ReLU()` inside the bias-free
`128 -> 64 -> 128` pooled residual MLP with default `nn.SiLU()`. Preserve both
matrix tensors byte-for-byte, their isolated seed-36036 initialization, the
residual scale `0.1`, ordinary global average pooling, direct pooled path,
classifier, spatial backbone, sole cross-entropy objective, optimizer, input
pipeline, temporal controls, seed, time budget, and evaluator cadence.

This is deliberately a one-line active-function change, not an exact-neutral
adapter. SiLU changes initial logits and every data gradient whenever the
accepted hidden preactivation is nonzero. Its mechanism is plausible: the
first bias-free matrix forms signed contrasts from nonnegative pooled features,
and ReLU discards negative contrasts and their local derivative, whereas SiLU
retains a bounded negative lobe and a smooth derivative around zero. The prior
remains low because SiLU also attenuates all finite positive activations,
changes the ReLU-calibrated branch amplitude, and is adjacent tuning of the
only head already shown to work.

One strictly qualified score is warranted. No activation comparison, scale
compensation, initialization adjustment, or post-result rescue is warranted.

## Exact Production Change

The complete intended production diff is:

```diff
 self.pooled_head = nn.Sequential(
     nn.Linear(widths[2], POOLED_HEAD_WIDTH, bias=False),
-    nn.ReLU(),
+    nn.SiLU(),
     nn.Linear(POOLED_HEAD_WIDTH, widths[2], bias=False),
 )
```

Use the constructor exactly as `nn.SiLU()`, whose installed default is
`inplace=False`. Do not change any backbone `F.relu`, the final pre-pooling
`F.relu`, or any other activation. Do not use functional SiLU, inplace mode,
GELU, a gain, bias, normalization, learned beta, dropout, a temporal gate, or
an unrolled head. Keep `POOLED_HEAD_WIDTH=64`, `POOLED_HEAD_SCALE=0.1`, and
`POOLED_HEAD_INIT_SEED=36036` unchanged.

Let `z in R^(B x 128)` be the nonnegative accepted final-BN/ReLU/GAP vector,
`W1 in R^(64 x 128)` and `W2 in R^(128 x 64)` the accepted head matrices,
`C` the accepted classifier, `a=W1 z`, and `s=0.1`. Then:

```text
accepted_hidden  = ReLU(a)
candidate_hidden = SiLU(a) = a * sigmoid(a)

accepted_refined  = z + s * W2 * ReLU(a)
candidate_refined = z + s * W2 * SiLU(a)

candidate_logits - accepted_logits
    = s * C * W2 * [SiLU(a) - ReLU(a)].
```

This changes no matrix MACs, tensor dimensions, trainable tensor, optimizer
membership, or state-dict key. Total parameters remain exactly `1,003,482`.

## Signed and Smooth Mechanism

Although `z` is nonnegative, `W1` has signed weights, so negative `a_j` can
encode a pooled channel contrast opposed to hidden feature `j`. ReLU maps that
entire half-space to zero and gives `W1` no data gradient through it. SiLU maps
moderately negative inputs to bounded negative responses and supplies a smooth
derivative near zero. In a narrow 64-unit bottleneck, signed hidden responses
could express opposing channel combinations without allocating paired ReLU
units, while dense near-zero gradients could improve how the branch discovers
useful co-occurrences under early mixup/RandAugment and the clean tail.

That account must not be overstated. For scalar `x`:

```text
SiLU(x)  = x * sigmoid(x)
SiLU'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)).
```

- For `x<0`, SiLU is negative while ReLU is zero, but SiLU approaches zero
  again as `x -> -infinity`; it does not preserve an unrestricted negative
  linear channel.
- SiLU is nonmonotone in its far-negative region, where its derivative can be
  negative. “Dense smooth gradient” therefore does not mean an everywhere
  positive gradient.
- At `x=0`, both activations output zero, but installed PyTorch ReLU uses
  derivative zero while SiLU has derivative `0.5`.
- For every finite `x>0`, `0 < SiLU(x) < ReLU(x)`. At `x=1`, SiLU is about
  `0.7311`, so the treatment attenuates positive evidence as well as adding a
  negative lobe.
- Consequently `SiLU(x)-ReLU(x) < 0` for every nonzero `x`. The signed effect
  after `W2` and `C` can point either way because those matrices are signed,
  but the hidden-coordinate delta itself is always nonpositive.

A gain or loss belongs to this complete signed, smooth, nonmonotone,
positive-attenuating function. It cannot establish that negative evidence or
smoothness alone was causal.

## Counter-Hypothesis and Local Evidence

The accepted branch may already be sufficient. Signed columns of `W2` let a
positive ReLU unit make positive or negative residual corrections, while the
direct path preserves every pooled channel. A negative `W1 z` may correctly
mean absence rather than inhibitory evidence. ReLU's sparse gate may protect
specialization; SiLU can couple weak responses across all 64 units, shrink the
branch norm under weights initialized for the accepted ReLU operating point,
and feed small noisy gradients into `W1`.

The local record argues for caution:

- **EXP036 is the sole positive head evidence.** The exact scale-0.1 ReLU MLP
  raised best accuracy from 94.32% to 94.48%, improved final loss to 0.2456,
  and retained 130.304 passes. It validates cheap nonlinear capacity at this
  placement, not an activation trend; its report explicitly declines adjacent
  activation and scale tuning.
- **EXP041 changed supervision around the same representation** and scored
  94.26%/0.2529 at 128.538 passes despite 0.976-0.989 sampled gradient
  cosines. Preserve sole refined-path CE.
- **EXP042 changed pooling with an exact-neutral content query** and scored
  93.80%/0.2787 at 127.933 passes. Preserve ordinary uniform GAP.
- **EXP043 projected convolution gradients** and scored 93.88%/0.2661 at
  129.807 passes. SiLU does not post-process or delete raw convolution
  gradients, though its changed forward function necessarily changes the
  gradients induced by the objective.
- **EXP044 added exact-neutral spatial dispersion to the head** and scored
  93.95%/0.2637 at 128.712 passes; mean/std correlations of 0.835-0.854 support
  preserving only the accepted pooled mean input.
- **EXP045 removed shortcut phase selection** and scored 94.11%/0.2512 at
  129.101 passes. Another affordable spatial-invariance treatment missed,
  reinforcing the system diagnosis that a prospective experiment should add
  no spatial work.
- **EXP046 changed CPU crop fill but was unscored** after a preregistered
  delay-free loader gate failed despite stable production-paced overlap. It is
  no accuracy evidence for or against SiLU, but supports returning to a
  GPU-local change with no worker-delivery semantics.

Thus SiLU is cleaner than the failed head-adjacent treatments because it adds
no feature, parameter, objective, classifier call, spatial selection, or data
path. It is also less independently motivated: it is a local substitution in
the accepted successful head and should be presented with an honest low prior.

## Initialization, State, and RNG Contract

Construction must remain byte-exact. The accepted model, final classifier,
and all backbone modules are initialized first. The pooled head remains inside
the same restoring CPU-only RNG fork:

```python
with torch.random.fork_rng(devices=[]):
    torch.random.default_generator.manual_seed(POOLED_HEAD_INIT_SEED)
    self.pooled_head = nn.Sequential(
        nn.Linear(widths[2], POOLED_HEAD_WIDTH, bias=False),
        nn.SiLU(),
        nn.Linear(POOLED_HEAD_WIDTH, widths[2], bias=False),
    )
    init.kaiming_normal_(self.pooled_head[0].weight)
    init.kaiming_normal_(self.pooled_head[2].weight)
```

`nn.SiLU()` has no parameter, buffer, random initialization, or forward/
backward RNG. The two linears remain at indices `0` and `2`, so parameter
names, traversal order, shapes, matrix-decay membership, and optimizer order
must remain accepted. Both matrices must receive the same seed-36036 Kaiming
draws in the same order, and every named parameter/buffer byte plus the global
post-construction CPU/CUDA RNG states must match `a7c42dc`.

Do not derive a “SiLU gain,” rescale either matrix, compensate branch RMS, or
zero-initialize an endpoint. The accepted weight bytes were prospectively
fixed before choosing the activation. Any initial branch-amplitude reduction
is part of the treatment and a descriptive diagnostic, not a defect to repair.

The data RNG must also remain accepted: crop/flip and worker-private
RandAugment states, loader sampling, batch-shared beta draw, mixup permutation,
transition timing logic, and clean-tail behavior are untouched. From cloned
states, accepted and candidate forwards/backwards must end with identical RNG
states even though their logits, losses, gradients, and updates differ.

## Fail-Closed Semantic Preflight

Use one ignored evaluator-free harness with an independent module loaded from
`git show a7c42dc:train.py`. Block `Eval` invocation and any CIFAR-10
`train=False` construction before importing accepted or candidate code. Emit
all diagnostics before assertions and require:

1. A source/AST audit proves that the sole production change is the pooled
   head's `nn.ReLU()` constructor becoming `nn.SiLU()`. Require frozen
   `prepare.py`, unchanged imports/constants/forward/training/evaluation code,
   successful compilation, and no stale score log.
2. Independently restored seed-42 construction yields exactly the accepted
   state-dict keys, parameter/buffer names, orders, shapes, dtypes, strides,
   and bytes; identical post-construction CPU/CUDA RNG; 52 trainable tensors;
   and `1,003,482` parameters. Independently reconstruct both seed-36036
   matrices and require byte equality.
3. Topology is exactly `Linear(128,64,bias=False) -> SiLU(inplace=False) ->
   Linear(64,128,bias=False)`. Require no activation state, no GELU/functional
   alias, and accepted direct path, scale `0.1`, GAP, classifier, and all
   backbone ReLUs.
4. On deterministic FP64 CPU scalar/vector fixtures spanning far-negative,
   moderate-negative, zero, near-zero, and positive inputs, match production
   activation and autograd against independent sigmoid formulas. Verify output
   signs, derivative `0.5` at zero, at least one negative-derivative point,
   strict positive attenuation, finite values, and no RNG advance.
5. On fixed FP32 CPU and CUDA model fixtures, capture production `z`, `a`,
   hidden output, residual correction, refined vector, and logits. Match SiLU
   to `a*sigmoid(a)` and candidate-minus-accepted logits to
   `0.1*C*W2*(SiLU(a)-ReLU(a))` under preregistered reduction tolerances.
   Require finite nonzero functional change and both positive and negative
   preactivation populations.
6. For fixed early-mixup and hard-label fixtures, independently reconstruct
   CE, all gradients, and fresh plus preseeded coupled-decay Nesterov updates.
   Require finite nonzero backbone/classifier/`W1`/`W2` gradients and updates,
   accepted two optimizer groups/options/order, unchanged decay semantics, no
   activation optimizer state, and identical terminal RNG. Candidate gradients
   are expected to differ at step one; do not impose common-gradient identity.
7. Reconfirm batch 256, batch-shared alpha-0.2 mixup and permutation semantics,
   both 65% controls, worker-isolated early RandAugment and exhausted-iterator
   cutoff, clean tail, accepted time-based LR/floor, one backward/step, finite
   guard, once-per-epoch evaluation, and every-fifth plus final cadence.

Report the following training-only diagnostics on fixed preregistered fixtures,
but never gate, tune, or select a variant from them:

- preactivation mean/std/quantiles and fractions `<0`, `=0`, near zero,
  positive, and in SiLU's negative-derivative region;
- ReLU and SiLU hidden mean/RMS/nonzero fraction;
- separate RMS of the negative-lobe term
  `SiLU(a) * 1[a<0]` and positive-attenuation term
  `(SiLU(a)-ReLU(a)) * 1[a>0]`;
- their separate and combined `W2` correction/logit RMS, sign balance, and
  cosine against the accepted branch/logits;
- accepted/candidate residual-to-direct norm ratio, logit delta, loss delta,
  and grouped gradient norm ratios/cosines.

These diagnose whether the proposed hidden regions are populated and quantify
the complete amplitude change. They must not authorize a gain, alternate
activation, new initialization, or score cancellation. A semantic failure
closes before timing unless an independently demonstrated harness or literal
one-line implementation defect can be repaired without changing treatment.

## Throughput and Exposure Gate

The candidate evaluates SiLU over only `256 * 64 = 16,384` hidden scalars per
batch. It adds sigmoid/multiply work relative to a ReLU mask but no matrix MAC,
spatial operation, data-loader work, tensor shape, state, or allocation of
practical scale. The accepted head accounted for about 1.4% of forward time,
so material exposure loss is unlikely; H20 eager forward/backward behavior
must nevertheless be measured rather than inferred.

On one idle H20, compare complete accepted and candidate production-equivalent
steps in both early-mixup and hard-label regimes. Include pinned H2D, LR writes,
zeroing, accepted beta/permutation/mixing when early, full forward, CE, finite
check, backward, coupled Nesterov step, and synchronization. Use identical
preregistered input/target/model/optimizer/RNG bytes for every arm, at least 20
disposable warmups per arm/regime, one live GPU arm at a time, and reset peak
allocation around each candidate window.

Repeat this exact eight-window block twice:

```text
AE, CE, AH, CH, CH, AH, CE, AE
```

where `A/C` denote accepted/candidate and `E/H` early-mixup/hard. Each retained
window contains at least 50 synchronized complete steps. In each block form
two local paired treatments from positions `(AE0,CE1,AH2,CH3)` and
`(AE7,CE6,AH5,CH4)`, yielding four combined retentions total:

```text
retention_i =
  (0.65 / candidate_early_i + 0.35 / candidate_hard_i)
  / (0.65 / accepted_early_i + 0.35 / accepted_hard_i)

projected_passes_i = 130.304 * retention_i.
```

Print all 16 windows, per-window step distributions/CVs, paired ratios,
retentions, projections, and allocation before asserting. Require every window
population CV `<=5%`, early and hard paired-ratio population CVs `<=1%`, every
`retention_i >= 127/130.304 = 0.9746439096`, median projected passes `>=127`,
candidate peak allocation `<2,048 MiB`, and finite updates. A stable miss ends
systems viability without a score and may not be rerun, compiled/fused only
for the candidate, converted inplace, rescaled, or replaced with GELU.

## Sole Score and Decision Contract

After all gates pass, reconfirm accepted baseline 94.48% at `a7c42dc`, one idle
NVIDIA H20, local CIFAR-10, frozen `prepare.py`/evaluator, exact `train.py`-only
scope, and no stale `run.log`. Launch exactly once:

```bash
timeout 600s uv run train.py > run.log 2>&1
```

Require exit zero, exactly one finite summary, `300.0-300.1` counted seconds,
wall time below 600 seconds, exactly 1,003,482 parameters, correct one-way
mixup and exhausted-iterator RandAugment transitions, unique every-fifth plus
final evaluations, and no traceback/OOM/worker/non-finite signature. Compute
realized exposure as `num_steps * 256 / 50000`.

Success requires both `best_test_acc >=94.58%` and realized exposure `>=127`
passes. Accepted final accuracy 94.45% and loss 0.2456 are descriptive
corroboration only; they cannot rescue or veto the primary result. A completed
score below 127 passes still consumes the sole run and cannot be rerun. Never
use intermediate evaluator results, training diagnostics, or endpoint metrics
to choose an activation, gain, scale, or second score.

## Strict Closure

- A valid `>=127`-pass score below 94.58% closes this exact SiLU replacement
  and, as search policy rather than experimental proof, its immediate
  smooth/signed activation neighborhood. Do not try GELU, approximate GELU,
  Swish-beta, Mish, ELU, leaky ReLU, another activation, width, bias, head
  scale, gain compensation, normalization, initialization, seed, decay/LR
  exception, temporal cutoff, or rerun.
- A score at or above 94.58% but below 127 passes is not success and does not
  authorize a speed rescue, compiler change, inplace mode, alternative
  activation, or repeat.
- A normal-exposure success supports only the complete fixed default-SiLU
  treatment. It does not prove the negative lobe or smoothness was causal, or
  that SiLU generally dominates another activation, and does not authorize a
  sweep.
- A stable pre-score timing failure closes exact-treatment systems viability
  without an accuracy claim. Repair after an invalid score is allowed only for
  an independently proven infrastructure/verifier fault while leaving the
  one-line production treatment byte-for-byte unchanged.

## Risks

- **Accuracy risk, high:** the only successful head uses exact ReLU, and four
  later head/readout changes regressed at normal exposure.
- **Attribution risk, high:** signed negative response, near-zero smoothness,
  positive attenuation, far-negative nonmonotonicity, and changed branch norm
  occur together and cannot be causally separated by one score.
- **Optimization risk, medium:** accepted Kaiming bytes and scale were selected
  with ReLU in place; preserving state deliberately forbids compensation.
- **Throughput risk, low but gated:** sigmoid backward is costlier than a ReLU
  mask, but it touches only 16,384 pooled scalars per batch.
- **Search risk, controlled:** one prospective activation, one fixed seed, one
  score, non-gating diagnostics, and strict closure prevent an activation
  sweep or endpoint-driven rescue.

## Falsifiable Hypothesis and Sources

If moderate negative preactivations or near-zero gradients in the accepted
64-unit pooled bottleneck contain useful class-boundary information that hard
ReLU suppresses, then replacing only that ReLU with default non-inplace SiLU,
while preserving every parameter byte, initialization, scale, classifier, and
spatial operation, will retain at least 127 passes and raise fixed-seed
`best_test_acc` from 94.48% to at least 94.58%. A valid normal-exposure miss
falsifies that complete exact-treatment claim, while neither outcome isolates
negative evidence from positive attenuation.

Offline sources: accepted `train.py` and `a7c42dc:train.py`; goal
`01-definition.md`, `02-system-understanding.md`,
`03-experiment-learnings.md`, and `04-results.tsv`; installed PyTorch SiLU
formula/semantics; EXP036 accepted head report; EXP041-046 reports; and the
prior EXP045 SiLU proposal `proposals/idea-03.md`. No network, test data,
evaluator, GitHub, or remote source was used.
