# Proposal: Signed Smooth SiLU in the Accepted Pooled MLP

## Recommendation

Advance only as a low-confidence, one-score activation test. Starting from
accepted commit `a7c42dc`, replace exactly the state-free hidden `nn.ReLU()` in
the bias-free `128 -> 64 -> 128` pooled residual MLP with default
`nn.SiLU(inplace=False)`. Preserve the two matrix tensors byte-for-byte, their
isolated seed-36036 Kaiming initialization, residual scale `0.1`, ordinary GAP,
classifier, spatial backbone, objective, optimizer, data path, schedule, and
all temporal controls. The production diff should be one module-constructor
replacement and nothing else.

This is intentionally not an exact-neutral treatment. At the same accepted
weights, SiLU changes the model function, logits, and training gradients on the
first forward pass. Its rationale is that the bias-free first pooled-head
matrix forms signed channel contrasts from nonnegative post-BN/ReLU GAP
features, while ReLU deletes every negative hidden contrast and its local
gradient. SiLU can retain graded, moderately negative evidence and keep a
smooth gradient around zero. That mechanism is plausible for a narrow
64-dimensional head, but there is no measured dead-unit or class-error
diagnosis showing it is needed.

The evidence is therefore weak and adjacent to successful-head tuning.
EXP036 established the placement and exact ReLU operating point, not an
activation trend, and explicitly cautioned that nearby activation choices were
unjustified by one success. EXP041, EXP042, and EXP044 then changed head-adjacent
supervision or representation at normal exposure and all regressed. This
proposal is cleaner than those treatments because it adds no parameters,
spatial statistic, classifier call, objective, or readout path, but it is still
a local activation substitution rather than an independently supported
orthogonal mechanism. No network, test-set evidence, literature lookup, or
activation sweep informed the choice.

## Prospective Choice: SiLU, Not GELU

Both candidates are state-free, smooth, and signed for negative inputs:

```text
SiLU(x) = x * sigmoid(x)
GELU(x) = x * Phi(x)                 # exact form
```

Choose exactly SiLU before any semantic measurement or score. For moderate
negative preactivations it retains a larger-magnitude negative lobe than GELU
(for example, at `x=-1`, SiLU is about `-0.2689` while exact GELU is about
`-0.1587`), making it the more direct test of the stated negative-evidence
hypothesis. It also avoids choosing between GELU's exact and tanh-approximate
forms and is a simple single pointwise primitive. GELU is rejected prospectively
and must not be run if SiLU fails.

This comparison does not claim SiLU is a pure negative-evidence intervention.
At `x=1`, SiLU is about `0.7311` rather than ReLU's `1`, so it also attenuates
positive evidence. At `x=0`, both outputs are zero, but SiLU has derivative
`0.5` while PyTorch ReLU uses zero derivative at the kink. As `x` becomes very
negative, SiLU approaches zero from below rather than preserving an unrestricted
negative linear tail. A gain or loss therefore belongs to the complete signed,
smooth, amplitude-changing activation, not uniquely to negative evidence.

## Exact Treatment and Startup Change

Let `z in R^(B x 128)` be the accepted nonnegative vector from final
BN/ReLU/GAP, `W1 in R^(64 x 128)` and `W2 in R^(128 x 64)` the accepted pooled
head matrices, `C` the accepted classifier, and `s=0.1`. The accepted and
candidate functions are

```text
a                 = W1 z
accepted_refined  = z + s * W2 ReLU(a)
candidate_refined = z + s * W2 (a * sigmoid(a))
accepted_logits   = C accepted_refined
candidate_logits  = C candidate_refined

delta_logits = s * C W2 [SiLU(a) - ReLU(a)]
```

The only production change is

```diff
 self.pooled_head = nn.Sequential(
     nn.Linear(widths[2], POOLED_HEAD_WIDTH, bias=False),
-    nn.ReLU(),
+    nn.SiLU(),
     nn.Linear(POOLED_HEAD_WIDTH, widths[2], bias=False),
 )
```

Use default non-inplace SiLU. Do not unroll the head, add a gain, normalize the
activation, alter the branch scale, or introduce a learned activation
parameter. The parameter count remains exactly `1,003,482`; matrix MACs and
all trainable state remain unchanged. Only pointwise activation arithmetic and
its backward differ.

Construction must remain exactly accepted. The pooled head is still registered
after accepted model/classifier initialization inside the restoring CPU-only
RNG fork, using `torch.random.default_generator.manual_seed(36036)`. Both
bias-free weights receive the same direct `init.kaiming_normal_` calls in the
same order. `nn.SiLU()` has no parameters, buffers, or RNG use, so all named
state bytes and post-construction CPU/CUDA RNG states must equal accepted.

Keep the accepted ReLU-calibrated Kaiming samples rather than selecting a SiLU
gain or rescaling `W1`/`W2`. This preserves initialization as required and
isolates the activation, but it means startup branch amplitude can decrease.
That amplitude change is part of the preregistered treatment and must be
measured descriptively, never normalized away.

## Mechanism and Counter-Hypothesis

Because `z` follows a ReLU, its components are nonnegative, but each row of the
bias-free `W1` has signed weights. Thus `a_j < 0` can encode that a weighted
combination of pooled channels opposes hidden feature `j`. Accepted ReLU maps
that region to zero and supplies no `W1` data gradient through that unit.
SiLU maps moderate negative values to negative hidden responses and provides a
nonzero derivative, allowing `W2` and `W1` to use graded inhibitory evidence.
In a width-64 bottleneck this might represent opposite channel contrasts more
economically than learning paired ReLU units.

The strong counter-hypothesis is representational sufficiency. `W2` already
has signed columns, so positive ReLU features can make either positive or
negative residual corrections; the direct `z` path also preserves all pooled
features. Negative `W1 z` may simply mean absence, for which zero is the right
code. SiLU additionally shrinks positive activations, changes the effective
residual-branch norm under an initialization calibrated for ReLU, and can
couple gradients across units that the accepted hard gate usefully sparsified.
The accepted head's `94.48%`/`0.2456` result and four subsequent head-adjacent
misses make preserving its exact geometry a serious prior.

Local evidence is limited to the following:

- EXP036: the exact ReLU pooled MLP scored `94.48%` with loss `0.2456` at
  `130.304` passes. This supports cheap post-pooling nonlinear capacity, but
  supplies no comparison between activation functions.
- EXP041: changing only training supervision around the same head retained
  `128.538` passes but scored `94.26%`; preserve sole refined-path CE.
- EXP042: exact-neutral content pooling retained `127.933` passes but scored
  `93.80%`; preserve uniform GAP and do not reinterpret SiLU as a pooling test.
- EXP044: a zero-open dispersion input retained `128.712` passes but scored
  `93.95%`; preserve the accepted 128-dimensional mean input and add no hidden
  features or parameters.

## Semantic Preflight

Use an ignored evaluator-free harness with an independent
`git show a7c42dc:train.py` oracle. Block evaluator invocation and CIFAR-10 test
construction before importing either module. Emit all measurements before
assertions. Require:

1. The production diff changes only the accepted pooled-head `nn.ReLU()` to
   default `nn.SiLU()`. Constants, imports, forward structure, data,
   augmentation, LR, loss, optimizer, seed, timing budget, evaluation cadence,
   and summary code remain accepted.
2. Accepted and candidate model construction from cloned seed-42 CPU/CUDA
   states yields identical named parameter/buffer keys, shapes, dtypes, and
   bytes; identical post-construction CPU/CUDA RNG; exactly `1,003,482`
   parameters; and no new state. Independently reconstruct the seed-36036
   matrices and require byte equality.
3. The module topology is exactly
   `Linear(128,64,bias=False) -> SiLU(inplace=False) -> Linear(64,128,bias=False)`.
   Reject GELU, functional aliases hidden elsewhere, inplace operation, bias,
   gain, dropout, normalization, learned slope, or any scale/init change.
4. On fixed FP64 CPU and FP32 CPU/CUDA fixtures, capture production `z`, `a`,
   hidden output, refined vector, and logits. Match candidate values against
   the independent `a * sigmoid(a)` formula and match the candidate-minus-
   accepted logit delta against `0.1 * C W2 [SiLU(a)-ReLU(a)]` within declared
   numerical tolerances. Require a finite, nonzero function change on a fixture
   containing both signs.
5. On an explicit scalar grid including negative, zero, and positive values,
   verify negative SiLU outputs, zero at zero, the independent analytic
   derivative `sigmoid(a) + a*sigmoid(a)*(1-sigmoid(a))`, and finite forward and
   backward values. This proves the intended primitive without claiming a
   beneficial magnitude.
6. For fixed early-mixup and hard-label training fixtures, independently
   replay loss, gradients, and fresh/preseeded Nesterov updates. Require finite
   nonzero backbone, classifier, `W1`, and `W2` gradients/updates, accepted
   parameter-group order/options, no activation state, and no RNG consumption.
   Candidate gradients are expected to differ from accepted at step one; do
   not impose common-gradient identity.
7. Reconfirm accepted batch-shared alpha-0.2 mixup, 65% mixup/RandAugment
   controls, clean tail, LR curve/floor, coupled `5e-4` matrix decay, worker
   semantics, once-per-epoch evaluation, and frozen evaluator contract.

Report but never gate, tune, or select a variant from: negative/near-zero hidden
preactivation fractions; ReLU versus SiLU hidden RMS and nonzero fraction;
negative versus positive SiLU contribution RMS; residual/direct norm ratio;
candidate/accepted logit RMS delta and cosine; loss delta; grouped gradient
norm ratios and cosines. These quantify the exact startup/function change and
whether the hypothesized negative region is populated. They cannot justify a
gain, rescale, new initialization, or switch to GELU. A semantic failure closes
the implementation unless it is an independently demonstrated harness defect.

## Throughput and Exposure Gate

The candidate adds no tensors, matrix MACs, or spatial work. It replaces a
ReLU over only `B x 64` pooled hidden values with sigmoid/multiply arithmetic;
the spatial backbone still dominates and the accepted head was only about
`1.4%` of forward time. Therefore a material throughput loss is unlikely, but
SiLU backward is more expensive than a ReLU mask and eager kernel behavior on
the H20 must be measured rather than inferred.

On one idle H20, compare accepted and candidate complete production-equivalent
steps in early-mixup and hard-label regimes. Include pinned H2D, LR writes,
zeroing, accepted mixing when active, full forward/loss/finite guard/backward,
coupled Nesterov update, and synchronization. Use at least 20 disposable
warmups and two counterbalanced `A/C/C/A` cycles, producing four retained
windows of at least 50 steps per arm and regime from restored deterministic
fixtures. Print every synchronized window before any assertion.

Using four-window medians, compute

```text
retention =
  (0.65 / candidate_mix_ms + 0.35 / candidate_hard_ms) /
  (0.65 / accepted_mix_ms  + 0.35 / accepted_hard_ms)

projected_passes = 130.304 * retention
```

Require every population CV `<=5%`, candidate peak allocation `<2,048 MiB`,
`retention >= 127/130.304 = 0.9746439096`, and projected passes `>=127.0`.
A stable timing miss ends the proposal before scoring. Do not rerun timing,
compile/fuse only the candidate, change inplace mode, rescale the branch,
substitute GELU, or relax the floor as a rescue.

## Sole Scored Run and Decision Contract

After both gates pass, reconfirm baseline `94.48%` at `a7c42dc`, one idle H20,
local CIFAR-10, frozen `prepare.py`, exact source scope, and no stale `run.log`.
Run exactly one fixed-seed score through the required frozen evaluator:

```bash
timeout 600s uv run train.py > run.log 2>&1
```

Require exit zero, one finite summary, `300.0-300.1` counted seconds, wall time
under 600 seconds, exactly `1,003,482` parameters, correct ordered augmentation
transitions, unique every-fifth plus final evaluations, and no runtime or
integrity fault. Compute realized exposure as `num_steps * 256 / 50000`.

Success requires both `best_test_acc >=94.58%` and realized exposure
`>=127.0` passes. Final test accuracy and loss are descriptive only and cannot
rescue or veto the preregistered result. A valid score below 127 passes remains
the sole score and may not be rerun, but it is not a successful protected-
exposure result. Never inspect intermediate test accuracy for control flow,
choose between activations from test behavior, or launch a second valid score.

## No-Rescue Closure

- A valid `>=127`-pass score below `94.58%` closes this exact SiLU replacement
  and the immediate smooth/signed activation neighborhood. Do not try GELU,
  approximate GELU, Swish-beta, Mish, ELU, leaky ReLU, another activation,
  width, bias, head scale, gain, normalization, initialization, seed, decay/LR
  exception, temporal cutoff, or rerun.
- A `>=94.58%` score below 127 passes is not success and does not authorize a
  speed rescue, compiler change, alternate activation, or repeat. It records a
  lower-exposure accuracy observation only.
- A normal-exposure success supports only the complete fixed SiLU treatment.
  It does not prove negative evidence was causal, that smoothness helped, or
  that SiLU dominates GELU generally, and it does not authorize a sweep.
- A stable timing failure closes systems viability of the exact treatment
  without an accuracy claim. An invalid scored run permits repair only of an
  independently demonstrated infrastructure or verifier defect while keeping
  the production treatment byte-for-byte unchanged.

## Falsifiable Hypothesis

If moderate negative preactivations in the accepted 64-unit pooled bottleneck
contain useful class-boundary evidence that hard ReLU suppresses, then replacing
only that ReLU with fixed default SiLU, while preserving every parameter,
initialization, scale, classifier, and spatial operation, will retain at least
127 passes and raise fixed-seed `best_test_acc` from `94.48%` to at least
`94.58%`. The honest prior is low because the treatment is adjacent head tuning,
also attenuates positive activations, and the accepted direct path plus signed
`W2` may already express the needed corrections.

## Local Sources

- `01-definition.md`, `02-system-understanding.md`,
  `03-experiment-learnings.md`, and `04-results.tsv`.
- `experiments/036/02-plan.md`, `03-execute.md`, and `04-analysis.md`: accepted
  pooled ReLU residual MLP, initialization, timing, and score.
- `experiments/041/04-analysis.md`: failed direct-path auxiliary CE.
- `experiments/042/04-analysis.md`: failed exact-neutral content-attention
  pooling.
- `experiments/044/03-execute.md` and `04-analysis.md`: failed spatial-
  dispersion input and current protected-exposure decision contract.
