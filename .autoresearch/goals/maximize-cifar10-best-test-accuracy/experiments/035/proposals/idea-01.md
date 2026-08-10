# Proposal: Fixed SiLU/Swish Throughout the Accepted ResNet-20

## Decision and falsifiable hypothesis

Replace all 19 dynamic `ReLU` operations in the accepted width-2 postactivation
ResNet-20 with parameter-free SiLU (Swish with fixed beta 1). Preserve the exact
accepted initialization, residual/shortcut ordering, tensor shapes, parameter and
buffer sets, FP32/default-TF32 execution, batch 128, SGD momentum and all-parameter
`1e-4` decay, elapsed-time LR schedule, N1/M7 plus probability-0.5 alpha-1 CutMix
through 80%, hard weak tail, seed, workers, timer, and evaluator. Do not add a
learned activation coefficient or a separately tuned gain.

SiLU maps `x` to `x * sigmoid(x)`. Unlike ReLU, it carries bounded negative
responses, supplies a nonzero local derivative around zero, and changes the hard
positive gate into a smooth one. The representation hypothesis is that this
continuous signed path will preserve weak evidence through the nine residual
blocks and make feature/gradient propagation less brittle under the short strong
phase. The opposing hypothesis is strong: BatchNorm already continually
recenters preactivations, accepted ReLU sparsity may be useful regularization, and
the last post-add activation feeds global average pooling without another BN, so
signed SiLU features can cancel class evidence.

**Falsifiable hypothesis:** fixed beta-1 SiLU will pass exact-corpus numerical and
trajectory gates, retain at least 98% of accepted fixed-time optimizer exposure,
enter the weak tail with test accuracy at least 89.0%, and raise seed-42
`best_test_acc` from the current 94.15% moving baseline to at least **94.25%**.
The formal success test is the moving-baseline-plus-0.10 rule queried immediately
before production; 94.25% is the presently registered threshold, not permission
to hardcode a stale baseline. A valid run below the threshold rejects this exact
all-site SiLU point. A preflight veto makes it invalid. Neither result authorizes
an in-experiment site subset, beta change, initialization change, or LR rescue.

## Evidence, transfer limits, and expected benefit

Hayou, Doucet, and Rousseau's ICLR 2019 analysis identifies Swish-like smooth
activations as capable of propagating information more deeply than ReLU-like
functions when paired with a compatible initialization near the edge of chaos.
That gives a real mechanism—information and gradient propagation—not merely a
generic “modern activation” claim. It does not establish an accuracy gain here:
the accepted network is only 20 layers deep, is BatchNorm-normalized, has
residual additions, is trained for 300 counted seconds rather than to long-horizon
convergence, and uses strong CutMix/RandAugment views. The local paper note also
states that the full text was unavailable during retrieval, so the proposal must
not infer a CIFAR-10 effect size from the abstract-level distillation.

The expected representation benefit is nevertheless more substantive than the
fixed slope-0.01 LeakyReLU proposal considered but not executed in EXP028. SiLU
does not merely leak one percent of a negative response: its negative branch and
derivative are appreciable near zero, its positive branch turns on smoothly, and
its non-monotonic region can suppress weak negative noise without clipping every
negative value to zero. Those properties could help the model retain localized,
mixed-label evidence during the accepted CutMix plateau and then refine it in the
hard weak tail.

The mechanism is also less targeted than the metric requires. No accepted-run
diagnostic shows dead channels or discontinuous gating is the current limiter,
and BN weakens the classic dying-ReLU argument. The last activation changes the
classifier input from nonnegative pooled evidence to signed pooled evidence, so
the effect is a network-wide representation change rather than a pure
conditioning improvement. The plausible result band is deliberately broad
(approximately 93.8-94.4%); a clean sub-threshold result is more likely than a
large gain.

## Relationship to prior experiments

This is an unexecuted activation-family test, not a rerun of a recorded failed
point. EXP028 proposed fixed slope-0.01 LeakyReLU but rejected it during review;
there is no production or safety result for that proposal. SiLU is materially
distinct because it changes both halves of the activation around zero, has a
smooth bounded derivative profile, and follows the Swish signal-propagation
mechanism in the cited paper rather than the much weaker “one-percent leakage”
mechanism. It still inherits EXP028's main objections: BN makes dead features an
unproven limiter, all 19 sites change at once, and the final pooled representation
becomes signed.

The intervention preserves more of the accepted system than several failed
representation experiments:

- Unlike EXP012 preactivation, it does not move BN/activation order, remove
  post-add nonlinearities, or alter shortcut propagation.
- Unlike EXP015 zero-gamma and EXP025 identity-start gates, it does not suppress
  or gradually recruit residual branches; all accepted branches remain active
  from the first forward pass.
- Unlike EXP014/031 pooling branches, it adds no classifier branch and no
  uncontrolled max feature.
- Unlike EXP024, it changes no width, transition ratio, tensor shape, or RNG draw.

That distinction does not erase the local warning. EXP012 and EXP015 show that
compute-neutral block-wide changes can suppress the protected strong-phase fit by
2.85-3.25 points. EXP034 shows that an almost unchanged initial BN-normalized
function can still develop six late early-phase one-class transients. EXP020,
EXP022, EXP024, EXP028, EXP031, and EXP033 establish that lower loss or apparently
small changes cannot override class/output/update trajectory vetoes. EXP029
shows that an eleven-line “cheap” operation can cost 1.97% of fixed-time
exposure. SiLU must therefore pass both full-step timing and immutable-corpus
trajectory gates before it earns the single scored run.

## Exact implementation scope

Change exactly the three source-level functional activation calls:

```python
# BasicBlock first conv-BN activation
out = F.silu(self.bn1(self.conv1(x)))

# BasicBlock post-add activation
return F.silu(out)

# stem conv-BN activation
out = F.silu(self.bn1(self.conv1(x)))
```

Because the first two calls execute once in each of nine blocks and the stem call
executes once, this produces exactly 19 dynamic SiLU sites. Use `F.silu` with its
fixed beta-1 semantics and default `inplace=False`. Add no `nn.SiLU` modules,
`nn.PReLU`, constants that imply a tunable beta, custom autograd, approximations,
phase conditionals, activation mixtures, or per-stage exceptions. No model ReLU
may remain. Evaluation must use the same SiLU graph as training.

The tracked production diff must contain only those three substitutions. In
particular, leave `_weights_init` byte-for-byte accepted:

```python
if isinstance(m, (nn.Conv2d, nn.Linear)):
    init.kaiming_normal_(m.weight)
```

Preserve exactly 19 Conv2d modules, 19 BatchNorm2d modules, one Linear, nine
blocks, two Option-A stride-two shortcuts, 1,073,962 trainable parameters, and
the accepted state-dict keys/order. Do not use in-place activation, compiler,
channels-last, autocast, fusion, a custom Triton kernel, or an evaluator/timer
change to recover throughput; each would add a second mechanism.

## Initialization choice and its risk

The cited signal-propagation result explicitly couples activation and
initialization. PyTorch's Kaiming gain interface has no declared SiLU mode, and a
single analytically “correct” gain would not obviously apply to this BN residual
graph or to both pre-add and post-add distributions. Changing the initializer
would therefore create a second global reparameterization with no unique local
choice. More importantly, EXP034's Conv-only fan-out experiment kept initial
relative logit L2 below 0.044% yet amplified relative stem updates to 13.99%,
whole-model updates to 1.95x control, and entered repeated one-class states.

This proposal consequently keeps every initial parameter and buffer bitwise
identical to control and interprets the accepted ReLU-oriented fan-in Kaiming
weights as a deliberate isolation choice, not as theoretically optimal for SiLU.
BN before the stem and first activation in each block should moderate input-scale
mismatch. It does not protect the post-add sites, especially the final post-add
site before GAP, and it cannot guarantee comparable gradients or running stats.
The safety screen must measure those effects rather than adding gain, residual
scale, LR, clipping, or BN-epsilon compensation.

## Static, semantic, and identical-initial-state gates

Before any timed work, require all of the following and serialize evidence before
asserting:

1. Syntax, Ruff, formatting, and tracked-scope checks pass; only `train.py`
   differs from the integration baseline and its diff is the three substitutions
   above.
2. Static and hook-based checks prove three source call sites and 19 dynamic SiLU
   invocations, no model ReLU invocation, and unchanged module topology, state
   keys/order, parameter count, optimizer membership, data rules, schedule,
   evaluator, timer, and summary.
3. Independently reconstruct control and candidate from the same registered CPU
   and CUDA seed-42 states. Require every initial parameter and buffer to be
   bitwise identical and post-construction CPU/CUDA RNG states to match exactly.
   SiLU must introduce no construction-time or forward-time RNG draw.
4. On CPU and CUDA float32 test vectors containing large negatives, the SiLU
   minimum neighborhood, values around zero, moderate positives, and random
   samples, compare candidate output with `x * sigmoid(x)` and autograd with
   `sigmoid(x) + x*sigmoid(x)*(1-sigmoid(x))` at tight dtype-appropriate
   tolerances. Require finite outputs/gradients and record, rather than invent,
   PyTorch's exact zero derivative behavior (expected 0.5).
5. Hook all 19 sites on immutable production hard and soft batches before the
   first update. Record preactivation/output mean, RMS, negative/zero fraction,
   per-example RMS, local input-gradient RMS, pooled-feature norm/sign balance,
   logits, loss, class histogram, and all BN buffers. Do not demand output parity,
   because activation divergence is the mechanism; require finite values, no
   silent site, no candidate-only greater-than-95% predicted-class share, and
   candidate/control logit, pooled-feature, loss, and global-gradient norm ratios
   each within `[0.25, 4.0]`.

The wide initial ratios are catastrophic bounds, not claims that a fourfold
change is desirable. Exact weight/RNG identity proves implementation isolation;
the activation statistics reveal whether the unretuned initialization starts in
a plausible regime.

## Immutable-corpus trajectory gate

Reuse rather than regenerate or filter the registered exact corpora:

- EXP022's 200-batch accepted strong post-policy corpus,
  `experiments/022/preflight-corpus.pt`, registered file SHA-256
  `e04dc2fe9d3994cef8bf192401bc36c63f306946fd3b9a2339b9f64040318946`.
- EXP028's 64-batch accepted weak hard-label corpus,
  `experiments/028/weak-corpus.pt`, registered file SHA-256
  `ffefe980241d9719c8d7f2b44fe81c1b3f94e35003b0a645d3fea5999a745032`.

Recompute file and tensor digests; verify batch counts, shapes, target ranks,
hard/CutMix coverage, and finite tensors. Run independent control and candidate
models from their identical registered initial states over byte-identical copies
of all 200 strong batches at LR 0.1 and then all 64 weak batches using the
accepted cosine-tail LR values sampled over progress 0.8 through 1.0. Use the
production-default CUDA backend settings, ordinary momentum SGD, coupled decay,
and no evaluator. Because production-default CUDA execution is not bitwise
reproducible, first run a predeclared control/control calibration under the same
ordering; ordinary shared early concentration or numerical spread in that
calibration is context, never a threshold tuned after seeing candidate results.

At each step record loss and debiased EMA, prediction histogram and maximum class
share, logit mean/RMS/per-example norm, global gradient/update/parameter/momentum
norms, per-layer relative updates, BN means/variances/counters, and all 19 site's
input/output/gradient RMS and sign fractions. Record final pooled-feature norms
and classwise margins separately because GAP cancellation is a candidate-specific
risk.

Production is authorized only if:

- all parameters, buffers, logits, losses, gradients, updates, momentum, and
  diagnostics remain finite, BN variances stay positive, and counters/state
  membership are complete;
- there are zero candidate-only steps with maximum predicted-class share above
  95% while the matched control is at or below 95%; shared control behavior must
  be interpreted against the predeclared control/control calibration, not erased;
- candidate/control logit RMS, whole-gradient norm, and whole-update norm stay
  below 5x at every step; candidate whole update stays below 25% of its pre-update
  parameter norm and below 5x its preceding 16-step median;
- no trainable tensor receives an update above 50% of its pre-update norm, no
  activation site becomes nonfinite or effectively silent, and candidate/control
  site-output, site-gradient, and pooled-feature RMS ratios remain within
  `[0.20, 5.0]`;
- terminal debiased candidate/control loss-EMA ratio is at most 1.5 in both strong
  and weak phases, with no persistent one-class or zero-margin terminal state;
- corpus/source/RNG declarations and every immutable input hash remain unchanged.

These checks veto catastrophic geometry; they do not select SiLU for having a
lower 264-step loss. EXP015 establishes that short-fit gains can invert over the
full strong phase, so no loss advantage may be used as an accuracy claim. Any
failed gate retires this exact all-site SiLU candidate before timing or production.
Do not rerun the corpus, substitute deterministic backends, relax a threshold,
or rescue only selected sites after observing a veto.

## Fixed-300-second throughput cost and paired timing gate

ReLU is essentially a thresholding pointwise operator. SiLU evaluates a
sigmoid-like nonlinear function and multiplication in forward and has a more
expensive derivative; its backward also needs activation/input information that a
ReLU mask can represent cheaply. PyTorch may fuse each SiLU expression into one
operator, but the 19 pointwise forward/backward sites still add arithmetic and
saved-tensor traffic. The measured accepted step is dominated by model backward
(75.46%) and forward (22.11%), so loader or optimizer headroom cannot hide a GPU
activation slowdown. At 26,898 accepted updates, a 1%, 2%, or 5% step slowdown
would cost approximately 266, 527, or 1,281 updates in the fixed 300-second
budget. SiLU has no exposure-improving mechanism to offset that cost.

On exactly one idle 97,871-MiB H20, run seven predeclared alternating
fresh-process control/candidate pairs under production-default backend flags.
Use the same persisted representative strong-hard, strong-CutMix, and weak-hard
tensors, include transfer, forward, cross-entropy, backward, ordinary SGD, and
synchronization, warm each arm for at least 100 complete steps, and measure at
least 1,000 complete synchronized steps per arm. Combine path means with the
production time fractions: 40% strong hard, 40% strong CutMix, and 20% weak hard.
Record wall step, CUDA-event forward/backward/update components, p50/p95, trial
order, clocks/utilization, memory allocation, losses, and numerical finiteness.

Proceed only if the weighted candidate/control mean step-time ratio is at most
`1.02`, every pair is at most `1.04`, both trial-mean CVs are at most 2%, and the
candidate p95 is at most 1.05x the control mean. Require projected exposure
`floor(26,898 * control_mean / candidate_mean) >= 26,360` updates (98% retention),
peak allocation below 700 MiB and no more than 64 MiB above paired control, and a
conservative total-runtime projection below 540 seconds. These are load-bearing
gates: a stable larger slowdown invalidates the fixed-time premise even if the
short trajectory is safe. Do not recover timing with in-place SiLU, approximate
Swish, compiler, layout, precision, batch-size, or data-loader changes.

## Single production run and verdict

Only after every structural, initial-state, trajectory, and timing gate passes,
query the moving baseline, verify exactly one idle H20, remove any stale completed
log, and run the exact candidate once:

```bash
timeout --kill-after=5s 595s uv run train.py > run.log 2>&1
```

Require exit zero, one finite ten-field summary, 300.0-301.0 counted training
seconds, total runtime below 600 seconds, exactly 1,073,962 parameters, at least
26,360 actual optimizer steps absent documented system-wide contention, one
augmentation switch near 80% with all eight workers stopped, hard weak targets,
first weak LR near 0.01, 45-55% CutMix among strong batches, no duplicate
evaluation epoch, at most one evaluation per epoch, and no more than the accepted
19 total evaluation looks. Production code may contain no diagnostic hooks.

Record accuracy/NLL at the registered early checkpoints, the switch against
89.73%, the first weak checkpoint against 93.16%, best/final accuracy, final NLL
against 0.1934, steps, epochs, runtime, VRAM, look count, source/log hashes, and
the exact preflight/timing artifacts. The switch and NLL comparisons diagnose the
mechanism but cannot override the primary metric.

- **Improvement:** all integrity conditions pass and `best_test_acc` is at least
  the queried moving baseline plus 0.10 points (currently at least 94.25%).
- **No improvement:** the single production run is valid but misses that gate,
  regardless of lower train loss, lower NLL, or favorable intermediate accuracy.
- **Invalid/crash:** any preflight or timing veto, tracked-scope/hardware/data/
  evaluator/timer/look-count violation, malformed/nonfinite summary, nonzero
  exit, or total runtime at least 600 seconds.

Expected failure signatures are interpretable. A switch below 89.0% would align
with EXP012/015's block-wide strong-underfit pattern. A healthy switch followed by
poor weak recovery would implicate signed-feature or BN-statistics resettling at
the policy transition. Better train fit with worse NLL/top-1 would suggest that
ReLU sparsity was useful regularization. Safe trajectory plus timing failure
would establish that general SiLU kernels cost too much exposure in this tiny
FP32 graph without saying anything about accuracy.

No follow-up within EXP035 may alter beta, use Mish/GELU, restore ReLU only at the
stem or final block, change initialization/gain, scale residuals, warm up LR, clip
gradients, change BN epsilon/momentum, approximate/fuse SiLU, or reroll seed or
corpus. Each is a new experiment requiring an independently registered
hypothesis.

## Risks

- **Mechanism risk — high:** no local evidence establishes dead or discontinuous
  ReLU gating as the frontier limiter, and BN reduces the classic motivation.
- **Initialization risk — medium-high:** accepted Kaiming weights isolate the
  activation but are not proven edge-of-chaos-compatible with SiLU in this BN
  residual graph; changing them would repeat a failed reparameterization family.
- **Representation risk — high:** all 19 nonlinearities change, and signed final
  features can cancel under GAP or interact poorly with localized CutMix targets.
- **Strong-fit risk — high:** prior global residual/activation-order interventions
  suppressed the protected 89.73% switch fit even at similar exposure.
- **Throughput risk — medium-high:** sigmoid-backed forward/backward can exceed
  the 2% timing allowance and erase hundreds of fixed-budget updates.
- **Evidence-transfer risk — high:** edge-of-chaos theory for deep random networks
  does not directly predict a gain for shallow BN ResNet-20 under this curriculum.
- **Metric-noise risk — medium:** the required 0.10-point gain is ten test
  examples at one fixed seed; it is protocol-valid but not a precise effect size.

## Sources

- Hayou, Doucet, and Rousseau, *On the Selection of Initialization and Activation
  Function for Deep Neural Networks*, ICLR 2019:
  `experiments/035/papers/activation-initialization-edge-of-chaos.md` and
  <https://openreview.net/forum?id=H1lJws05K7>.
- `experiments/010/04-analysis.md` — accepted 94.15% recipe, 89.73% switch,
  93.16% first weak checkpoint, 0.1934 final NLL, and 26,898-step anchors.
- `experiments/028/proposals/idea-03.md` and `experiments/028/01-idea-review.md` —
  prior unexecuted fixed LeakyReLU proposal and its local objections.
- `experiments/012/04-analysis.md`, `experiments/015/04-analysis.md`, and
  `experiments/034/04-analysis.md` — block-wide underfit and initialization-path
  safety warnings.
- `experiments/022/preflight-corpus.pt` and
  `experiments/028/weak-corpus.pt` — registered immutable production corpora.
- `02-system-understanding.md`, `03-experiment-learnings.md`, `04-results.tsv`,
  and `.autoresearch/project-notes/project-insights.md` — current bottleneck,
  protocol constraints, failure mechanisms, and moving frontier.
