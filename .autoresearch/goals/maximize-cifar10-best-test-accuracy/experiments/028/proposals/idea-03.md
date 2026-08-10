# Proposal: Fixed 0.01-Slope LeakyReLU Throughout ResNet-20

## Intervention and hypothesis

Replace all 19 accepted postactivation `ReLU` operations with parameter-free
`LeakyReLU(x, negative_slope=0.01)`. Preserve the width-2 ResNet-20 graph,
convolution/BN/shortcut ordering, parameter count, global-average classifier,
batch 128, FP32/default-TF32 execution, SGD momentum and decay, elapsed-time LR
schedule, N1/M7 plus probability-0.5 alpha-1 CutMix through 80%, hard weak tail,
seed, workers, timer, and evaluator.

The candidate changes one representation primitive. Negative BN responses and
negative residual sums retain 1% of their value and gradient rather than becoming
exact zeros. This may reduce inactive features and preserve weak signed evidence
through the short strong phase without adding parameters or a learned branch.
The counter-risk is equally direct: accepted ReLU sparsity may itself regularize,
and repeatedly applying slope 0.01 to a negative identity-path value still
attenuates it sharply. The final pooled feature also becomes signed, changing the
classifier geometry rather than merely easing optimization.

**Falsifiable hypothesis:** matched-gain fixed-slope LeakyReLU retains at least
99% of accepted exposure, preserves a healthy strong-phase checkpoint, and raises
`best_test_acc` from 94.15% to at least 94.25%. Point prediction is 94.25-94.35%,
with switch accuracy at least 89.0%. A valid lower score rejects this full-network
slope-0.01 point; the slope must not be tuned after seeing the result.

## Evidence and local fit

He et al.'s rectifier analysis gives the forward/backward variance rule for a
negative slope and reports that PReLU can improve image-classification fitting at
negligible compute. This proposal uses the fixed small-slope special case, so it
does not inherit evidence for learned slopes or ImageNet-scale models. The slope
0.01 is deliberately conservative: it opens the negative half-space without
turning the network into a nearly linear residual model.

Locally, width and conservative CutMix improved the frontier, while increased
decay, stronger CutMix, zero-gamma, full preactivation, and early removal of
strong regularization either suppressed strong fit or failed the 94.25% gate.
LeakyReLU keeps the accepted augmentation and active residual branches intact.
Unlike preactivation, it does not move BN or delete post-add nonlinearities; unlike
zero-gamma/ECA, it has no recruitment parameter that can saturate. Nevertheless,
it touches every block from step one, so exact-corpus optimization safety is
mandatory.

## Exact implementation and initialization gain

Add one constant:

```python
NEGATIVE_SLOPE = 0.01
```

Replace the three source-level `F.relu(...)` sites—the stem, block `conv1-BN`,
and block post-add return—with
`F.leaky_relu(..., negative_slope=NEGATIVE_SLOPE)`. This yields exactly 19 dynamic
activations. Keep `inplace=False`; add no `nn.PReLU`, activation module, parameter,
buffer, conditional phase behavior, or per-stage slope.

Match initialization to the declared nonlinearity:

```python
init.kaiming_normal_(
    m.weight,
    a=NEGATIVE_SLOPE,
    mode="fan_in",
    nonlinearity="leaky_relu",
)
```

for the same Conv/Linear tensors currently initialized by the shared helper. The
gain becomes `sqrt(2 / (1 + 0.01**2)) = 1.414142857`, only 0.005% below ReLU's
`sqrt(2)`. With the same seed, tensor shapes, and construction order, the same
standard-normal draws must be consumed; candidate initial weights should equal
control weights times `1/sqrt(1.0001)` within FP32 rounding. This tiny matched-gain
change is part of the predeclared activation package, not a second tuned lever.
BN affine/running initialization remains unchanged, and parameter count remains
exactly 1,073,962.

Do not keep ReLU gain while claiming variance matching; conversely, do not retune
LR, decay, BN epsilon/momentum, residual scale, classifier, or slope to compensate
for the candidate trajectory.

## Structural and immutable-corpus safety gates

Before timing, require:

1. Static checks prove three LeakyReLU call sites, 19 dynamic invocations, fixed
   slope 0.01, no remaining model `F.relu`, nine residual blocks, 19 convolutions,
   19 BNs, two Option-A shortcuts, unchanged parameter/state keys and ordering,
   and exactly 1,073,962 trainable parameters.
2. From independently reconstructed seed-42 models, verify the expected matched-
   gain ratio for every Conv/Linear tensor and identical BN state, optimizer group
   membership, post-construction RNG state, data policy, and schedule. No new RNG
   draw is permitted.
3. On synthetic negative, zero, positive, extreme, and random inputs, compare the
   activation and autograd to the direct piecewise formula. Require `y=x` and
   derivative 1 for positive values, `y=0.01*x` and derivative 0.01 for negative
   values, finite outputs/gradients, and declared PyTorch derivative behavior at
   zero. Record negative fractions and RMS before/after each of the 19 sites.
4. Materialize one immutable **200-batch production corpus** before either arm:
   80 strong-hard, 80 strong alpha-1 CutMix, and 40 weak-hard batches in a fixed
   interleaving. Persist tensor values, targets, kind, and SHA-256. Independently
   restore accepted/candidate model and SGD state; both arms must consume the
   byte-identical corpus. Seed-only forkserver replay is forbidden by EXP-019/021.
5. On reset hard and soft one-step comparisons, require finite logits, loss,
   gradients, parameters, momentum, and BN buffers; no candidate-only >95% class
   concentration; candidate/control loss and logit-RMS ratios `<=1.25`; global
   gradient and update-norm ratios in `[0.67,1.50]`; and no trainable tensor update
   ratio above 2.0 when the control norm is nontrivial.
6. Continue aligned training for all 200 corpus steps. Require no nonfinite state,
   no candidate-only concentration event, positive finite BN variances, candidate
   terminal loss EMA `<=1.25x` control, and global parameter-displacement ratio
   `<=1.5`. Record per-site negative fraction, negative/positive RMS, gradient RMS,
   prediction histograms, and loss trajectory; serialize failures before asserting.

The candidate is not expected to match control outputs—the signed negative path
is its mechanism. The bounds detect scale or optimization collapse without
mistaking ordinary mechanistic divergence for an implementation error. Any failed
gate retires this exact slope/init package; do not rescue it with slope 0.001/0.1,
ReLU in selected stages, learned PReLU, or unchanged gain.

## Paired H20 timing and exposure gates

On exactly one idle 97,871-MiB H20, run five alternating fresh-process
control/candidate pairs using the same persisted hard/CutMix/weak tensors. After
100 warmups per arm, time at least 1,000 synchronized complete steps including
transfer, forward, loss, backward, SGD, and synchronization. Measure the three
target/view paths separately and combine means 40/40/20. Record forward and
backward CUDA-event components because LeakyReLU backward may use a different
kernel even though convolution remains dominant.

Proceed only if:

- weighted candidate/control mean step time `<=1.01`, every pair `<=1.02`, both
  trial-mean CVs `<=2%`, and candidate p95 `<=1.03x` control mean;
- projected exposure
  `floor(26,898 * control_mean / candidate_mean) >=26,630` updates (99% retention),
  with the accepted 80/20 time split;
- peak allocation `<650 MiB` and no more than 32 MiB above paired control;
- all losses/gradients remain finite and a conservative total-runtime projection,
  including the unchanged evaluation schedule and loader switch, is below 540s.

LeakyReLU has no throughput mechanism, so even a stable >1% slowdown invalidates
the premise: a representation gamble with materially fewer updates would be
confounded by the fixed-time budget. No compiler, fusion, channels-last, autocast,
or in-place rewrite may be used as a timing fallback.

## Production verification and interpretation

If and only if every gate passes, run the exact seed-42 candidate once via
`uv run train.py > run.log 2>&1` on the sole idle H20. Require exit zero, finite
standard summary fields, 300.0 counted seconds, total below 600s, at least 26,500
actual updates, exactly 1,073,962 parameters, one 80% switch with eight workers
stopped, approximately 50% strong CutMix, hard weak-tail targets, and no duplicate
evaluation epoch.

Record switch accuracy against 89.73%, first weak accuracy against 93.16%, best
and final accuracy, final NLL against 0.1934, steps, epochs, evaluation count,
runtime, VRAM, and phase-wise activation negative fractions. Accept only if
`best_test_acc >=94.25%` with every integrity condition satisfied.

- Switch below 87.08% indicates that signed leakage harmed the short strong phase.
- Healthy switch but worse first-weak accuracy implicates signed-feature/BN
  resettling at the augmentation transition.
- Better fit but worse NLL/top-1 indicates that removing ReLU sparsity weakened
  generalization rather than fixing inactive features.
- A safety/timing failure is invalid; a complete lower-accuracy run is
  no-improvement. Neither may trigger a slope or initialization rescue.

## Risks

- BN makes permanently dead channels less obvious than in unnormalized networks,
  so “dying ReLU” may not be a real limiter here.
- Negative post-add values are attenuated again by later activations; the candidate
  preserves signed evidence only weakly and changes the identity-path mapping.
- The final feature map is no longer nonnegative. GAP can cancel positive and
  negative evidence, which may conflict with localized CutMix features.
- Matched Kaiming gain is theoretically correct but differs negligibly at slope
  0.01; any outcome is overwhelmingly the activation effect, not evidence that the
  0.005% scale adjustment matters.
- Fixed LeakyReLU has weaker direct empirical support than learned PReLU, and the
  cited ImageNet result does not establish a CIFAR-10 ResNet-20 gain.
- A 0.10-point pass is ten test examples at one fixed seed; it is protocol-valid
  but should not be treated as a precise effect estimate.

## Sources

- He et al., *Delving Deep into Rectifiers: Surpassing Human-Level Performance on
  ImageNet Classification*, ICCV 2015:
  https://openaccess.thecvf.com/content_iccv_2015/html/He_Delving_Deep_into_ICCV_2015_paper.html.
- `experiments/010/04-analysis.md` — accepted 94.15% recipe, switch/weak-tail
  trajectory, exposure, NLL, and integrity anchors.
- `experiments/012/04-analysis.md` and `experiments/015/04-analysis.md` — local
  preactivation and zero-gamma underfit evidence.
- `experiments/027/04-analysis.md` — preserving the complete strong phase remains
  necessary despite full exposure.
- `02-system-understanding.md`, `03-experiment-learnings.md`, and `04-results.tsv`
  — bottleneck, protocol lessons, recurring failures, and current frontier.
