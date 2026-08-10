# Proposal: Ghost BatchNorm with 64-Example Virtual Groups

## Intervention and hypothesis

Keep the accepted EXP-010 optimizer batch at 128, but make every training-mode
BatchNorm layer normalize the first and second contiguous 64-example halves
independently. The two halves share the same affine `gamma` and `beta`; gradients
from both accumulate into the same parameters and SGD still performs exactly one
update after all 128 examples. In evaluation mode, use the ordinary single set of
running statistics and affine parameters, with no virtual grouping or evaluation
batch-size restriction.

Preserve the complete accepted width-2 postactivation ResNet-20 recipe: 32/64/128
channels, nine residual blocks, Option-A shortcuts, FP32/default-TF32 execution,
batch 128, SGD momentum 0.9, all-parameter decay `1e-4`, elapsed-time LR schedule,
N1/M7 plus probability-0.5 alpha-1 CutMix through 80%, hard crop/flip weak tail,
seed 42, worker lifecycle, timer, and fixed evaluator.

**Hypothesis:** statistics from 64 examples introduce modest layerwise
normalization noise while retaining the gradient diversity of a 128-example SGD
update. This should regularize the already-capable width-2 model without the
strong-view suppression produced by higher decay, zero-gamma, wider CutMix, or
residual-branch gates, and raise `best_test_acc` from 94.15% to at least 94.25%.
Point prediction: at least 96.7% of accepted updates, switch accuracy at or above
88.5%, and `best_test_acc` around 94.25-94.35%.

This is a normalization-noise experiment, not a large-optimizer-batch rescue:
batch 128 is only moderately large, and early CIFAR feature maps already give BN
many spatial samples. The expected effect may therefore be below the ten-example
acceptance margin.

## Why it is plausible here

Hoffer, Hubara, and Soudry introduced Ghost Batch Normalization to decouple the
batch used for SGD from the smaller groups used to estimate normalization
statistics, reporting a reduced generalization gap without increasing the number
of optimizer updates. Later GhostNorm analysis attributes part of the effect to
regularization specific to independent normalization groups. Those results make
the direction credible, but neither establishes a gain for a short, single-GPU,
batch-128 CIFAR-10 run already using CutMix and RandAugment.

The local motivation is narrower. EXP-007 showed that width-2 capacity was worth
its fixed-time cost, and EXP-010 then gained 0.60 points from regional mixed-target
regularization while retaining 99.10% exposure. Since then, extra coupled decay,
75% CutMix, zero-gamma branches, depth-width exchange, asymmetric widening, and
learned ECA recruitment have failed or been vetoed. Ghost BN changes no
convolution, shortcut, target, optimizer equation, or phase duration. It injects
noise through the 19 existing normalization points and is disabled automatically
for evaluation, making it a more isolated generalization lever than another
architecture branch.

The mechanism is still risky. EXP-010's 89.73% switch fit leaves limited room for
another regularizer, and all 19 BN layers change from the first update. The system
is 75.46% backward-bound, so two normalization calls per layer plus explicit
running-stat reduction can also reduce fixed-time exposure even though optimizer
batch size is unchanged. Safety and H20 timing are therefore launch gates, not
post-hoc diagnostics.

## Exact implementation in `train.py`

Add `GHOST_BATCH_SIZE = 64` and a `GhostBatchNorm2d` subclass of
`nn.BatchNorm2d`. Replace the stem and block `BatchNorm2d` constructors with this
class; do not alter module construction order. The subclass creates no trainable
state beyond standard BN, so parameter count remains exactly **1,073,962**, state
dict key names/shapes remain compatible with the accepted model, and convolution
and classifier initialization consume the same RNG stream.

Training `forward` must:

1. Assert FP32 input, `N == BATCH_SIZE == 128`, and divisibility by
   `GHOST_BATCH_SIZE`; split only the sample dimension into contiguous `[0:64]`
   and `[64:128]` views. Do not reshuffle groups or draw RNG.
2. Normalize each view with
   `F.batch_norm(view, None, None, self.weight, self.bias, True, 0.0, self.eps)`
   and concatenate in original order. Passing no running buffers prevents either
   ghost from mutating evaluation state; the shared affine tensors ensure their
   gradients sum across both groups.
3. Under `torch.no_grad()`, increment `num_batches_tracked` exactly once and update
   the one `running_mean`/`running_var` pair exactly once from all 128 logical-batch
   activations. Use `torch.var_mean(x, dim=(0,2,3), correction=0)`, convert the
   biased variance to the unbiased estimator with
   `m/(m-1)`, where `m=N*H*W`, and apply the ordinary BN exponential factor
   (`momentum=0.1`, or cumulative averaging if momentum is `None`). Detach the
   statistic tensors before the in-place buffer update.

Evaluation `forward` must delegate to ordinary BatchNorm evaluation semantics
(`super().forward(x)` while `self.training` is false). Evaluation can therefore
use any batch size chosen by `Eval`; it neither splits examples nor mutates
buffers. This buffer policy deliberately differs from calling the same BN module
twice with running state: sequential calls would update the buffer twice,
introduce ghost-order dependence, and turn momentum 0.1 into an effective 0.19
per optimizer step. Full-128 updates isolate training-normalization noise while
retaining one population-stat observation per logical batch. Candidate buffers
will still differ from control because upstream activations differ.

Do not implement the ghosts by reshaping them into channels, which requires
full-activation permutations/copies at every BN, and do not create two BN modules,
which would duplicate affine parameters and evaluation buffers. Do not change BN
epsilon/momentum, batch size, LR, decay, CutMix, precision, memory format, or
evaluation frequency as a fallback.

## CutMix and phase interaction

CutMix runs in the worker collator before the model sees a batch. When its batch
gate fires, all 128 targets are probability vectors and every image is already a
regional mixture; donor pairing may cross the eventual 64-example boundary.
Ghost BN uses only activations, not targets, so cross-group donors require no label
rewrite. Both ghosts in a given optimizer step remain either on the accepted
CutMix path or the accepted hard-target path. Contiguous grouping is preferred to
an extra permutation because DataLoader shuffle already randomizes samples and an
additional draw would change the registered RNG trajectory.

At the 80% loader switch, the same module continues to use two groups of 64 on
weak hard-label batches. Its full-128 running buffers can resettle on the weak
distribution exactly once per step, as accepted BN does. Dense tail evaluation
uses those single buffers, never the current validation batch's statistics. Thus
the candidate tests training-statistic noise across both phases without changing
the accepted CutMix probability, target geometry, or evaluation definition.

## Structural, semantic, and exact-corpus safety gates

Before any timing or accuracy run, require all of the following:

1. Assert 19 `GhostBatchNorm2d` modules, nine residual blocks, 19 convolutions,
   1,073,962 parameters, unchanged parameter names/order/shapes, two Option-A
   shortcuts, and FP32 parameters/buffers. Load one accepted state dict into both
   arms and require bitwise-identical parameters and initial buffers.
2. For synthetic constant, random, and high-dynamic-range inputs at N=128,
   compare every candidate layer against a direct two-half reference. Require
   finite forward/backward values, matching outputs and input/affine gradients
   within FP32 tolerance, one counter increment, and running mean/unbiased
   variance equal to the declared full-128 formula. In eval mode require the
   candidate to match ordinary BN from the same buffers, accept non-64-divisible
   batch sizes, leave state immutable, and emit finite `[N,10]` model logits.
3. Materialize **200 exact post-transform batches** before either arm: 80 strong
   hard, 80 strong alpha-1 CutMix, and 40 weak hard batches, interleaved in a fixed
   order that represents the 80/20 phase mix. Persist tensors, targets, target
   kind, and one SHA-256 manifest. Independently restore accepted weights,
   buffers, SGD state, and RNG state; both arms must consume byte-identical
   tensors and targets in the same order. Fresh forkserver seed replay is
   forbidden by EXP-019/021.
4. Before updating and after every aligned update, record loss, logit RMS,
   prediction histogram, gradient norm, update norm, every BN buffer's finite
   status/range, and all BN counters. Require finite loss/logits/gradients,
   parameters, momentum, and buffers; positive running variances; every counter
   equal to the logical step count; no candidate-only prediction concentration
   above 95%; candidate/control RMS logit-displacement and global update-norm
   ratios no greater than 2.0; and no per-step candidate loss above 2.0x control.
5. Through all 200 aligned steps require no candidate-only concentration event and
   candidate terminal loss EMA no more than 1.25x control. Then evaluate both
   models on a fixed non-test diagnostic tensor in eval mode; require finite
   logits, no state mutation, and no candidate-only greater-than-95%
   concentration. Record per-layer train-normalized activation RMS and
   candidate/control running-stat displacement as diagnostics, not improvement
   evidence.

Any semantic, state-update, finiteness, concentration, displacement, or trajectory
failure retires this exact group-64 policy. Repair implementation defects only;
do not rescue it by choosing group 32/96, applying ghosts to fewer layers, changing
BN momentum, or weakening a gate.

## H20 timing and resource gates

On exactly one otherwise-idle 97,871-MiB H20, use the same persisted corpus for
five alternating fresh-process control/candidate pairs after conditioning. Each
arm gets 100 warmup and at least 1,000 synchronized complete training steps,
including transfer, forward, hard/probability-target loss, backward, SGD, buffer
updates, and synchronization. Measure strong-hard, strong-CutMix, and weak-hard
paths separately and combine means with 40/40/20 weights.

Proceed to production only if all hold:

- weighted candidate/control mean step time `<=1.03`, every pair `<=1.05`, and
  per-arm trial-mean CV `<=2%`;
- candidate p95 step time `<=1.08x` its paired control mean;
- projected exposure
  `floor(26,898 * control_mean / candidate_mean) >=26,000` updates (96.66%
  retention), with at least 20,800 projected strong and 5,200 weak-tail updates;
- candidate peak allocation `<700 MiB` and no more than 64 MiB above control;
- finite values and correct once-per-step BN counters in every timing trial;
- a conservative total-runtime projection below 540 seconds after charging the
  accepted evaluation schedule and loader switch.

The strict 3% mean gate reflects the local evidence that exposure loss has often
manifested as strong underfit and that Ghost BN adds no image-throughput
mechanism. Count both extra normalization launches and the full-batch
`var_mean`/buffer-update kernels; no channels-last, compiler, autocast, fewer
evaluations, or fused custom extension may be introduced as a timing rescue.

## Production verification and falsification

If and only if all safety and timing gates pass, run the exact candidate once at
seed 42 with `uv run train.py > run.log 2>&1` on the sole idle H20. Do not retry or
tune group size after observing the trajectory. Require:

- exit zero, all finite standard summary fields, 300.0 counted training seconds,
  and total runtime below 600 seconds;
- exactly 1,073,962 parameters, at least **26,000 optimizer steps**, one 80%
  augmentation switch, eight workers stopped, approximately 50% CutMix during
  the strong phase, hard weak-tail targets, and no duplicate evaluation epoch;
- all 19 BN counters agree with optimizer-step count at the final summary and all
  running variances remain finite and positive;
- `best_test_acc >=94.25%` for improvement.

Record switch accuracy against EXP-010's 89.73%, first weak accuracy against
93.16%, best/final accuracy, final NLL against 0.1934, steps, epochs, evaluation
count, runtime, memory, and strong/weak CutMix counts. A switch below 87.08% is the
registered strong-underfit diagnosis but cannot stop, tune, or rerun the candidate.

A healthy switch followed by worse first-weak accuracy implicates train/eval
normalization mismatch or slower buffer resettling. Healthy conversion followed
by a lower peak rejects group-64 normalization noise as useful regularization at
this operating point. A lower switch with lower loss suggests excessive
normalization noise rather than insufficient capacity. A valid result below
94.25% is no-improvement; a safety/timing/protocol failure is invalid. Because
the result is one fixed seed and 0.10 point is ten test examples, a bare pass is
protocol-valid but should not be overstated as a precise effect estimate.

## Risks and interpretation limits

- Group 64 halves only the sample contribution to statistics; spatial averaging
  remains large, especially at 32x32 and 16x16, so the regularization effect may
  be too weak to clear 0.10 point.
- CutMix makes images non-i.i.d. through donor pairing, including across ghost
  boundaries. Independent statistics may amplify mixed-region artifacts instead
  of encouraging useful invariance.
- Training uses group-local statistics while evaluation uses full-population
  running buffers. This is the intended Ghost-BN discrepancy, but it can hurt the
  abrupt strong-to-weak conversion despite the explicit full-128 buffer policy.
- Two fused BN calls and one full-batch reduction per layer may exceed the 3%
  exposure budget on small CIFAR feature maps; unchanged optimizer batch does not
  guarantee unchanged throughput.
- Replacing all 19 layers gives the cleanest canonical test but cannot identify
  whether early or late normalization supplied benefit or harm.
- The literature is strongest for genuinely large optimizer batches and longer
  horizons. This experiment's batch 128, width-2 postactivation graph, short
  fixed-time budget, and composite augmentation are a substantial transfer.

## Decision rule

Accept only if every integrity/resource condition passes and
`best_test_acc >=94.25%`. A valid lower result falsifies full-network group-64
Ghost BatchNorm under the accepted recipe. Do not rescue it inside EXP-026 by
changing group size, limiting ghosts to selected stages, altering running-stat
momentum, recalibrating BN after training, or retuning LR/decay. Any of those is a
new mechanism requiring a new reviewed experiment.

## Sources

- `experiments/010/04-analysis.md` — accepted 94.15% CutMix recipe, exposure,
  switch, first-weak, NLL, and integrity anchors.
- `goals/maximize-cifar10-best-test-accuracy/02-system-understanding.md` —
  backward-cost bottleneck, memory headroom, and exposure/generalization limits.
- `goals/maximize-cifar10-best-test-accuracy/03-experiment-learnings.md` and
  `04-results.tsv` — recurring underfit, exact-corpus protocol, and failed
  optimizer/architecture/regularization routes.
- Hoffer, Hubara, and Soudry, *Train longer, generalize better: closing the
  generalization gap in large batch training of neural networks*, NeurIPS 2017 /
  arXiv 1705.08741: https://arxiv.org/abs/1705.08741.
- Dimitriou and Arandjelovic, *A New Look at Ghost Normalization*, arXiv
  2007.08554: https://arxiv.org/abs/2007.08554.
- Ioffe and Szegedy, *Batch Normalization: Accelerating Deep Network Training by
  Reducing Internal Covariate Shift*, ICML 2015:
  https://proceedings.mlr.press/v37/ioffe15.html.
