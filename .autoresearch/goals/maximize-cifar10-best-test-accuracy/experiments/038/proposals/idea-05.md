# Proposal: Very-Mild Strong-Phase Stochastic Depth

## Decision and hypothesis

Apply batchwise stochastic depth only to the six same-width residual blocks of
the accepted width-2 postactivation ResNet-20, and only while the accepted
N1/M7 plus CutMix strong phase is active. The stem, all three stage-entry
blocks, both shape-changing Option-A transitions, the weak tail, and every
evaluation remain full-depth. Use depth-increasing drop probabilities

```text
layer1[1], layer1[2], layer2[1], layer2[2], layer3[1], layer3[2]
       0.01,       0.02,       0.03,       0.04,       0.05,       0.06
```

with no rate ramp and no post-result adjustment. For block residual branch
`F_i(x)`, survival `q_i = 1 - p_i`, and one scalar batchwise draw
`z_i ~ Bernoulli(q_i)`, the strong-training pre-ReLU sum is

```text
x + z_i * F_i(x) / q_i
```

and the block returns `ReLU` of that sum. Thus the residual sum, though not the
nonlinear block output, is unbiased relative to the full block. A dropped
branch is not evaluated at all, which is essential: per-example masks that
still execute both convolutions would regularize but would not attack the
measured 75.46% backward bottleneck. Weak-tail training and `model.eval()` use
`x + F_i(x)` exactly, with no scaling.

The hypothesis is that an expected 0.21 omitted branches per strong step gives
a small ensemble-of-depths regularizer while saving enough convolution/BN
forward and backward work to process about 1-2% more optimizer steps in the
fixed budget, without compounding the accepted recipe's strong-phase underfit.
The accuracy prediction is deliberately weak: the point could clear 94.25% if
the mild regularization and extra strong-view exposure complement CutMix, but a
neutral or negative result is at least as plausible.

## Exact implementation and RNG semantics

Add stochastic-depth metadata only to the six blocks listed above. Immediately
after the existing `t0`, use a dedicated CPU `torch.Generator` seeded exactly
once with `42038` to draw a length-six FP32 vector in the fixed order above on
every strong optimizer step. Convert it to six booleans and install the mask on
the blocks before forward. The draw and Python dispatch stay inside counted
time. This generator is independent of the global CPU/CUDA RNG and loader
workers, so model initialization, shuffle, transforms, and CutMix retain their
accepted streams. Evaluations consume no drop draws. On the first weak batch,
disable dropping and clear all masks; no weak or evaluation call may advance
the dedicated generator.

When a branch is dropped, return the same-width shortcut through the existing
post-add ReLU without calling either Conv or BN. Since the input is the output
of a preceding postactivation block, this is an exact identity in normal
finite execution. The skipped parameters retain `grad is None`; ordinary
PyTorch SGD therefore performs no momentum, coupled-decay, or parameter update
for those tensors on that step. This per-branch sparse update is part of the
registered method, not an implementation accident. On survival, divide the
BN2 branch output by `q_i` immediately before addition. Do not scale the
shortcut, alter BN momentum, compensate LR/decay, drop stage entries, use
per-sample masks, or combine another regularizer.

The six probabilities sum to 0.21. Approximately 80.68% of strong batches use
all nine blocks, 17.71% drop exactly one same-width branch, and 1.61% drop two
or more. Each same-width residual branch contributes about 11.7% of this
network's convolution MACs, so the arithmetic upper estimate is a 2.46%
reduction in full-model convolution work during the strong phase. Because only
80% of counted time uses dropping and control flow costs something, the likely
whole-run exposure gain is around 1-2%, not a headline stochastic-depth gain.

## Evidence and central risk

The local stochastic-depth note distills Huang et al. (ECCV 2016): random
residual bypass can shorten training graphs and regularize an ensemble of
depths, but its strongest evidence is on networks vastly deeper than this one.
The systems profile makes it unusually relevant here because forward plus
backward account for 97.57% of GPU-stage time. It is one of few regularizers
that can remove rather than add counted kernels.

Transfer to this recipe is nevertheless poor. The accepted model has only nine
residual blocks, so one sampled omission removes one ninth of its transformation
depth, whereas dropping a block in a 100- or 1,200-layer network is granular.
EXP012's full preactivation already reduced switch accuracy by 2.85 points, and
EXP015's six initially suppressed same-width branches reduced it by 3.25 points
despite favorable 64-step loss and full later recruitment. Those experiments
show that this short, heavily augmented strong phase does not have spare branch
activity. The proposed rates are therefore intentionally tiny, transitions
remain live, the inverted scale is at most `1/0.94 = 1.06383`, and all blocks
are restored for the hard weak tail. None of that eliminates the risk that
even occasional whole-batch omission worsens representation learning or that
sparse per-parameter momentum histories destabilize geometry.

## Construction and exact-corpus gates

Before timing, an experiment-local controller must establish all of the
following without production hooks:

1. With dropping disabled, accepted and candidate construction has identical
   named state, parameter count (1,073,962), parameter order, global CPU/CUDA
   RNG hashes, logits, gradients, and ordinary SGD configuration. All stage
   entries and Option-A transitions must be ineligible for dropping.
2. Forced-survive and forced-drop FP64 oracles must match the declared formula
   and identity path for every eligible block. The maximum branch multiplier
   is exactly `1.063829787...`; eval and weak mode must be bitwise equal to the
   accepted full network from equal state.
3. Two generators seeded `42038` must produce identical mask bytes for at
   least 10,000 steps; global RNG states must remain unchanged. A separately
   coded oracle must match every production decision. The 200-step registered
   prefix must report exact per-block survival counts, joint-drop histogram,
   and mask SHA, and contain at least one actual drop. Eval and weak calls
   inserted between strong calls must not alter the subsequent mask sequence.
4. Reuse the registered EXP022 200-batch strong corpus (94 hard/106 CutMix,
   SHA-256 `e04dc2fe9d3994cef8bf192401bc36c63f306946fd3b9a2339b9f64040318946`)
   followed by the EXP028 64-batch weak corpus (SHA-256
   `ffefe980241d9719c8d7f2b44fe81c1b3f94e35003b0a645d3fea5999a745032`).
   Run two accepted/accepted calibrations before the accepted/candidate replay
   and qualify every ratio against the control envelope using denominator-safe
   absolute-plus-relative rules.
5. Require finite loss/logits/parameters/gradients/momentum/BN state, no
   candidate-only persistent >95% one-class predictions, terminal strong and
   weak loss EMA no more than 1.5x control, logit RMS and whole update/gradient
   norms no more than 5x qualified controls, and maximum parameter update below
   25% of its parameter norm. Record blockwise update and momentum norms so a
   skipped branch cannot create a delayed candidate-only spike.
6. Require every transition BN and every BN during the 64 weak steps to advance
   exactly as the full model does. For each eligible block, its two BN counters
   must equal the oracle's strong survival count plus 64, proving that dropped
   branches truly executed no Conv/BN path and that the deterministic tail
   restored every block. Strong branch executions and `grad is None` events
   must exactly match the mask oracle.

These are safety and mechanism gates, not short-horizon accuracy proxies;
EXP015 proved that favorable 64-step loss cannot establish phase-scale fit.

## Paired timing and production decision

Use one unscored conditioning process and seven counterbalanced fresh-process
control/candidate pairs on the idle H20. Each arm starts from identical model
and optimizer state and uses byte-identical hard/CutMix/weak tensors. Warm 100
steps, then measure at least 1,000 synchronized complete steps. Candidate
strong timing must execute the seeded production mask generator inside `t0`;
report its exact survival histogram. Time weak full-depth steps separately.
Record mean/median/p95, CV, H2D/forward/loss/backward/update CUDA stages, images
per second, and peak allocation.

Advance only if the weighted strong-step candidate/control mean ratio is
`<=0.985`, every pair is `<1.0`, ratio CV is below 3%, forward-plus-backward
accounts for at least 80% of the saving, weak full-depth ratio is `<=1.01`,
peak allocation is below 650 MiB, and a conservative total-wall projection is
below 540 seconds. Project the accepted 26,898-step exposure using separately
measured strong and weak ratios and require at least 27,200 total steps. A
candidate that merely regularizes but does not measurably shorten the strong
graph fails this joint mechanism and must not receive the sole production run.

If all gates pass, run seed 42 exactly once for 300 counted seconds with no more
than 19 unique evaluations and the accepted fixed evaluator. Require the exact
predeclared mask stream, 45-55% CutMix, a single 80% strong-to-weak transition,
all six blocks full-depth thereafter, clean worker shutdown, finite summary,
and total wall below 600 seconds. `best_test_acc >=94.25%` is improvement; a
complete lower score is no-improvement with no rate rescue or rerun. Report the
80% switch accuracy against EXP010's 89.73%, first-weak recovery, final NLL,
strong/weak step counts, realized branch survivals, and actual timing gain.

## Finalist recommendation

**Do not promote this to the top three unless the other proposals are weaker.**
It is technically coherent and uniquely couples a backward-cost attack to
regularization, so it is a useful reserve candidate. But the expected compute
gain is only about 2.5% in the strong phase, the original evidence concerns far
deeper networks, and EXP012/015 give direct local evidence that suppressing
residual computation worsens strong-phase fit in this nine-block model. The
very mild rates reduce that danger at the same time that they reduce the likely
accuracy effect below the 0.10-point acceptance margin.
