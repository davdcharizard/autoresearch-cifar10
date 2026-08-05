# Proposal: Late Whole-State EMA

## Recommendation

Keep the accepted 691,674-parameter WRN-16-2, batch-256 FP32 SGD/Nesterov
training, alpha-0.2 mixup through 65% counted time, 35% hard-label tail,
time-based LR schedule, augmentation, seed, loader, and evaluation cadence
unchanged. Add one detached FP32 exponential moving average (EMA) of the complete
inference state beginning exactly when mixup ends. Evaluate that EMA state at
each already-scheduled evaluation after initialization, restoring the live
training state immediately afterward.

Fix `EMA_START_FRACTION = 0.65` and `EMA_DECAY = 0.999`; update once after every
eligible optimizer step. Do not tune the start, decay, cadence, or state policy
after observing the scored run, and do not combine EMA with an architecture,
target, optimizer, or schedule change.

## Why This Exact Treatment

The accepted model reaches 94.07% after 141.9 data passes with near-zero late
training loss, while more exposure, stronger regularization, altered late
weight decay, and two raw-capacity treatments failed the 94.17% gate. EXP-011's
extra block gained 0.08 points but worsened test loss to 0.2782, reinforcing
that late specialization/generalization is the opportunity rather than missing
training throughput. The saved weight-averaging literature says a carefully
selected late window can mildly improve generalization at low cost.

Starting at 65% is not a free search over windows: it uses the accepted recipe's
existing semantic boundary. It excludes the high/medium-LR mixed-target phase
and averages only the hard-label refinement trajectory. At accepted throughput,
the remaining 105 seconds contain about 9,700 updates. A per-step decay of
`0.999` has an e-folding horizon of about 1,000 updates (roughly 11 counted
seconds) and an effective variance-reduction sample size near 2,000 updates.
The initial copied state therefore contributes less than 0.01% by the end, but
the average remains short enough to track a still-improving cosine trajectory.
This deliberately reduces the lag risk of the older 0.995-every-10-steps idea,
whose approximately 2,000-step e-folding horizon was criticized as too slow.

## Exact Update Rule

Use the same pre-step `progress = total_training_time / TIME_BUDGET_S` already
used for mixup. The first eligible step is therefore the first step for which
`progress >= 0.65`, exactly the first hard-label step.

1. Run the unchanged forward, loss, backward, and `optimizer.step()` on the live
   model.
2. After that first eligible `optimizer.step()`, initialize the shadow by
   cloning the current model inference state. Initialization is a direct copy,
   so no bias correction is needed and no EMA interpolation occurs on that
   step.
3. After every subsequent eligible `optimizer.step()`, and before the existing
   `torch.cuda.synchronize()`, update each floating shadow tensor `s` from its
   corresponding live tensor `x` as
   `s <- 0.999 * s + 0.001 * x`, under `torch.no_grad()`.
4. Keep the update inside the existing timed training interval so all EMA work
   is charged to the 300-second budget. A dtype/device-compatible
   `torch._foreach_lerp_(shadow, live, 0.001)` is appropriate; a grouped
   per-tensor `lerp_` implementation must be mathematically identical.

The live model remains FP32 and is the only model differentiated or optimized.
EMA tensors are detached FP32 CUDA tensors. The optimizer's momentum buffers,
parameter groups, LR, gradients, and live parameter objects are never averaged
or replaced.

## Parameters And BatchNorm State

The shadow covers every named parameter, including convolution and classifier
weights and all BatchNorm affine scales/biases. It also covers every BatchNorm
buffer:

- `running_mean` and `running_var` are floating tensors and receive the same
  `0.999 / 0.001` EMA update as parameters. They are sampled after the current
  training forward has updated them, so each shadow snapshot corresponds to the
  same live training state as its parameters.
- `num_batches_tracked` is integral and cannot be interpolated. Copy its latest
  live value exactly at initialization and after every eligible step. It does
  not affect evaluation output, but tracking it makes the swapped state complete
  and auditable.

Do not use live BN buffers with EMA parameters and do not perform a BN
recalibration pass. The whole-state policy avoids the known mismatch of
averaged weights with instantaneous live running statistics, while avoiding an
extra uncounted traversal of training data. Averaging running variance is still
an approximation because BN statistics are nonlinear in the weights; that is a
pre-registered risk of this exact candidate.

The shadow contains about 693,498 floating values: 691,674 parameters plus
1,824 BN running-statistic values, or about 2.65 MiB in FP32. Thirteen integral
BN counters add only 104 bytes. A same-sized temporary live-state backup during
evaluation remains negligible relative to the accepted model's roughly 1.1 GiB
peak allocation and the H20 capacity.

## Evaluator Swap And Restore Semantics

Before EMA initialization, retain the accepted live-model evaluations. After
initialization, use only the EMA view at every existing `epoch % 5 == 0` or
budget-exhausted evaluation. Never evaluate live and EMA states in the same
epoch.

For an EMA evaluation, under `torch.no_grad()`:

1. Clone all live parameters and buffers into a temporary backup without
   replacing any `Parameter` or buffer object.
2. Copy the shadow values into those same live objects. Copy floating shadows
   and the latest integral `num_batches_tracked` values.
3. Call `evaluator.evaluate(model, device)` exactly once. The frozen evaluator
   switches the module to eval mode, so no BN state is mutated.
4. In an unconditional `finally` block, copy every backed-up live value back
   into its original object, even if evaluation raises.

Assert bitwise equality of every restored live parameter and buffer against the
backup in a preflight semantic test. Also assert that parameter object IDs and
the optimizer's parameter references are unchanged, live optimizer state is
unchanged by a successful swap, and an injected evaluator exception still
restores all state. The existing next-epoch `model.train()` call restores mode;
the swap helper must not alter evaluation cadence or update `best_acc` itself.

## Cost And Throughput Preflight

Before the single scored run, run one fully local, evaluator-free H20 preflight.
Replace `prepare.Eval` with a fail-closed dummy before importing `train.py`, so
neither real evaluation data nor test accuracy can be touched. The preflight
must pass all state/swap invariants above and compare accepted versus candidate
production steps from pinned host inputs through the final CUDA synchronize.

Use exact cloned WRN-16-2 models and optimizer states, fixed inputs/targets, and
independent but initially identical CPU/CUDA RNG streams. Time mixup and
hard-label regimes separately in a balanced accepted/candidate order after at
least 25 complete warmup steps per path, using three 50-step windows per
path/regime. The candidate mixup regime must exercise the `EMA is None` branch;
the candidate hard-label regime must begin with an initialized shadow and
include a whole-state EMA update every timed step. Include nonblocking host
copies, LR writes, mixup sampling/permutation when applicable, zeroing,
forward/loss/finite guard/backward, optimizer step, EMA branch/update, and the
final synchronization exactly as production does.

For each path/regime report all window means and require population CV/mean no
greater than 5%. Define each path's production estimate as
`0.65 * mixup_median_ms + 0.35 * hard_median_ms`, throughput retention as
`accepted_aggregate / candidate_aggregate`, and projected passes as
`141.9 * retention`. Launch scoring only if retention is at least 95%, projected
passes are at least 134.8, logits are finite `[256, 10]`, loss and all live/EMA
states remain finite FP32, the candidate live weights stay exactly aligned with
its matched accepted path, and there is no OOM. These are operational gates
only and may not inspect evaluator output.

The 95% gate is intentionally stricter than prior capacity gates because this
candidate claims negligible overhead. Passing it ensures any accuracy result
tests averaging rather than materially reduced exposure. EMA swap copies occur
only during the already-excluded sparse evaluation time, but total wall time
must still remain below 600 seconds.

## Invariants

- Modify only `train.py`; add no dependency, checkpoint, data download, remote
  call, or evaluator modification.
- Preserve the accepted architecture, 691,674 parameter count, FP32 training,
  optimizer groups, LR/warmup/floor, weight decay, batch size, augmentation,
  alpha-0.2 mixup cutoff, seed 42, maximum steps, and loader settings.
- Initialize EMA exactly once on the first completed hard-label update; log its
  step and counted time and the final number of EMA updates.
- Update the shadow after SGD and before synchronize; EMA never feeds training
  forward/backward or changes optimizer state.
- Evaluate at most once per epoch, using live state before initialization and
  whole EMA state afterward. Always restore live state before training resumes.
- Keep the scored command fixed at
  `timeout 600s uv run train.py > run.log 2>&1`; run it once on one H20 with the
  fixed seed, and remove `run.log` after analysis.

## Expected Mechanism

The hard-label tail continues useful low-LR refinement, but each reported live
checkpoint is one stochastic SGD/BN state and can specialize to recent batches.
Jointly averaging weights, BN affine parameters, and their contemporaneous
running statistics should suppress short-horizon iterate and calibration noise
without weakening early mixup or changing the learned representation. Unlike
additional dropout, CutMix, or stronger mixup, it does not add training-time
target or feature corruption. Unlike more depth/width, it does not ask the
fixed budget to optimize extra capacity.

## Risks And Interpretation

- **Small available headroom:** accepted best and final accuracy are already
  equal, so residual iterate variance may be too small for a 0.10-point gain.
- **Trajectory lag:** even the shorter `0.999` EMA can trail improving live
  weights. Evaluating EMA only after 65% can therefore hide a superior live
  late checkpoint; this is required to preserve one evaluation per epoch.
- **BN nonlinearity:** averaged running statistics may not exactly calibrate the
  averaged network. No live-buffer fallback or BN recalibration is allowed in
  this experiment.
- **Hidden launch cost:** per-step state interpolation may cost more than its
  2.65 MiB data volume suggests. The matched 95% gate must reject it before
  scoring if so.
- **Metric selection:** sparse evaluations still expose multiple EMA iterates,
  exactly as accepted training exposes multiple live iterates. Cadence and seed
  stay fixed; there is no decay/window search or rerun.

## Falsifiable Hypothesis

One fixed-seed H20 run of the accepted WRN with whole-state EMA initialized at
65% counted time and updated every subsequent step with decay 0.999 will retain
at least 134.8 projected passes in preflight, complete 300 counted seconds and
under 600 total seconds, and raise `best_test_acc` from 94.07% to at least
94.17% without more than one evaluation per epoch.

A valid scored result below 94.17% is a no-improvement and rejects this exact
65%-start, 0.999-decay, whole-state policy as a sufficient standalone treatment;
do not rerun or rescue it with a different decay, live BN buffers, recalibration,
or a second simultaneous change. If the scored run unexpectedly realizes fewer
than 134.8 passes despite passing preflight, the verdict remains formal
no-improvement if accuracy misses, but the mechanism is operationally
confounded rather than a stable negative.

## Evidence

- `knowledge/papers/weight-averaging.md`: late-window averaging can mildly
  improve generalization with little memory or implementation cost, while the
  averaging window must exclude under-trained early iterates.
- `experiments/002/proposals/idea-01.md` and `experiments/002/01-idea-review.md`:
  the earlier parameter-only EMA proposal exposed trajectory-lag and live-BN
  mismatch risks; the shorter horizon and whole-state policy directly fix those
  design ambiguities without combining treatments.
- `experiments/002/04-analysis.md`: accepted mixup creates a natural 65% phase
  boundary and leaves roughly 9,700 hard-label updates for late averaging.
- `experiments/011/04-analysis.md`: more raw low-resolution depth gave a small
  accuracy gain but worse test loss, motivating generalization control rather
  than another adjacent capacity increase.
