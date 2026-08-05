# Idea 02: EMA Over the Hard-Label Tail

## Proposal

Keep the accepted WRN-16-2, alpha-0.2 mixup through 65% of counted time, and
all optimizer and schedule settings unchanged. At the existing mixup-to-hard-
label transition, initialize a shadow model from the live model and update an
exponential moving average after every subsequent optimizer step with decay
**0.999**. Evaluate the shadow model, rather than the live model, at each
already-scheduled evaluation after EMA initialization. The final reported model
is therefore the EMA model; no additional evaluation is introduced.

This is one isolated intervention: late iterate averaging. It does not change
the live optimization trajectory except for the small, counted runtime cost of
updating the shadow state.

## Diagnosis and Rationale

The accepted EXP-002 run reaches 94.07%, with final accuracy equal to best
accuracy after a 35% clean-label, low-learning-rate tail. EXP-001 likewise
finished only 0.04 points below its best. Three subsequent changes to the input
regularization policy all regressed despite normal exposure. The next useful
question is therefore not another augmentation variant, but whether the late
SGD iterates occupy a sufficiently good basin that smoothing their residual
parameter noise improves the test solution.

The local ICML 2025 note on *When, Where and Why to Average Weights?* reports
that carefully windowed checkpoint averaging can mildly improve generalization
at low implementation and memory cost. Its important qualification is to avoid
early, under-trained iterates. Starting EMA exactly when mixup ends gives a
semantically clean window: every averaged update comes from the validated
hard-label refinement regime, while the representation-building portion of
training remains unchanged.

Decay 0.999 has an effective e-folding horizon of roughly 1,000 optimizer
steps. At EXP-002 throughput (about 195 steps per epoch), that is 5.1 epochs;
about 95% of final EMA mass lies in the last 3,000 steps, or 15.4 epochs. The
35% tail contains roughly 9,900 steps, so initialization bias is negligible
(`0.999^9900` is about `5e-5`) while the average remains local to the final
basin. A slower decay would retain materially earlier, higher-LR weights;
a faster decay would be too close to the final live checkpoint to test the
averaging mechanism meaningfully.

## BatchNorm State Policy

The EMA must include **all floating-point model state**, not just trainable
parameters. In particular, apply the same 0.999 update to BatchNorm affine
parameters, `running_mean`, and `running_var`; copy non-floating buffers such as
`num_batches_tracked` from the live model. This prevents the clearly incorrect
combination of averaged convolution weights with stale buffers from the model
state at EMA initialization.

This can be implemented with `torch.optim.swa_utils.AveragedModel` using
`use_buffers=True` and a small multi-tensor averaging function: use
`torch._foreach_lerp_` for floating/complex dtype groups and direct copies for
integer groups. PyTorch's built-in buffer-averaging path is preferable here to
`update_bn`: a post-training BatchNorm recalibration would require an additional
full pass over training data, consume wall-clock training work outside the
accepted path, and introduce a second augmentation sample. EMA of buffers is
not mathematically identical to exact population statistics under the final
averaged weights, but it is a coherent time average of the BatchNorm state
paired with the same checkpoint trajectory and introduces no uncounted pass.
This approximation is the main technical risk of the proposal and must not be
described as exact recalibration.

## Exact Integration

Only `train.py` changes:

1. Add `EMA_DECAY = 0.999` and use the existing `MIXUP_END_FRACTION` as the EMA
   start point; do not introduce a separately tunable window.
2. Construct an `AveragedModel` shadow on the same GPU before training. Model
   construction/deep copy consumes no random samples and adds only one copy of
   the roughly 692k-parameter network.
3. On the first optimizer step for which `progress >= 0.65`, call
   `update_parameters(model)` once to copy the current post-step model and
   buffers exactly. On every later hard-label step, update with decay 0.999.
4. Place the EMA update after `optimizer.step()` and before the existing CUDA
   synchronization and elapsed-time measurement. Its compute cost is therefore
   charged to `total_training_time` and can reduce optimizer-step exposure; it
   is not hidden in excluded evaluation time.
5. At an evaluation, use the live model before EMA initialization and the EMA
   shadow afterward. Keep the existing every-fifth-epoch plus budget-exhausted
   condition, so there is still at most one evaluator call per epoch. Update
   `best_acc` from whichever model was preregistered for that phase; do not
   evaluate both and select post hoc.
6. Log the EMA initialization once and report the number of EMA updates in the
   final summary for attribution. Do not alter the evaluator or `prepare.py`.

The shadow adds approximately 2.7 MiB of parameter storage plus small BatchNorm
buffers, against a measured peak near 1.1 GiB on a 97.9 GiB H20. It requires no
extra activations or forward pass. The raw arithmetic is about 700k lerps per
hard-label step; grouped foreach updates should make memory traffic cheap, but
Python/grouping overhead remains an empirical throughput risk.

## Falsifiable Hypothesis

Late EMA will reduce residual SGD/checkpoint variance without weakening early
learning, raising `best_test_acc` from 94.07% to at least the required
**94.17%**. A supporting signature is final EMA accuracy at or above 94.17%
with final test loss no worse than EXP-002's 0.2432 and at least 95% of
EXP-002's 141.9 data-equivalent passes (about 134.8 passes).

The mechanism is falsified by any valid run below 94.17%. More specifically:

- normal exposure plus unchanged or worse test loss means the late trajectory
  is already sufficiently stable, or the EMA/BatchNorm state approximation
  adds no useful generalization;
- a regression with materially reduced exposure is a throughput failure and
  does not cleanly reject weight averaging itself;
- a good intermediate EMA evaluation followed by a worse final result suggests
  decay 0.999 lags the rapidly moving late solution rather than smoothing a
  stationary basin.

No decay or start-time adjustment should be made after seeing the run. A
completed fixed-seed result is one experiment, not a sweep.

## Determinism and Constraint Compliance

The EMA path performs no random operation, so the CUDA/CPU random streams and
the live model's fixed-seed minibatch prefix remain unchanged. Its runtime cost
can end the time-budgeted run at an earlier step, which is an intended part of
the tested method rather than a paired-trajectory guarantee. Model construction
must occur without reinitializing a second network; it should deep-copy the
already initialized live model so no RNG is consumed.

The experiment stays within `train.py`, uses existing PyTorch APIs, one H20,
the unchanged 300-second counted budget, and no more than one evaluation per
epoch. Run exactly once with seed 42 and the mandated redirected local command;
do not retry a completed run or compare multiple EMA decays. Before the full
run, perform only a matched warm timing preflight and require projected exposure
of at least 134.8 passes. Verify the diff, GPU identity, one-time EMA start,
evaluation model selection, 300-second accounting, sub-600-second total time,
and complete summary.

## Risks and EXP-006 Recommendation

This is a **medium-confidence EXP-006 candidate**. It should be considered now
because it composes with every accepted component, is orthogonal to the three
failed augmentation refinements, has negligible memory cost, and can produce a
clean yes/no result in one run. Unlike a width change, it does not trade away a
large fraction of the successful 142-pass exposure.

The case against selecting it is equally concrete: EXP-002's final accuracy
already equals its best, so there may be little iterate variance left to remove;
EMA gains are often below the task's 0.10-point acceptance threshold; averaged
BatchNorm buffers are an approximation; and per-step shadow updates could cost
enough steps to cancel a small statistical gain. It should rank behind a
deconfounded capacity increase only if that candidate's timing preflight shows
near-baseline exposure and a stronger mechanism. Otherwise, late EMA is the
lowest-risk orthogonal EXP-006 test and is preferable to another mixup cutoff or
strength trial.

## Sources

- `.autoresearch/goals/maximize-cifar10-test-accuracy/knowledge/papers/weight-averaging.md`
- `.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/001/04-analysis.md`
- `.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/002/04-analysis.md`
- `.autoresearch/goals/maximize-cifar10-test-accuracy/03-experiment-learnings.md`
