# Proposal: Horizon-Derived Clean-Tail EMA with Exact Evaluation Swap

## Summary

Add an inference-only exponential moving average (EMA) to the accepted EXP-004
WRN/CutMix/SAM training path. Begin at charged progress 0.75, exactly when
CutMix ends and period-two SAM begins. Sample the post-optimizer online state
every 31 steps. Derive each decay from elapsed charged time and a preregistered
18.75-second half-life rather than importing a conventional scalar momentum.

Store EMA tensors separately, but keep only one functional model. For each
post-start evaluation, snapshot the online model state, copy EMA parameters and
buffers into the live model, run the evaluator exactly once, and restore the
online state in `finally`. Before EMA activation, evaluate the online model.
There is no second evaluation, ensemble, extra forward pass, or uncharged
BatchNorm recalibration.

The online training recipe remains EXP-004: full front-loaded CutMix, all six
drop-path draws, clean-tail rho-0.05 SAM on every second eligible one-based
step, BF16/channels-last WRN-16-4, Nesterov SGD, time-based LR, batch 256, and
seed 42.

## Diagnosis

The current limiter is a detectable generalization gain, not memory. EXP-004
reaches 95.40% in 25,560 steps while using 1,190.5 MiB of a 97,871 MiB H20.
Its later children changed sample history, augmentation allocation, or SAM
geometry and did not improve. EXP-006's final four accuracies span 0.15 points
despite a loss range of only 0.0011, while prior selected runs reversed by
0.14-0.29 points. EMA directly targets late-iterate/model-selection variation
without replacing validated data or gradients.

The SWA reference reports gains from averaging later SGD points in CIFAR
residual networks and interprets the mean as a more central solution in a wide
basin. The EMA-scaling reference shows that update frequency and momentum must
be tied through an effective horizon. This benchmark is wall-clock scheduled
and SAM steps are about twice as slow as ordinary steps, so a fixed `0.99` or
`0.999` per update has no stable interpretation. Charged-time decay does.

## Preregistered Horizon, Start, and Cadence

Define:

```python
EMA_START = 0.75
EMA_UPDATE_EVERY = 31
EMA_TAIL_HALF_LIVES = 4.0
EMA_HALF_LIFE_S = (
    (1.0 - EMA_START) * TIME_BUDGET_S / EMA_TAIL_HALF_LIVES
)
```

With a 300-second budget, the clean tail is 75 seconds and the half-life is
18.75 seconds. Four half-lives leave the first tail sample with nominal weight
`0.5**4 = 0.0625` at the end: early clean/SAM state remains represented but
cannot dominate. This choice is derived from the fixed horizon, not selected
from accuracy.

EXP-004 completed 4,898 tail steps, so cadence 31 should produce approximately
158 EMA samples at intervals near 0.48 charged seconds, only 2.6% of one
half-life. This is fine enough to approximate continuous EMA while keeping
parameter traffic sparse. Because 31 is odd while SAM uses even steps, samples
alternate post-ordinary and post-SAM parity rather than selecting one optimizer
subsequence.

On the first eligible cadence point, copy online state exactly into EMA. On
later points use the actual elapsed charged timestamp:

```python
delta_s = total_training_time - ema_last_training_time
decay = 2.0 ** (-delta_s / EMA_HALF_LIFE_S)
ema.lerp_(online, 1.0 - decay)
ema_last_training_time = total_training_time
```

The product over any interval is
`2**(-sum(delta_s)/EMA_HALF_LIFE_S)`, so changing cadence or losing a few steps
to bookkeeping does not change the intended time horizon. The timestamp is the
existing accumulated charged time at step start; its bounded one-step offset is
small relative to 18.75 seconds and consistent across samples.

Eligibility is `progress >= 0.75 and next_step % 31 == 0`. Update only after
SAM parameters have been restored and after the sole `optimizer.step()`. Keep
EMA tensor operations before the existing CUDA synchronization and `dt`
calculation, charging all their cost. Never sample perturbed SAM weights.

## Shadow-State Implementation

Do not construct a second model. Preallocate three name-checked groups on the
same device before charged training:

1. `ema_parameters`: `empty_like` tensors for every online parameter;
2. `ema_float_buffers`: `empty_like` tensors for persistent floating buffers,
   specifically BN running means and variances;
3. `ema_int_buffers`: `empty_like` tensors for integral buffers, specifically
   BN `num_batches_tracked`.

Build parallel name-keyed online lists and assert names, shapes, dtypes,
devices, and memory formats. EMA tensors have no gradients and are absent from
the optimizer, SAM parameter list, and SAM snapshots.

At the first sample, use `torch._foreach_copy_` for compatible parameter and
floating-buffer lists and `copy_` for integer buffers. At later samples use
`torch._foreach_lerp_` for parameters and floating buffers with the derived
coefficient, and copy each integer counter from online state. BN affine weights
are ordinary parameters and are averaged.

Do not average gradients, optimizer momentum, SAM snapshots, RNG states,
CutMix counters, or schedules. The shadow is never used to generate training
outputs or gradients.

## BatchNorm Without Recalibration

Average BN `running_mean` and `running_var` with exactly the same elapsed-time
coefficient as model weights. Copy `num_batches_tracked` at each sample because
an integer counter cannot be meaningfully averaged. This gives the EMA model a
self-consistent smoothed inference state.

The approach is approximate: averaged moments need not equal moments generated
by averaged weights. Using the latest online moments would instead pair
lagged weights with statistics dominated by recent batches. Leaving initial
moments is invalid. A canonical SWA recalibration pass would traverse training
images again; doing it outside the charged timer is free training work, while
charging it would remove validated CutMix/SAM updates. Therefore no
recalibration pass is allowed.

The online BN path stays exact. SAM's primary pass updates each online BN once;
the replayed second pass keeps tracking disabled. EMA samples the resulting
post-update buffers and never changes them through inference.

## Single-Model Evaluation Swap and Exact Restore

Preallocate `eval_restore` tensors for every parameter and persistent buffer.
At an epoch's sole evaluation:

```python
if ema_updates == 0:
    test_loss, test_acc = evaluator.evaluate(model, device)
    eval_source = "online"
else:
    was_training = model.training
    copy_online_state_to(eval_restore)
    try:
        copy_ema_state_to(model)
        test_loss, test_acc = evaluator.evaluate(model, device)
        eval_source = "ema"
    finally:
        copy_restore_state_to(model)
        model.train(was_training)
```

Copy full parameters, floating BN buffers, and integer counters in both
directions. Do not swap optimizer state: parameter objects remain the same and
only their data changes temporarily. `try/finally` is mandatory so an evaluator
exception cannot leave EMA weights attached to online momentum.

After restoration, compare every live tensor with its restore snapshot using
exact equality and raise on any mismatch. Also assert optimizer parameter
object identities and momentum-buffer object identities are unchanged. These
checks occur in the evaluation-excluded region and do not influence training.

Before progress 0.75, evaluation source is online. From the first EMA sample
onward, source is EMA only. Never evaluate both in one epoch. `best_acc` spans
the one preregistered source available at each epoch, and final metrics are EMA.
The live model is always restored before the next `model.train()` and batch.

## RNG and Parent-State Parity

EMA allocation, copy, lerp, swap, equality checks, and restore use no random
draws. They must leave CPU global RNG, CUDA global RNG, data-loader workers,
and dedicated CutMix CPU/CUDA generators untouched. The first 75% of charged
training is exactly the parent because EMA does no work before the transition.

After 0.75, each shared online step preserves parent ordering: primary
forward/backward, optional SAM snapshot/perturb, CUDA RNG replay and BN
suppression, exact SAM restore, one optimizer update, then EMA sampling. The
only possible dose difference is a small reduction in late steps from charged
EMA overhead; the latency gate bounds it.

Add deterministic parity tests that snapshot every RNG state immediately
before and after an EMA update and an EMA evaluation swap. All must match.
Verify online parameters, buffers, optimizer momentum, gradients, module mode,
and RNG state are bit-identical before versus after the complete evaluation
swap/restore transaction.

## Cost and Parent-Relative Gate

The 2.75M FP32 parameters occupy about 10.5 MiB. EMA state plus evaluation
restore storage should add roughly 22-30 MiB including buffers and metadata,
keeping peak allocation near 1.22 GiB. Approximately 158 foreach EMA updates
move only a few GiB in total over 75 seconds. There is no added network pass.

Before the metric run, compare actual EXP-004 and EMA-integrated loops on
physical GPU 0, batch 256, BF16/channels-last, without compilation. Measure at
least 200 ordinary iterations, 100 production-faithful SAM iterations, and
enough 31-step sequences to include at least 30 EMA updates after warmup. Use
alternating randomized-order rounds and synchronization. Also benchmark the
EMA swap/evaluate/restore path against parent evaluation.

Proceed only if:

- weighted candidate training latency is at most `1.02 * parent`, using
  EXP-004's observed approximately 90.4% ordinary / 9.6% SAM step mix;
- projected 300-second exposure is at least 25,200 steps;
- swap/restore exactness and all RNG/optimizer/BN assertions pass;
- projected total runtime remains below 600 seconds;
- peak VRAM remains below 1.30 GiB and no OOM/nonfinite error occurs.

All thresholds are same-harness parent-relative; no absolute throughput floor
can reject the measured parent.

## Instrumentation and Durable Audit

Log constants in the setup line. Record and print:

- EMA update count; first/last step, progress, and charged timestamp;
- interval and realized-decay min/mean/max;
- total EMA span and cumulative coefficient on the first sample;
- online/EMA evaluation counts and the source of every evaluation;
- swap count, restore-check count, and zero mismatches;
- final online-to-EMA FP32 parameter absolute and relative L2 distance;
- final per-BN running-mean/variance distance summaries;
- per-BN EMA-to-live running-variance ratios, to diagnose the known limitation
  that linear variance averaging omits between-state mean shifts;
- CutMix and SAM counts already required by EXP-004.

Do not call `.item()` on model tensors during charged training. At every sample,
compute the consecutive-sample squared parameter distance into a no-grad GPU
scalar, retain those small scalars without synchronizing, and copy the current
online state into a dedicated previous-sample buffer. Read the stacked scalars
only after charged training. Compute EMA-to-live and BN distances after charged
training/evaluation boundaries. Before deleting `run.log`, copy the exact
terminal summary, EMA audit line, CutMix/SAM counts, and source counts into the
execution report so the result remains independently reviewable.

Expected audit values are about 158 updates, first progress near 0.75, roughly
75 seconds EMA span, cumulative initial coefficient near 0.0625, EMA-only late
evaluations, and zero restoration mismatches.

## Expected Effect and Falsification

Parent EXP-004 is 95.40%, so formal improvement requires at least 95.50%. The
strong hypothesis is:

> Horizon-derived clean-tail EMA will reach `best_test_acc >= 95.70%` (at least
> +0.30 points), retain at least 25,200 steps, preserve complete CutMix and
> period-two SAM semantics, and add less than 0.11 GiB peak VRAM.

A realistic expected range is 95.45-95.70% (+0.05 to +0.30). The parent has no
documented best-to-final gap and its cosine tail may lack SWA-style diversity,
so the upper end requires genuine basin-centering rather than checkpoint
smoothing alone. The formal 95.50 and meaningful 95.70 thresholds remain fixed.

Run one fixed-seed configuration. Below 95.50% is a tree no-improvement.
95.50-95.69 formally improves but falsifies the meaningful >=0.30-point
hypothesis. Do not tune start, half-life, cadence, BN policy, or a live/EMA blend
after observing test accuracy.

## Risks

- SAM and EMA may be redundant: SAM already ended with best equal to final and
  improved loss, leaving little trajectory noise for averaging.
- The cosine LR falls from about 0.034 to 0.002 during the tail; iterates may be
  too correlated for the broader-optimum effect reported under SWA schedules.
- A four-half-life horizon can lag a late improving model or average across a
  changing drop-path scale. The 6.25% start residue bounds, but does not remove,
  this risk.
- Averaged BN buffers are an approximation and may mismatch averaged weights.
  No post-hoc recalibration or alternate live-buffer result is permitted.
- Late online checkpoints become invisible after the source switch. Measuring
  both would violate the once-per-epoch and no-model-selection contract.
- A faulty restore could poison all later training. Preallocated full-state
  snapshots, `finally`, exact equality, and optimizer identity checks are hard
  requirements.
- Charged shadow updates can slightly reduce SAM pulses. The 2% latency and
  25,200-step gates bound this confound, and exact counts must be reported.
- Protocol noise is 0.14-0.29 points; a marginal accepted delta remains weaker
  evidence than the >=95.70 target.

## Verification

1. On scalar tensors, verify two half-lives give coefficient 0.25 and that
   products across cadence partitions equal `2**(-elapsed/18.75)`.
2. On a tiny Linear+BatchNorm module, verify first copy, parameter/float-buffer
   lerp, integer-counter copy, no EMA gradients, and optimizer exclusion.
3. Verify full-state EMA swap/restore is bit-exact under success and forced
   evaluator exception, including model mode, parameters, BN buffers/counters,
   optimizer parameter/momentum identities, and all RNG states.
4. On the full WRN SAM smoke, verify EMA samples post-restore/post-optimizer
   state while perturbation norm, replayed masks, one BN update, exact SAM
   restore, and one momentum update remain unchanged.
5. Pass the parent-relative GPU-0 overhead gate above.
6. Launch once with
   `timeout 600s env CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`.
7. Confirm the 97,871 MiB H20, approximately 300 charged seconds, under 600
   total seconds, one evaluation per epoch, expected EMA/CutMix/SAM audits,
   complete summary, and both 95.50 formal and 95.70 meaningful thresholds.
8. Verify only `train.py` changed, seed 42 and evaluator remain fixed, no
   dependency or recalibration pass was added, and durable evidence is copied
   before transient log deletion.
