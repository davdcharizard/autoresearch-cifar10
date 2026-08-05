# Proposal: Batch 128 With a Fully Scaled LR Curve

## Summary

Change the accepted EXP-027 operating point from batch 256 with a
`0.2 -> 0.002` time-cosine learning-rate curve to batch 128 with an exactly
halved `0.1 -> 0.001` curve. Preserve the accepted `(2,2,3)` WRN, worker-safe
early RandAugment, alpha-0.2 batch-shared mixup through 65% counted time, FP32
Nesterov SGD, weight decay, seed, and evaluation cadence. Double only the
nonbinding safety step cap so the run remains time-terminated.

The treatment tests whether finer optimizer, BatchNorm, and batch-shared mixup
decisions improve the new 94.32% model's generalization boundary. It is more
defensible than batch 128 at LR 0.2, which would also double LR per sample and
make instability or changed diffusion inseparable from batch granularity. It is
also better motivated than a larger batch: EXP-009 and EXP-016 already show
that substantially greater image exposure alone did not improve the prior
learner.

## Current Limiter and Mechanism

The accepted model is compute-bound inside training, not by input, memory, or
the 600-second wall limit. EXP-027 completed 25,978 updates and 133.00736 data
passes in 300 counted seconds, while backpropagation accounts for about 74% of
an isolated step. It reached 94.32% best / 94.22% final accuracy with 0.2523
final loss and nearly interpolated its training tail. The remaining problem is
therefore boundary quality per backward pass rather than lack of raw examples.

At equal image exposure, batch 128 produces twice as many parameter updates,
each from a noisier gradient estimate. Halving the complete LR curve preserves
the first-order LR-per-sample scale and approximately preserves accumulated SGD
diffusion per processed example. The proposed signal is thus not an arbitrary
twofold noise injection. It is finer discretization plus the smaller-batch
stochasticity that linear scaling does not neutralize: BatchNorm uses half as
many examples per statistic, mixup draws one coefficient for groups of 128
instead of 256, and optimizer state is advanced twice as often.

This is an operating-point experiment, not a pure batch-size ablation. With
momentum fixed at 0.9, its memory horizon in examples is approximately halved.
BatchNorm's default running-stat update also happens twice as often per data
pass. Twice as many half-LR applications make cumulative matrix decay per data
pass approximately, but not exactly, equal to accepted. These changed update
semantics are integral to the predetermined batch/LR treatment and must not be
retuned after observing timing or accuracy.

## Exact Production Change

Starting from accepted commit `67c8e98`, modify only these constants in
`train.py`:

```python
BATCH_SIZE = 128       # accepted: 256
LR = 0.1               # accepted: 0.2
MIN_LR = 0.001         # accepted: 0.002
MAX_STEPS = 128000     # accepted: 64000
```

`MAX_STEPS` is a semantics-preserving guard adjustment, not an optimization
lever. Both configurations then cap at 16,384,000 processed examples. Leaving
the cap at 64,000 would halve the candidate's example allowance and could stop
a sufficiently fast run at 163.84 passes before its 300-second budget.

No other constant or code path may change. In particular, retain:

- `STAGE_BLOCKS=(2,2,3)`, width factor 2, and 987,098 parameters;
- momentum 0.9, Nesterov, `5e-4` matrix decay, no decay on vectors, FP32, and
  `zero_grad(set_to_none=True)`;
- five-percent counted-time warmup followed by the same cosine formula;
- batch-shared `Beta(0.2,0.2)` mixup until progress 0.65 and hard labels after;
- worker-private one-operation magnitude-5 RandAugment and its exhausted-epoch
  cutoff at or after progress 0.65;
- crop/flip, loader workers/pinning/shuffle/drop-last/persistence/prefetch and
  multiprocessing context;
- seed 42, counted-time accounting, evaluator, and once-per-epoch cadence.

Do not preserve the accepted 0.002 floor as a fallback, change momentum to
compensate for update frequency, accumulate gradients, or try an intermediate
LR. The fully scaled curve is one indivisible treatment. The 0.001 floor remains
nonzero, but it deliberately tests a weaker terminal per-step amplitude; this
is a risk in light of EXP-008 and not grounds for an adaptive retry.

## Preserved Epoch and Temporal Semantics

With 50,000 training examples and `drop_last=True`, batch 256 yields 195 batches
and batch 128 yields 390. Both epochs contain exactly 49,920 examples and drop
80. Consequently, epoch boundaries and every-fifth-epoch evaluations retain
the same example-domain meaning. The LR and mixup transitions remain based on
counted seconds, not step or epoch count.

RandAugment must still switch off only after fully exhausting the first iterator
whose ending counted time is at least 195 seconds. It may therefore lag the
per-step mixup transition by less than one candidate epoch, now fewer than 390
steps rather than 195. The shared byte must never flip during a live iterator,
on a budget break, or more than once.

## State and RNG Controls

Run all preflights in separate processes so they cannot consume the scored
run's RNG. Before timing, require:

- fixed seed 42 constructs a byte-identical 987,098-parameter model and leaves
  identical parent CPU/CUDA RNG state when accepted and candidate constants are
  compared before loader iteration;
- changing the four constants does not alter module construction order,
  initialization, optimizer parameter membership, or initial optimizer state;
- the LR function is exactly one half of accepted at progress
  `0, 0.025, 0.05, 0.50, 0.65, 1.0`, including 0.1 peak and 0.001 floor;
- a finite candidate production step yields `[128,10]` logits, finite loss and
  gradients, and two disjoint decay/no-decay parameter groups;
- a batch-128 marker-loader replay proves that after one active iterator is
  exhausted and the shared flag flips, every item in the next iterator is
  inactive; paired clean-tail replay must preserve accepted crop/flip outputs;
- source audit proves exactly one non-reenable cutoff path, the unchanged
  worker-private RandAugment RNG swap/restore, and no evaluator or test-data use.

Do not demand identical accepted and candidate RNG trajectories after training
iteration begins. Batch partitioning changes sampler delivery among workers;
`randperm(128)` consumes a different CUDA RNG path; there are twice as many
batch-shared Beta draws per epoch; and worker/BatchNorm histories change. Those
are intended consequences, not seed rerolling. The controls instead prove that
the fixed seed and pre-iteration model oracle are unchanged and that
RandAugment remains isolated from the clean tail.

## Matched Timing and Feasibility Gates

### GPU exposure

In one local process on the H20, benchmark accepted batch 256 and candidate
batch 128 with separate byte-identical models and optimizers. Use production
FP32 forward/backward/Nesterov steps on device-resident synthetic CIFAR tensors.
Measure the real mixup and hard-label paths independently after at least 20
warmup steps. Use balanced randomized `A-B-B-A` windows driven by a private
timing RNG, synchronize CUDA around steps, and require at least three measured
windows per configuration/path with every window CV at most 5%.

Because the phase boundary is based on counted time, define each path's weighted
image rate as:

```text
image_rate(B) = 0.65 * B / median_mixup_step_s
              + 0.35 * B / median_hard_step_s
retention = candidate_image_rate / accepted_image_rate
projected_passes = 133.00736 * retention
projected_updates = projected_passes * 50_000 / 128
```

Calibrating to EXP-027's realized 133.00736 passes avoids treating isolated
CUDA timing as an absolute scored-time oracle; the system profile observed that
its absolute step time differs from the scored mean. Proceed only if all values
are finite, projected passes are at least **120.0**, projected updates are at
least **46,875**, retention is at least **0.9022**, and projected updates remain
below 128,000. The pass floor retains 90.2% of the accepted data exposure while
guaranteeing at least 1.804 times EXP-027's 25,978 optimizer decisions. It
therefore preserves both sides of the stated tradeoff. Do not lower the floor.

### Loader and wall time

Run fresh matched real-training-loader arms for both batch sizes with active and
inactive RandAugment, including the real multiprocessing context, eight
persistent workers, prefetch factor 2, and paced consumer work. Require correct
finite tensors/labels, exactly 195 versus 390 batches, clean worker shutdown,
no active marker in the post-cutoff epoch, and per-arm epoch-time CV at most 5%.
Use complete epochs, since they contain the same 49,920 examples.

Project candidate wall time both by adding any positive candidate-minus-
accepted epoch overhead to EXP-027's 345.3-second wall and by an absolute paced
candidate-epoch estimate with the accepted 45.3-second setup/evaluation margin.
Require both conservative projections at or below **500 seconds**. This is a
fail-closed margin below the 600-second hard timeout. Loader failure, RNG-tail
leakage, timing instability, less than 120 projected passes, or either wall
projection above 500 ends EXP-028 as a feasibility failure without a score.

## One-Run Rule and Result Gates

Only after all semantic, GPU, and loader gates pass, confirm one idle NVIDIA H20,
remove stale `run.log`, and execute exactly once:

```bash
timeout 600s uv run train.py > run.log 2>&1
```

Use seed 42. Never rerun a valid completion, reroll the seed, or alter batch,
peak/floor LR, momentum, step cap, cutoff, or feasibility thresholds in
EXP-028. A crash, timeout, non-finite loss, worker failure, missing summary,
duplicate evaluation epoch, or malformed transition is a failure and does not
authorize a modified replay.

A valid result must satisfy all of the following:

- exit 0, one H20, exactly 300 counted seconds, and total wall below 600 seconds;
- one mixup transition at or just after 195 counted seconds;
- one later exhausted-iterator RandAugment transition with candidate step lag
  in `[0,390)` and no re-enable;
- 987,098 parameters, fewer than 128,000 steps, at least **46,875 steps**, and
  at least **120.0 realized passes** (`num_steps * 128 / 50_000`);
- unique evaluation epochs and `best_test_acc >= 94.42%`, exactly 0.10 points
  above the indexed 94.32% baseline.

Record final accuracy/loss, best-final gap, epochs, steps, passes, both
transition times/steps, counted/wall seconds, and peak VRAM. Final accuracy at
or above 94.32% is useful corroboration but is not an extra formal metric gate.
A valid sub-94.42% score rejects this exact operating point. Whether the miss
comes from the weaker floor, shortened example-domain momentum/BN horizon, or
lost image throughput may inform later ideation, but none may be tuned inside
EXP-028.

## Hypothesis

If the accepted deeper-plus-invariance model is limited by coarse batch/update
granularity, then batch 128 with the fully scaled `0.1 -> 0.001` LR curve will
retain at least 120 data passes, execute at least 46,875 optimizer steps in 300
counted seconds, and improve one fixed-seed `best_test_acc` from 94.32% to at
least 94.42%.

## Evidence and Risks

- `02-system-understanding.md`: training/backward compute is binding, while
  input, memory, and wall headroom are not; this mandates direct H20 timing.
- `experiments/027/04-analysis.md`: the current base scored 94.32% at 133.007
  passes and 25,978 steps; its depth-plus-early-invariance interaction must stay
  intact.
- `experiments/001/04-analysis.md`: batch 256 / LR 0.2 was introduced by linear
  scaling from the original batch-128 / LR-0.1 regime, supporting these exact
  peak values as a conservative lineage rather than a sweep result.
- EXP-009 and EXP-016 reached 159.07 and 171.70 passes but regressed, so a larger
  batch justified only by throughput has weak local evidence.
- EXP-008 shows that inadequate terminal update amplitude can harm refinement.
  A 0.001 floor is nonzero but may still be too weak; this is the proposal's
  clearest accuracy risk and is deliberately not repaired post hoc.
- Smaller batches may lose H20 image throughput, shorten momentum and BatchNorm
  horizons in example units, and refresh batch-shared mixup coefficients twice
  as often. The strict paired timing/exposure gate prevents a severe throughput
  confound, while the one-run rule keeps the optimizer treatment falsifiable.
