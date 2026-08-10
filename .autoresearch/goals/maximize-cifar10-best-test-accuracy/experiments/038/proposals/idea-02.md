# Proposal: Batch-192 Linear-Scale Pareto Knee

## Decision

Raise the accepted EXP-010 training batch from 128 to 192 and scale the full
learning-rate curve by exactly 1.5: `0.1/0.01/1e-4` becomes
`0.15/0.015/0.00015`. Preserve the width-2 postactivation ResNet-20,
N1/M7 plus 50% alpha-1 CutMix strong phase, hard weak tail, ordinary momentum
SGD, all-parameter weight decay, and all other training semantics. Replace the
batch-dependent dense-tail evaluator trigger with exactly 19 elapsed-progress
checkpoints, matching accepted EXP-010 and rejected EXP-013.

This is a new Pareto-point hypothesis, not a retry of EXP-013. Batch 256's
fresh-pair result established a stable endpoint: it delivered 1.18914x image
throughput but retained only `1 / 1.68189 = 59.46%` of batch-128 optimizer
updates, missing its pre-registered exposure premise. Batch 192 is the exact
midpoint in batch and LR. Its purpose is to retain roughly three quarters of
the optimizer decisions while capturing a useful fraction of the image-rate
gain. It must earn a production run through new paired timing and
update/exposure gates; EXP-013's threshold is neither relaxed nor reinterpreted.

## Exact Source Changes

Starting from accepted commit `7c1e7d8`, change only `train.py`:

```python
BATCH_SIZE = 192
LR = 0.15
ANNEAL_START_LR = 0.015
MIN_LR = 1.5e-4
EVAL_CHECKPOINTS = (
    0.2,
    0.4,
    0.6,
    0.7,
    *(0.8 + 0.2 * index / 14 for index in range(15)),
)
```

At the evaluation site, remove `dense_tail_due = progress >=
LR_HOLD_FRACTION` and change the condition to
`if checkpoint_due or training_done:`. Retain the existing `while` advancement
so delayed crossings cannot cause repeated observations. A simulated trace
using measured candidate step times must prove that all 19 progress thresholds
map to 19 unique epochs. No other source line may change.

The trailing checkpoint is exactly 1.0. If floating-point timing causes the
normal final `training_done` evaluation to satisfy the last threshold in the
same epoch, the single existing evaluator call still counts once; the trace and
production log must show exactly 19 unique evaluator calls, not 18 or 20.

Explicit exclusions: no warmup, momentum or decay adjustment, gradient
accumulation, AMP, channels-last, compilation, fused optimizer, batch fallback,
augmentation change, CutMix adjustment, second LR branch, or post-result LR
tuning. Evaluation batch size in `prepare.py` remains unchanged and independent
of the training batch.

## Quantitative Mechanism

Both batch sizes consume exactly 49,920 image slots per complete shuffled pass
because `drop_last=True`:

```text
batch 128: floor(50,000 / 128) = 390 updates; 390 * 128 = 49,920
batch 192: floor(50,000 / 192) = 260 updates; 260 * 192 = 49,920
```

Linear scaling preserves the first-order LR mass per complete pass at every
schedule level:

```text
plateau: 390 * 0.10000 = 260 * 0.15000 = 39.000
tail start: 390 * 0.01000 = 260 * 0.01500 = 3.900
tail floor: 390 * 0.00010 = 260 * 0.00015 = 0.039
```

With unchanged coupled decay `1e-4`, the first-order decay shrinkage per pass is
also preserved. Batch/LR is exactly constant:
`128 / 0.1 = 192 / 0.15 = 1,280`, matching the generalization-sensitive ratio
identified by He, Liu, and Tao. The 10x LR discontinuity at the 80% strong-to-
weak transition and the cosine curve's relative shape are unchanged.

This is only a first-order invariance. Momentum `0.9` remembers about ten
updates, so its image horizon increases from about 1,280 to 1,920 slots;
gradient and BN noise decrease; each CutMix draw groups more samples; and
finite-curvature behavior at LR 0.15 can differ. Those are the intended
batch-192 tradeoffs, not nuisances to tune away.

The fresh EXP-013 endpoints permit a transparent timing prior. Linear
interpolation between its batch-128 and batch-256 median trial means predicts

```text
t192 ~= (10.8437 + 18.2380) / 2 = 14.5409 ms
step ratio ~= 1.3410
image-throughput ratio ~= 1.5 / 1.3410 = 1.1186
projected updates ~= 26,898 / 1.3410 = 20,058
projected slots ~= 20,058 * 192 = 3.851M
```

Thus the point prediction is about 74.6% of accepted updates and 11.9% more
image slots, versus batch 256's measured 59.5% updates and 18.9% more image
slots. The midpoint would retain about 25% more optimizer decisions than the
measured batch-256 projection while giving up only about 5.9% of its image
slots. Kernel scaling need not interpolate linearly, which is why this estimate
cannot satisfy the launch gate.

## Evidence and Why This Is New

EXP-010 provides the accepted trajectory: 26,898 updates, about 3.443M image
slots, 89.73% at the 80% switch, 93.16% at the first weak checkpoint, 94.15%
best/final accuracy, 0.1934 final NLL, and 598.7 MB peak allocation. It also
shows that class-bearing regional mixing, rather than raw exposure, is valuable;
the candidate therefore preserves its entire data recipe.

EXP-013 produced useful negative feasibility evidence rather than an accuracy
result. Five fresh pairs measured a highly stable 1.68189 step ratio (control
CV 0.474%, candidate CV 0.197%), showing both that batch scaling is real on the
H20 and that batch 256 lies just beyond the experiment's chosen update/exposure
balance. Its analysis explicitly identified batch 192 as unexplored, conditional
on a new knee measurement and independent 1.5x LR review. The new proposal
answers that condition rather than weakening EXP-013 after the fact.

The NeurIPS 2019 batch/LR study supplies the independent mechanistic constraint:
prevent the batch-to-LR ratio from increasing when scaling batch size. The exact
1.5x curve does so across plateau, tail start, and endpoint. This does not prove
local CIFAR-10 accuracy, but it rules out the under-scaled-LR variant and makes
192 a coherent operating point rather than an arbitrary batch-only test.

Finally, later experiments add a caution absent from the original EXP-013
brainstorm: global-LR optimizer-path changes can create transient class
concentration even when simple scale arguments pass. Batch 192 therefore gets
an immutable-corpus, image-aligned safety gate before timing. The scaled batch
is expected to reduce gradient noise, but LR 0.15 must demonstrate that claim
safely rather than inheriting safety from batch 128 or 256.

## Expected Impact

If the measured timing is near the interpolation prior, the candidate should
process about 3.80-3.90M slots (76-78 nominal dataset passes) while retaining
about 19.8k-20.3k optimizer updates. The hypothesis is that this balance gives
enough extra strong-view and hard-tail exposure to offset the reduced update
noise without incurring batch 256's 40.5% loss of optimizer decisions.

Point prediction: `best_test_acc = 94.30%`, with a plausible single-run range
of roughly 94.00-94.40%. The formal success threshold remains 94.25%. The
expected gain is modest because batch scaling changes optimization efficiency,
not representation capacity; a neutral result is at least as plausible as a
gain. The experiment is valuable only if the pre-registered Pareto premise and
healthy trajectory both execute.

## Functional and Safety Gates

Before performance timing:

1. Prove the tracked diff contains only the four constants and fixed-count
   evaluation control. Compile, Ruff, formatting, pre-commit, and diff checks
   must pass.
2. From seed 42, require candidate/control models to have identical state dicts,
   parameter ordering, RNG state after construction, 1,073,962 parameters, and
   one ordinary SGD group with momentum `0.9` and decay `1e-4`.
3. Require both train loaders to consume 49,920 slots per pass, with lengths
   390 and 260; preserve `drop_last`, eight workers, 45-55% CutMix incidence,
   probability-target row sums of one, hard weak targets, and clean worker
   replacement/shutdown.
4. Check LR values at progress 0, 0.8, immediately after 0.8, and 1.0 against
   the declared 1.5x curve; verify per-pass LR and decay products numerically.
5. On a persisted production-distribution corpus packed into 384-image chunks,
   compare three batch-128 control updates with two batch-192 candidate updates,
   so each checkpoint aligns both image exposure and LR mass. Run two accepted
   controls plus the candidate through at least 60 strong chunks (180 versus
   120 updates) and 20 weak chunks (60 versus 40 updates). Require finite
   logits, losses, gradients, momentum, BN state, and parameters throughout.
6. Qualify every candidate safety statistic against both controls. Reject only
   candidate-specific events: more than 95% of predictions in one class when
   neither control does so, non-finite state, persistent loss or logit RMS
   outside `[0.2, 5.0]` times the control envelope, or parameter/update RMS more
   than 5x a denominator-safe control envelope at an image-aligned checkpoint.
   Record initial, final-strong, first-weak, and final-weak class histograms,
   entropy, loss, logit RMS, gradient RMS, update RMS, and BN running-stat
   divergence. No unqualified absolute ratio may veto the candidate.

The 384-image construction is for safety and trajectory geometry only; it is
not a throughput proxy and cannot replace real batch-192 timing.

## Paired Timing and Update-Exposure Gates

On one idle 97,871 MiB H20, run seven alternating fresh-process pairs. Each arm
must recreate seed-42 model and optimizer state, use the complete hard/soft
target training path, warm up 100 steps, and time 500 synchronized H2D,
forward, CE, backward, and SGD steps. Alternate which arm runs first. Record all
trial means, medians, p95s, CVs, image rates, and peak allocation.

The candidate advances only if all conditions hold:

- candidate/control median-of-trial-mean step ratio `<= 1.3889`;
- image-throughput ratio `1.5 / step_ratio >= 1.08`;
- ratio-projected updates `floor(26,898 / step_ratio) >= 19,366`, retaining at
  least 72.0% of accepted optimizer decisions;
- ratio-projected slots `projected_updates * 192 >= 3,718,272`, at least 8.0%
  above EXP-010's 3,442,944 slots;
- projected updates are at least 20% above EXP-013 batch 256's measured 15,992,
  while projected slots are at least 90% of its 4,093,952;
- candidate p95 image throughput is at least 1.05x control, trial-mean CV is
  below 3% in each arm, all values are finite, and candidate peak allocation is
  below 900 MB.

These joint gates define the new knee: an 8% minimum exposure gain while
preserving materially more updates than batch 256. They deliberately do not
reuse EXP-013's 20% exposure floor, because that floor justified a different
point which discarded about 40% of updates. Conversely, passing only one side
is insufficient: batch 192 may not proceed merely because it is faster in
images or merely because it retains updates.

After synthetic timing, audit at least 1,000 integrated production strong steps
and a weak-loader cycle. Require loader delivery at least 1.25x candidate GPU
consumption, p95 iterator wait no greater than 20% of candidate mean step,
wall/count ratio no greater than 1.05, 45-55% mixing, healthy workers, finite
state, and a conservative total projection below 540 seconds. Benchmark one
unchanged evaluator pass and simulate candidate epochs; require exactly 19
unique evaluation opportunities and no extra final duplicate.

## Production Trajectory Gates and Decision Rule

If every preflight gate passes, execute exactly one seed-42 production run with
stdout/stderr redirected to `run.log` and a 600-second timeout. Never stop or
retune based on a low but finite checkpoint.

Integrity requires 300 counted seconds, total below 600 seconds, exactly 19
unique evaluator calls, one switch near 80%, eight strong workers stopped,
45-55% strong CutMix batches, only hard targets afterward, 1,073,962 parameters,
and actual steps/slots satisfying the same 72%/8% paired-exposure floors. A
timing projection that passes but an actual exposure floor that fails makes the
mechanism attribution invalid; it is not permission to rerun.

Pre-register these trajectory interpretations:

- `80% switch accuracy >= 87.08%`: minimum non-underfit gate inherited from
  EXP-011/012; expected range is 88.5-90.0% around EXP-010's 89.73%;
- first weak checkpoint `>= 92.50%`: evidence that the scaled hard tail recovers
  promptly, against EXP-010's 93.16%;
- final NLL `<= 0.2050` and best-final gap `<= 0.20` points: evidence that any
  max-accuracy gain is not a single unstable spike;
- no late candidate-only class concentration or non-finite state.

These are mechanism/trajectory gates, not opportunities for early stopping or
adaptive rescue. The primary verdict remains: `best_test_acc >= 94.25%` with
all protocol-integrity conditions is an improvement; a complete lower result
is no-improvement and cannot be rerun. If the primary metric passes while a
healthy-trajectory expectation misses, report the metric truthfully but do not
claim that the planned exposure/update mechanism was cleanly supported.

## Principal Risks

- **Lost optimizer decisions:** even the predicted knee removes about one in
  four accepted updates; extra examples may not replace the stochastic search
  and frequent parameter decisions that benefit the short schedule.
- **LR-0.15 transient instability:** preserved LR mass and batch/LR ratio do not
  bound curvature, momentum overshoot, early logit scale, or class collapse.
- **Reduced useful noise:** batch 192 may generalize worse even with a constant
  batch/LR ratio; PAC-Bayes correlation is not a local guarantee.
- **Longer momentum image horizon:** fixed momentum integrates 50% more examples
  per nominal memory window, changing response speed at the abrupt 80% LR and
  data-policy switch.
- **BN and CutMix composition:** larger batches change BN estimates and the
  donor/recipient composition of each mixed target, so a gain cannot be
  attributed to raw slot count alone.
- **Nonlinear H20 knee:** batch 192 may land on an unfavorable kernel algorithm
  or provide too little image gain despite abundant memory.
- **Evaluation-control interaction:** fixed progress checks remove max-over-more-
  looks bias but differ from the accepted per-epoch tail cadence; this is a
  required measurement control and the result is the net method.
- **Single-seed resolution:** the 0.10-point gate is ten test examples. A bare
  pass is protocol-valid but weak causal evidence and cannot be confirmed by a
  reroll.

## Evidence

- `goals/maximize-cifar10-best-test-accuracy/01-definition.md`: only-`train.py`,
  fixed seed/time/device/evaluator protocol, and 94.25% moving gate.
- `goals/maximize-cifar10-best-test-accuracy/02-system-understanding.md`: model
  backward dominates step cost, memory is idle, and batch scaling is an open
  fixed-time exposure question.
- `goals/maximize-cifar10-best-test-accuracy/experiments/010/04-analysis.md`:
  accepted 94.15% trajectory, 26,898 updates, 598.7 MB, and healthy CutMix
  strong-to-weak transition.
- `goals/maximize-cifar10-best-test-accuracy/experiments/013/04-analysis.md` and
  `00-paired-timing.md`: stable batch-256 1.68189 step ratio, 1.18914x image
  throughput, 15,992 projected updates, and explicit batch-192 avenue.
- `goals/maximize-cifar10-best-test-accuracy/03-experiment-learnings.md`: fresh
  alternating process timing is mandatory; global-LR changes need
  production-distribution concentration controls.
- `experiments/038/papers/control-batch-size-and-learning-rate.md`: empirical and
  theoretical evidence to preserve batch/LR ratio during batch scaling.
