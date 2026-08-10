# Proposal: Canonical Lookahead Around Nesterov With Existing Full-State EMA

## Summary

Wrap EXP-011's existing Nesterov SGD trajectory in canonical Lookahead with one
fixed, literature-common configuration:

```text
LOOKAHEAD_K = 5
LOOKAHEAD_ALPHA = 0.5
LOOKAHEAD_MOMENTUM_POLICY = retain
LOOKAHEAD_STATE = parameters_only
LOOKAHEAD_START_STEP = 1
```

Maintain one non-gradient slow copy of every optimizer-owned parameter. After
each group of five completed optimizer updates, interpolate the slow parameters
halfway toward the current fast parameters and copy the result back into the
same optimizer-owned `Parameter` objects. Keep Nesterov momentum buffers
unchanged. Do not interpolate BatchNorm buffers, counters, optimizer state,
gradients, SAM snapshots, or the existing EMA shadows.

The ordering on every training step is exact:

1. run the parent primary forward/backward;
2. on scheduled late SAM steps, perturb, replay the stochastic forward with BN
   tracking disabled, backpropagate, and restore the fast parameters;
3. call the sole `optimizer.step()` using the primary or SAM gradient;
4. if the one-based step is divisible by five, apply the Lookahead slow update
   and fast-parameter copyback while retaining momentum;
5. call the inherited cadence-31 charged-time full-state EMA updater, which
   therefore samples the post-Lookahead state on coincident steps;
6. synchronize and include all optimizer, Lookahead, and EMA update work in the
   existing charged step duration.

Evaluation remains exactly EXP-011: once per completed epoch, evaluate the live
model before EMA activation and only the existing full-state EMA afterward.
Never evaluate the slow copy as a second source. Keep physical GPU 0, seed 42,
the 300-second charged budget, architecture, data stream, CutMix, drop path,
SAM, learning-rate schedule, and evaluator unchanged.

This is a rigorous but **weak-to-moderate evidence candidate**. It is cheap and
canonical, but EXP-011 already combines Nesterov, late SAM, and a successful
evaluation EMA. Lookahead adds another low-pass mechanism to that stack, so a
large stable gain is less plausible than in the paper's less-smoothed optimizer
baselines. There is no local evidence that online optimizer variance, rather
than representation or decision boundaries, is the remaining limiter.

## Fixed Choice and Rejected Variants

Use `k=5` and `alpha=0.5` because the Lookahead paper reports this common
configuration across its optimizer comparisons and CIFAR experiments. These
values are fixed before any local timing, loss, or accuracy observation. The
experiment is not a sweep, and neither coefficient may change after preflight
or metric evidence.

Reject the following variants for EXP-015:

- **Late-only Lookahead:** starting at progress 0.75 would bundle Lookahead with
  an unvalidated activation schedule and provide only about 988 parent-dose
  tail synchronizations. The canonical all-training wrapper is the cleaner
  first test.
- **Momentum reset or pullback:** modifying the momentum buffer at each sync
  introduces another policy not selected by the distilled paper evidence. It
  also changes the intervention from a standard parameter wrapper to a custom
  optimizer.
- **BatchNorm-buffer Lookahead:** running moments and counters are forward-pass
  state, not optimizer-owned variables. Interpolating them every five steps
  would create a second, arbitrary full-state averaging kernel on top of the
  existing EMA.
- **Slow-only or slow/live evaluation:** this creates an extra checkpoint source
  or changes the parent evaluation protocol. The slow state is an online
  optimizer mechanism, not a separately selected inference model.
- **Replacing the parent EMA:** EXP-011's EMA is a validated +0.21-point package
  with negligible overhead. Removing it would test substitution rather than
  the proposed additive online variance reduction.
- **Adaptive `k`, `alpha`, start, or momentum behavior:** no metric, loss,
  displacement, latency, or EMA statistic may select a fallback. A valid
  preflight rejection ends the experiment before accuracy.

## Exact Parameter and Momentum Semantics

After the online model, optimizer, and SAM parameter inventory are constructed,
allocate one `torch.empty_like(..., memory_format=torch.preserve_format)` slow
tensor for each named trainable parameter. Copy the live parameter values into
the slow tensors once before the charged training timer starts. Allocation and
copying consume no RNG. Do not instantiate a second model, because that risks
new initialization and registered-state semantics.

The slow inventory must have exactly the same ordered names, shapes, dtypes,
devices, strides or preserved memory formats, and element count as the optimizer
parameter inventory. Slow tensors have `requires_grad=False`, never receive a
gradient, never appear in any optimizer parameter group, and must not alias the
online parameters, SAM snapshots, EMA shadows, or restore shadows.

On one-based update `t`, after `optimizer.step()`:

```python
if t % LOOKAHEAD_K == 0:
    torch._foreach_lerp_(slow_parameters, fast_parameters, LOOKAHEAD_ALPHA)
    torch._foreach_copy_(fast_parameters, slow_parameters)
```

The first operation implements
`slow_j = slow_(j-1) + 0.5 * (fast_j - slow_(j-1))`; the second makes the
optimizer-owned fast parameters bitwise equal to the new slow parameters.
There is no bias correction, warm start, decoupled weight decay, or alternate
interpolation direction.

Retain all inner SGD state exactly. In particular, the momentum buffers created
by `torch.optim.SGD(momentum=0.9, nesterov=True)` retain their values and object
identities across a Lookahead sync. They are neither scaled by `alpha`, copied
into a slow state, reset, nor replaced. The next Nesterov update therefore
starts from the pulled-back parameters with the accumulated fast-path momentum.
This parameter/momentum mismatch is deliberate canonical wrapper behavior and a
central attribution risk, not an implementation defect to repair after seeing
the trace.

Weight decay remains the inherited coupled SGD weight decay on every fast
update. Lookahead subsequently contracts the five-step endpoint toward the slow
path, so the effective optimization and regularization trajectory changes as a
package; a gain cannot be attributed solely to variance reduction.

## SAM and Update Ordering

On a scheduled SAM step, preserve EXP-011's exact sequence. The primary
gradient is used only to build the rho-0.05 perturbation. The second stochastic
pass replays the CUDA RNG and disables BN tracking. Its gradient remains after
the temporary perturbation is restored, then the sole Nesterov
`optimizer.step()` updates the unperturbed fast parameters. Lookahead runs only
after this update. Therefore:

- slow parameters never contain or interpolate toward the temporary SAM
  perturbation;
- there is still one optimizer and one BN-statistics update per batch;
- on a coincident step, Lookahead interpolates the post-SAM-optimizer endpoint;
- EMA subsequently samples that pulled endpoint, never the perturbation.

In the clean tail, `SAM_PERIOD=2` and `LOOKAHEAD_K=5` have least common multiple
10. Lookahead syncs therefore alternate between ordinary odd steps and SAM
even steps. At the EXP-011 realized dose, steps 20,858 through 25,798 contain
988 Lookahead syncs, exactly 494 after ordinary updates and 494 after SAM
updates. This parity is a useful audit, not a guarantee about the candidate's
terminal step count.

## Scope of Slow State and BatchNorm Policy

Lookahead owns **parameters only**. It must not read or write any registered
buffer. BatchNorm `running_mean`, `running_var`, and `num_batches_tracked` keep
their parent online behavior: the primary pass updates them once, the SAM
second pass does not track them, and a Lookahead interpolation leaves them
bitwise unchanged.

This policy follows optimizer-wrapper semantics but has a real limitation.
Immediately after a pullback, the live parameters are slow-path parameters while
the BN statistics summarize primary forwards along the intervening fast path.
No BN recalibration is allowed: an extra training-data pass would add exposure,
consume RNG, and violate the controlled fixed-budget comparison. The result is
attributable to parameter-only Lookahead plus inherited BN tracking, not to an
ideal recalibrated slow model.

The mismatch is partly bounded operationally by the existing evaluation
protocol. A complete CIFAR epoch has 195 optimizer steps, divisible by five, so
every complete pre-EMA epoch ends immediately after a Lookahead sync and
evaluates `fast == slow` parameters with current online BN buffers. In the EMA
tail, the inherited full-state EMA averages both the resulting parameter path
and the current floating BN buffers, while copying integer counters. Do not
change that EMA buffer policy to compensate for Lookahead.

## Existing EMA Interaction and Double-Smoothing Risk

Lookahead makes the slow sequence itself an exponential recurrence over
five-step fast endpoints:

```text
slow_j = 0.5 * slow_(j-1) + 0.5 * fast_endpoint_j
```

After each recurrence, it resets the fast parameters to the slow state while
retaining momentum. EXP-011 then applies another exponential recurrence to
cadence-31 online states during the final quarter, with an 18.75-second
half-life and a copy-initialized first state. The two mechanisms are not
equivalent: Lookahead changes future gradients and iterates, while the EMA is an
evaluation-only state estimator. They nevertheless both suppress high-frequency
trajectory movement.

The cadences are coprime (`gcd(5,31)=1`). Under EXP-011's 160 EMA samples, 32
EMA samples occur on steps divisible by both cadences (`lcm=155`), split evenly
between ordinary and SAM-derived updates. The other 128 samples observe a fast
state one to four inner updates after a pullback. Thus the EMA does not merely
average slow checkpoints, but it does average a trajectory repeatedly anchored
to the slow sequence.

The positive hypothesis is that Lookahead reduces harmful Nesterov/SAM endpoint
variance while retained momentum preserves useful travel, allowing the
full-state EMA to center a better basin. The stronger counter-hypothesis is
double smoothing: five-step pullbacks attenuate useful SAM-driven exploration,
then the 18.75-second EMA smooths the already contracted path again. Because
EXP-011's final EMA/live parameter distance was only 1.51%, the parent does not
show gross unstable wandering. This substantially lowers the expected upside.

## Evaluation Source and Metric Integrity

Retain `EVAL_EVERY=1` and the existing `ChargedTimeEMA.evaluate` routing without
adding a Lookahead evaluator:

- before the first charged EMA sample, evaluate the current live model once;
- after the first EMA sample, swap in and evaluate only the existing full-state
  EMA once, then restore the live state exactly;
- never evaluate the slow tensor collection directly;
- never evaluate both live and EMA in an epoch;
- never use loss, accuracy, displacement, or checkpoint results to choose the
  source or update policy.

The slow state is persistent optimizer mechanism state, but it is not registered
in `model.state_dict()` and is not part of the evaluator swap. The EMA shadow
must cover exactly the model's parameters and persistent buffers as before. The
EMA restore returns to the current online fast state; it must not restore from
or overwrite the Lookahead slow state.

`best_test_acc`, `final_test_acc`, and `final_test_loss` therefore retain the
EXP-011 definition. The once-per-epoch ceiling is unchanged, and there is no
hidden slow/live/EMA model selection channel.

## State, RNG, and Correctness Audits

### Deterministic CPU smokes

1. On scalar and multi-tensor fixtures, verify exact slow/fast values through
   steps 1-11, including no update at steps 1-4 and the fixed recurrence at
   steps 5 and 10.
2. Initialize distinct parameters, momentum buffers, floating BN buffers, and
   integer counters; verify a sync changes only parameters and slow tensors.
3. Assert optimizer parameter identities and momentum-buffer values and
   identities are unchanged by the sync itself.
4. Assert CPU RNG, global device RNG when available, and dedicated CutMix
   generator states are unchanged around construction and every sync.
5. Reconcile every slow tensor against the named optimizer inventory and prove
   non-aliasing with optimizer parameters, SAM snapshots, and every EMA shadow.
6. Force successful and exceptional EMA evaluations after a Lookahead sync and
   prove exact restoration of live parameters/buffers, unchanged slow tensors,
   unchanged optimizer ownership, and unchanged state-management RNG.

### Physical-GPU-0 integration smoke

Run a consecutive production-faithful ten-step cycle in BF16/channels-last mode
with one arm using parent EXP-011 and one using Lookahead. Replay identical
batches and stochastic states. Require exact parent/candidate equality through
step 4; at step 5, independently compute the FP32 reference interpolation and
require the candidate fast and slow parameters to match within a fixed
dtype-aware tolerance. Exercise step 10 through the full SAM perturb/replay,
restore, optimizer update, Lookahead sync, and EMA ordering.

At every sync, assert:

- SAM parameters were restored before interpolation;
- fast parameters equal slow parameters after copyback;
- BN buffers and counters are bitwise unchanged by Lookahead itself;
- optimizer parameter and momentum-buffer identities are unchanged;
- no CPU, CUDA, CutMix-CPU, or CutMix-CUDA RNG state changed;
- on a cadence-31/cadence-5 coincidence, the EMA shadow equals a reference
  update from the post-pullback model, not the pre-pullback endpoint.

Do not require parent/candidate weights or BN values to remain equal after the
first sync; Lookahead is intended to change subsequent activations, gradients,
and BN statistics. Continue requiring identical data order, CutMix decisions,
drop-path draws, SAM schedule, and RNG stream position for the same finite step
prefix.

### Production audit fields

Report `k=5`, `alpha=0.5`, momentum policy, state scope, slow tensor/element
counts, total sync count, first/last sync step, ordinary/SAM tail sync counts,
EMA-coincident sync count, and zero inventory/alias/RNG/momentum-identity/buffer-
mutation/order failures. Measure pre-pullback parameter L2 and relative L2 only
at fixed preregistered diagnostic points (first sync and first clean-tail sync)
to avoid an extra full-model norm on every fifth step. These values are
report-only and cannot alter the run.

All sync and diagnostic tensor work occurs before the existing CUDA
synchronization and is charged. Defer scalar reads and finiteness checks until
the next excluded evaluation interval or final audit where possible; do not add
per-sync `.item()` synchronizations that would distort optimizer exposure.

## Accuracy-Blind Latency and Dose Gate

After correctness passes, run one complete paired preflight on physical GPU 0
with `CUDA_VISIBLE_DEVICES=0`. Use the exact EXP-011 parent materialized from
commit `d68f73a`, identical real CIFAR batches and replayed stochastic state,
BF16, channels-last, optimizer/SAM/EMA ownership, and alternating candidate /
parent order across five rounds. Never call the frozen evaluator or inspect
test accuracy in preflight.

The timing mixture must include the full production paths, not just isolated
foreach copies. Weight these six classes by EXP-011's realized schedule:

```text
early non-sync:       16,686 / 25,798
early Lookahead sync:  4,171 / 25,798
late ordinary non-sync: 1,976 / 25,798
late ordinary sync:      494 / 25,798
late SAM non-sync:      1,977 / 25,798
late SAM sync:            494 / 25,798
```

Include cadence-31 EMA updates and cadence-155 EMA/Lookahead coincidences in
each arm as appropriate. Measure end-to-end charged step latency with the same
synchronization boundary as production, plus candidate-only peak allocation.

The first complete numeric preflight is decisive. Require parent round drift
`<=0.03`, median absolute deviation divided by median paired ratio `<=0.005`,
candidate/parent weighted median latency ratio `<=1.02`, projected optimizer
steps `>=25,000`, projected total runtime `<600s`, finite state, and every
correctness audit above. A projected dose below 25,300 may still proceed to the
formal accuracy test but preregisters that full-dose mechanism support is
unlikely. Do not rerun a completed timing result, change `k` or `alpha`, reset
momentum, disable audits, or sparsify Lookahead after a gate failure.

One slow FP32 parameter copy is about 10.5 MiB. The sync every five steps reads
fast and slow state, writes slow, then copies slow to fast. H20 memory bandwidth
should make this small relative to forward/backward work, but two full-parameter
foreach passes every fifth step are materially more frequent than EXP-011's
cadence-31 EMA. The latency gate is therefore necessary despite no extra model
forward.

## One Metric Launch and Thresholds

After all accuracy-blind checks pass, reconfirm that physical GPU 0 is the
approximately 97,871 MiB NVIDIA H20, remove any stale `run.log`, and launch
exactly once:

```bash
timeout 600s env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 uv run train.py > run.log 2>&1
```

There is no seed repeat, metric retry, coefficient fallback, momentum-policy
fallback, or early stopping from finite intermediate accuracy. Preserve the raw
log through durable transcription and adversarial result review.

Formal validity requires exit 0, charged training approximately 300 seconds,
total runtime below 600 seconds, exactly one evaluation per epoch, a complete
summary, fixed seed 42, physical GPU 0, unchanged 2,748,890-parameter model,
only `train.py` modified, and zero Lookahead/SAM/EMA/state/RNG audit failures.

The tree improvement threshold is:

```text
best_test_acc >= 95.71%
```

Mechanism-supporting evidence is deliberately stronger and additionally
requires:

```text
num_steps >= 25,300
EMA updates >= 155
final-16 EMA accuracy mean >= 95.69%
Lookahead syncs == floor(num_steps / 5)
tail ordinary/SAM Lookahead-sync imbalance <= 1
all inherited CutMix, SAM, EMA parity, restore, and source-count audits pass
```

Report the final-16 EMA range, mean, final value, and
`best_test_acc - final16_mean` selection premium. A best from 95.71 upward with
a tail mean below 95.69 is a formal tree improvement but weak evidence that
Lookahead lifted the stable plateau. A result below 95.71 is one
no-improvement; do not try another Lookahead setting.

## Evidence, Expected Effect, and Falsification

The Lookahead paper supports the optimizer wrapper, one extra parameter copy,
and common `k=5`, `alpha=0.5` configuration, including CIFAR evidence
(`papers/lookahead-optimizer.md`). It does not establish an effect size for
this exact WRN, fixed-time cosine schedule, front-loaded CutMix, late period-two
SAM, retained-momentum implementation, or downstream full-state EMA.

The modern averaging study supports combining annealing and weight averaging,
but it does not show that nested online Lookahead plus evaluation EMA is better
than a well-calibrated EMA alone (`knowledge/papers/when-where-why-average.md`).
The EMA scaling work reinforces that cadence and horizon matter; it offers no
reason to retune EXP-011's 18.75-second half-life conditionally
(`knowledge/papers/how-to-scale-your-ema.md`). Classical SWA evidence supports
late trajectory centralization but also emphasizes trajectory diversity and BN
semantics, both of which are potential weaknesses here
(`knowledge/papers/stochastic-weight-averaging.md`).

EXP-011 is already strong: `95.61%` best, `95.493125%` final-16 mean, only
`1.506%` terminal EMA/live relative parameter distance, and negligible EMA
overhead. Those observations do not diagnose high online variance. The goal's
history also shows that sub-0.30-point differences can reverse under this
single-run protocol. Lookahead therefore has a plausible systems fit but a
poorly anchored accuracy ceiling.

A realistic preregistered expectation is **0.00 to +0.15 percentage points**,
with meaningful downside possible if pullbacks attenuate SAM exploration or
the unrecalibrated BN state becomes less compatible with the parameters. That
range can cross the formal +0.10 threshold but is below the approximately
0.20-point stable-tail lift needed to reach the `95.69` scientific mean. On
evidence and expected impact, this proposal should rank below a similarly cheap
candidate that directly changes decision-boundary learning with a coherent,
externally anchored rule.

The experiment is falsified as an accuracy intervention by
`best_test_acc <95.71%`. It is scientifically unsupported despite a formal
maximum if the final-16 mean is below 95.69, the selection premium grows, or
the full-dose gates fail. It is invalid if sync count/order, parameter-only
scope, momentum retention, RNG neutrality, SAM restoration, EMA ordering,
evaluation routing, or bounded execution fails. None of these outcomes permits
a new `k`, `alpha`, start time, buffer policy, or momentum policy within
EXP-015.

## Effort and Risk

**Estimated effort: medium.** The core recurrence is small, but trustworthy
integration requires exact ownership, momentum, SAM, BN, EMA-ordering, RNG, and
latency audits.

**Risk: high for scientific impact, low-to-medium for implementation.** The
method should fit the compute and memory budget, but the strongest available
evidence predicts a redundant smoothing interaction more readily than a stable
0.20-point plateau gain.
