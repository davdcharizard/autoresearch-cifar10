# Proposal: Readiness-Gated Clean-Tail Full-State EMA Without SAM

## Summary

Add one fixed, activation-anchored, bias-corrected full-state EMA directly to
EXP-002. Preserve the successful WRN-16-4, front-loaded probability-0.5 CutMix,
drop-path, Nesterov SGD, cosine schedule, seed 42, data stream, BF16/channels-last
path, and 300-second charged budget. Do not add SAM, a second forward, BN
recalibration, live/EMA blending, checkpoint selection, or a coefficient sweep.

This branch asks a clean causal question that EXP-011 could not answer: can late
trajectory averaging alone improve EXP-002, without the validated EXP-004 SAM
tail? EXP-002 is `95.23%`, so local improvement requires `95.33%`. The goal-wide
best remains EXP-011 at `95.61%`; a local pass below that level is useful
isolation evidence, not a new global result.

Every GPU command must expose only physical GPU 0 with
`CUDA_VISIBLE_DEVICES=0` and first verify the approximately 97,871 MiB NVIDIA
H20. Only tracked `train.py` may change.

## One Fixed Estimator

Use exactly:

```python
EMA_START = 0.75
EMA_UPDATE_EVERY = 31
EMA_TAIL_HALF_LIVES = 4.0
EMA_HALF_LIFE_S = (1.0 - EMA_START) * TIME_BUDGET_S / EMA_TAIL_HALF_LIVES
# 18.75 seconds when TIME_BUDGET_S == 300
EMA_READY_MASS = 0.75
EMA_READY_ESS = 90.0
```

Let `t0 = EMA_START * TIME_BUDGET_S = 225.0s`. Let `x_i` be the complete
floating model state after optimizer step `s_i`, sampled using that step's
entry charged timestamp `t_i`. A step is due only when entry progress is at
least `0.75` and the one-based post-update step number is divisible by 31.
Define:

```text
d_i = 2 ** (-(t_i - t_(i-1)) / 18.75), with t_(0) = t0
m_0 = 0
q_0 = 0
m_i = d_i m_(i-1) + (1 - d_i)
q_i = d_i^2 q_(i-1) + (1 - d_i)^2
a_i = (1 - d_i) / m_i
A_1 = x_1
A_i = (1 - a_i) A_(i-1) + a_i x_i, i > 1
ESS_i = m_i^2 / q_i
```

The exact normalized terminal weights are:

```text
w_(i,n) = (1 - d_i) product_(j=i+1..n)(d_j) / m_n
sum_i w_(i,n) = 1
m_n = 1 - 2 ** (-(t_n - t0) / 18.75)
ESS_n = 1 / sum_i w_(i,n)^2
```

The interval from `t0` to the first sample does not change `A_1`, which is
exactly `x_1`; it does determine the first sample's later normalized weight and
the reconstructable mass identity. This wording avoids treating the initial
decay as an immediate model interpolation.

This zero-mass recurrence deliberately removes EXP-011's copy-in artifact.
EXP-011 initialized its first state with unit mass, leaving about 6.30% terminal
weight on that first sample. Under comparable timestamps the corrected kernel
has terminal mass about 0.937, effective sample size around 100 or more, and
mean state age around 22 seconds. EXP-002 is faster than EXP-004 because it has
no SAM, so it should provide roughly 220 cadence samples rather than EXP-011's
160; the run must reconstruct the actual kernel from its own timestamps rather
than asserting sibling-run approximations.

All parameters and persistent floating buffers use the same `a_i`. Persistent
integer buffers copy the latest sampled value. Gradients, optimizer momentum,
RNG state, data-loader state, CutMix generators, training counters, and
nonpersistent objects are excluded. Shadows are allocated before charged
training, require no gradient, preserve dtype/device/layout, do not alias live
state, and never enter the optimizer.

## Readiness-Gated Evaluation

EXP-016 exposed a serious attribution problem: evaluating a partially filled
average lets `best_test_acc` be won by an estimator other than the one being
claimed. Direct EMA has no hard boxcar fill point, so define readiness by the
implemented kernel itself. The EMA may be evaluated only after both:

```text
m_i >= 0.75
ESS_i >= 90
```

With EXP-002's realized 27,950-step throughput, mass reaches 0.75 after about
37.5 clean-tail seconds, at roughly 113 samples and ESS about 98. This should
leave approximately 18 once-per-epoch ready-EMA evaluations. Before readiness,
evaluate the live model. From readiness onward, evaluate only EMA. Never
evaluate both sources in one epoch. Record the source, sample count, mass, ESS,
and charged timestamp for every evaluation and at the best epoch.

Evaluation uses one exception-safe full-state swap into the existing model,
one call to the frozen evaluator, and an exact live-state restore in `finally`.
It swaps parameters and every persistent buffer, including latest-copy integer
BN counters. It does not replay training data, recalibrate BN, ensemble logits,
or choose the better source. The model's module modes, RNG states, live tensors,
optimizer parameter identities, and optimizer momentum identities must be
identical after both successful and injected-failure restoration tests.

`best_acc` continues to span the single preregistered source used at each epoch;
this preserves ordinary online checkpoint semantics. A formal improvement won
before readiness remains a valid tree metric but does not support the EMA
mechanism. The report must separately give the best ready-EMA accuracy and all
ready-EMA tail statistics.

## Mechanism and Counter-Hypothesis

EXP-002 finished at `95.19%` after peaking at `95.23%`, despite 27,950 optimizer
steps and a clean final quarter. Sparse EMA can use abundant H20 memory to
reduce late-iterate variance without another network pass or meaningful loss of
training exposure. Time-derived decay preserves the estimator horizon if
charged throughput changes, and cadence 31 is already validated operationally
by EXP-011. Because there is no SAM in this branch, every sample is an ordinary
post-Nesterov state; no SAM parity claim is made.

The counter-hypothesis is strong. EXP-002's best-to-final gap is only 0.04
points, the cosine tail may already converge smoothly, and EMA can lag a still
improving model. Full-state linear averaging of BN statistics is only an
approximation to the statistics induced by averaged weights. Moreover,
`best_test_acc` is a maximum: reducing checkpoint variance can lower the
max-selection premium unless EMA also raises the late mean. EXP-011's 95.61
best sat on a 95.493 final-16 mean, so any result here must carry tail mean,
range, final value, and best-minus-mean premium.

The literature supports weight averaging under annealed learning rates and
scaling decay by update frequency/time, but it does not establish that EMA
alone supplies a 0.10-point gain on this fixed recipe. This is a controlled
mechanism-isolation experiment, not a high-confidence global-best attempt.

## Implementation Contract and Audits

Preserve EXP-002's online ordering exactly:

1. determine charged progress and LR/drop-path settings;
2. move the same batch and apply the same dedicated-generator CutMix decision;
3. perform one autocast forward, one backward, and one Nesterov update;
4. if the post-update one-based step is due, update EMA from the restored live
   state before the existing CUDA synchronization;
5. synchronize and charge all training-side EMA work in `dt`.

There is exactly one model forward/backward and one optimizer step per training
batch. No SAM constants, perturbation, snapshot inventory, second-pass BN
suppression, RNG replay, or SAM audit counters may appear. The parent and
candidate online state must remain bit-identical under replayed batches and
stochastic state; only wall-clock progress may cause a bounded realized-dose
difference.

The final audit must make the estimator independently reconstructable:

- fixed start, cadence, half-life, readiness mass/ESS, and state inventory;
- update count, first/last step/progress/time, boundary-to-first interval, and
  later interval min/mean/max;
- decay and normalized-alpha min/mean/max, terminal mass, `q`, residual mass,
  and agreement with the elapsed-time mass identity;
- exact reconstructed weight sum, oldest/newest/min/max weights, ESS, and
  weighted mean state age;
- the corresponding copy-in kernel on the same timestamps, including its
  first-state weight, ESS, and mean age, as report-only isolation evidence;
- readiness step/time/sample count and every evaluation's live/EMA source,
  mass and ESS; source and kernel state at best accuracy;
- finite nonzero consecutive-sample parameter distance and terminal EMA/live
  absolute and relative parameter distance;
- EMA/live BN mean and variance distances, per-BN EMA/live variance ratios,
  latest integer equality, and positive finite EMA variances;
- exact evaluator-call/evaluation/epoch equality, balanced swaps/restores,
  optimizer identity, module-mode restoration, and zero coverage, alias, RNG,
  nonfinite, and restoration failures;
- unchanged CutMix applied/eligible counts, one-pass/one-update counts, epochs,
  steps, parameters, charged/total time, and peak allocation.

Do not call `.item()` on model tensors or synchronize solely for audits during
charged training. Retain small no-grad GPU diagnostics for reading after the
charged budget, or derive scalar kernel diagnostics from the already available
Python charged timestamps.

## Correctness Smokes

Before timing, require deterministic, accuracy-blind smokes for:

- irregular intervals, a very short first interval, one sample, and many
  samples, comparing the online recurrence, mass, `q`, ESS, and every weight
  with an FP64 direct weighted sum;
- proof that the activation-to-first interval leaves `A_1=x_1` while changing
  its later reconstructed weight;
- complete parameter/floating-buffer/latest-integer state coverage, exact key
  equality with a fresh `state_dict`, non-aliasing, and optimizer exclusion;
- readiness false before either threshold, true only after both, live routing
  before readiness, and EMA routing afterward;
- exact successful and injected-exception swap restoration, module modes,
  optimizer/momentum identities, and CPU/global-CUDA/CutMix RNG neutrality;
- a production WRN BF16/channels-last trace showing post-Nesterov sampling,
  exact candidate/parent online equality, unchanged CutMix draws, and exactly
  one forward/backward/update per batch.

## Decisive Accuracy-Blind GPU-0 Preflight

After correctness passes, run one complete paired preflight on physical GPU 0.
Materialize exact parent EXP-002 in an experiment-owned `/tmp` module and load
the candidate separately. Use real CIFAR training batches, BF16,
channels-last, the production optimizer and dedicated CutMix generators. Guard
both evaluators before traces begin so any attempted test-loader access raises;
assert zero test batches and zero accuracy values.

Run seven alternating-order paired rounds. Each arm executes a 248-step
production-weighted trace: 186 early steps and 62 clean-tail steps, with two
candidate cadence updates in the tail and no SAM path. Replay identical batch,
CutMix, crop/flip/drop-path, CPU, and CUDA stochastic state. Separately exercise
enough state-only timestamps to cross readiness, then test one live-source and
one EMA-source guarded evaluation on success and injected failure. State-only
arithmetic is correctness work, not included in the charged-latency ratio.

Print every parent/candidate time and the explicit projections. The first
complete numeric result is decisive. An exception, assertion, timeout, or
malformed output before numeric gates permits one recorded harness repair; a
numeric failure ends this leaf without a metric launch. Do not retune the
estimator or gate after seeing numbers.

Proceed only if:

```text
physical index 0 = NVIDIA H20, approximately 97,871 MiB
exactly one CUDA device visible under CUDA_VISIBLE_DEVICES=0
parent round drift <= 0.03
parent MAD / parent median <= 0.01
paired-ratio MAD / paired-ratio median <= 0.01
median candidate / parent charged latency <= 1.005
maximum paired round ratio <= 1.02
projected steps = 27,950 / median_ratio >= 27,700
projected EMA updates >= 220
projected ready-EMA evaluations >= 14
projected total runtime < 600 seconds
all arithmetic, online-parity, inventory, BN, RNG, readiness, source,
and success/failure restoration checks pass
```

The 1% seven-round dispersion ceiling incorporates EXP-016's observed
0.005307 ratio MAD rather than repeating its under-calibrated 0.005 five-round
abort. Peak allocation is reported against EXP-002's 1,178.9 MiB and checked
for OOM, but it is not a hard rejection threshold because VRAM is explicitly a
soft goal constraint and the H20 has ample headroom. The preflight must include
model, optimizer momentum, EMA shadows, evaluation backup, and audit storage;
candidate-only peak is interpreted cautiously because EXP-015 showed that a
smoke can understate full-run peak.

## Sole Metric Run and Decision Rules

After preflight passes, reconfirm GPU 0, remove any stale log, and launch once:

```bash
timeout 600s env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
  uv run train.py > run.log 2>&1
```

There is no seed repeat, metric retry, early stopping from intermediate test
output, half-life or readiness fallback, live/EMA blend, or addition of SAM.
Retain the raw log until its exact summary and audits have been durably
transcribed and Claude has completed adversarial result review.

Validity requires exit 0, physical GPU 0, fixed seed 42, only `train.py`
modified, approximately 300 charged seconds, total runtime below 600 seconds,
one evaluator call and one evaluation per epoch, the unchanged 2,748,890
parameters, at least 27,700 optimizer steps, at least 220 EMA samples, readiness
reached by 270 charged seconds, at least 14 ready-EMA evaluations, last sample
at or above 99.5% progress, a complete summary, and zero integrity failures.
A trustworthy realized-dose or readiness shortfall is `no-improvement`, not
permission to rerun.

Read the parent metric with `tree.sh show ... 002` immediately before the
decision and require `95.23`. Apply:

- `best_test_acc >= 95.33%`: formal local `improvement` over EXP-002;
- valid result below `95.33%`: `no-improvement` for this exact EMA package;
- if the formal best occurred before readiness on the live model, retain the
  tree verdict but classify the EMA mechanism hypothesis as unsupported;
- if ready-EMA best is at least `95.33%`, EMA is consistent with the local-gain
  hypothesis; report all ready-EMA values, mean/range/final and premium rather
  than treating one maximum as stable evidence;
- `>=95.61%` reaches the current global-best level; only `>=95.71%` clears it by
  the goal's required 0.10-point resolution;
- `95.33-95.60%` is a useful local isolation result but not evidence that EMA
  alone matches the full EXP-004+EMA lineage globally.

For global interpretation, compare the realized ready-EMA plateau with
EXP-002's 95.23 best/95.19 final, EXP-004's 95.40 best/final, and EXP-011's
95.61 best, 95.493125 final-16 mean, and 95.46 final. Throughput differences,
max-selection premium, and the absence of SAM must remain explicit. Do not
attribute any sub-0.30-point difference more strongly than the known fixed-run
selection noise supports.

## Falsification and Risks

The proposal is falsified locally by a valid result below 95.33, or
mechanistically by a formal pass whose best and all threshold-clearing values
come from the pre-readiness live model. A ready-EMA maximum above 95.33 with a
depressed ready-EMA mean would support checkpoint selection more than stable
generalization. Failure to reach 95.61 is expected and does not negate useful
isolation evidence, but it confirms that direct EMA does not replace the
SAM+EMA package at the goal frontier.

Principal risks are EMA lag under a still-improving cosine tail, approximate BN
buffer compatibility, loss of max-selection premium, and an effect size below
the observed 0.14-0.29-point run-selection noise. The readiness gate reduces
estimator misattribution but also hides early EMA checkpoints. Full-state
copies can fail catastrophically if restoration is incomplete, so exception
restoration and independent state enumeration are hard validity conditions.

## Verification Checklist

1. Prove only `train.py` differs from EXP-002 and no SAM path was introduced.
2. Match normalized EMA, mass, second moment, ESS, and exact weights to FP64.
3. Verify full-state coverage, latest integers, no aliases, and no optimizer
   ownership.
4. Verify kernel-based readiness and source-at-best accounting.
5. Verify parent online-state and RNG parity plus exact success/failure restore.
6. Pass the single seven-round accuracy-blind physical-GPU-0 preflight.
7. Run exactly one fixed-seed metric launch and apply local, mechanism, stable
   tail, and global-context readings without retuning.
