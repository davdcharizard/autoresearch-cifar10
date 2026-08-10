# Proposal: Activation-Anchored Bias-Corrected Full-State EMA

## Summary

Add the successful EXP-011 clean-tail EMA mechanism directly to base EXP-004,
but remove EXP-011's unintended copy-in point mass. Keep the same charged-time
support, 18.75-second half-life, cadence 31, full-state convention, and
single-source evaluation policy. Initialize exponential mass at zero at the
exact 225-second activation boundary and normalize every floating-state update
by the accumulated mass.

This is one fixed estimator intervention, not a half-life sweep, an EMA/live
blend, or an evaluation change. Preserve EXP-004's WRN, data stream, seed 42,
front-loaded CutMix, drop path, period-two clean-tail SAM, optimizer, schedules,
BF16/channels-last execution, and 300-second charged budget. Every GPU command
must expose only physical GPU 0 with `CUDA_VISIBLE_DEVICES=0` and verify that it
is the approximately 97,871 MiB NVIDIA H20.

## Exact Estimator

Use the fixed configuration:

```python
EMA_START = 0.75
EMA_UPDATE_EVERY = 31
EMA_TAIL_HALF_LIVES = 4.0
EMA_ACTIVATION_TIME_S = EMA_START * TIME_BUDGET_S  # 225.0
EMA_HALF_LIFE_S = (
    (1.0 - EMA_START) * TIME_BUDGET_S / EMA_TAIL_HALF_LIVES
)  # 18.75
```

Let `t_0 = EMA_ACTIVATION_TIME_S`, let `x_i` be the complete floating model
state sampled at charged step-entry time `t_i`, and define

```text
m_0 = 0
d_i = 2 ** (-(t_i - t_(i-1)) / EMA_HALF_LIFE_S)
m_i = d_i * m_(i-1) + (1 - d_i)
a_i = (1 - d_i) / m_i
A_i = (1 - a_i) * A_(i-1) + a_i * x_i
```

For `i=1`, set `A_1 = x_1` exactly; algebraically `a_1=1`, so this avoids
reading uninitialized shadow storage. For every later sample, apply the same
`a_i` with fused lerps to all parameters and persistent floating buffers.
Persistent integer buffers copy the latest sample and are never averaged.

The recurrence is the normalized form of a zero-initialized exponential
numerator. Its exact terminal mass and checkpoint weights are

```text
m_n = 1 - product(d_1 ... d_n)
    = 1 - 2 ** (-(t_n - t_0) / EMA_HALF_LIFE_S)

w_(i,n) = (1 - d_i) * product(d_(i+1) ... d_n) / m_n
sum_i w_(i,n) = 1
```

This activation anchor matters. Under EXP-011's realized rounded timings
(`t_1=225.1324`, `t_n=299.9060`, 160 samples), the corrected terminal mass is
`1 - 2**(-74.9060/18.75) = 0.937282`, leaving `0.062718` unnormalized missing
mass. EXP-011's copy-in implementation instead gave its first sample
`2**(-74.7736/18.75) = 0.063025`, or 6.30% of the final normalized model.
Using its observed near-regular spacing only as a descriptive approximation,
the corrected kernel has oldest weight about 0.0328%, newest weight about
1.8388%, effective sample size about 101.47, and mean state age about 21.80 s;
the copy-in kernel had about 6.3025%, 1.7235%, 79.18, and 25.13 s respectively.
The candidate run must reconstruct exact weights from its actual interval list
rather than asserting these rounded sibling-run values.

## Mechanistic Rationale

EXP-004 reached 95.40 with a clean final quarter and period-two SAM but no
trajectory averaging. Its child EXP-011 showed that cadence-31 full-state EMA
is essentially free in charged throughput and reached the current 95.61 global
best. However, EXP-011's first-sample copy initialization was not the normalized
charged-time exponential it intended: a higher-learning-rate state near the
start of the clean tail retained 6.30% terminal weight.

The corrected estimator preserves averaging support and variance reduction
while removing that stale point mass. Cadence 31 is odd relative to SAM period
2, so due samples alternate ordinary and SAM-derived post-update states and
their counts differ by at most one. The counter-hypothesis is equally clear:
the old 6.30% anchor may have been useful regularization, and following later
low-learning-rate iterates more closely may lower accuracy. Literature supports
averaging with annealing, but does not establish either the sign or a 0.10-point
effect for this correction.

This fork has a lower local bar than descendants of EXP-011, but interpretation
must retain global context. EXP-011's best was 95.61 while its final-16 EMA mean
was only 95.493125. Therefore a narrow local pass need not mean the corrected
kernel is globally better or that it raised the stable plateau.

## Full-State and Cadence Semantics

Construct the shadow inventory from `model.named_parameters()` and persistent
`model.named_buffers()`, and assert exact key equality with a fresh
`model.state_dict()`.

- All 44 trainable parameter tensors use the normalized recurrence.
- All 26 persistent floating buffers, including BatchNorm `running_mean` and
  `running_var`, use the identical normalized recurrence.
- All 13 integer `num_batches_tracked` buffers copy the newest sampled value.
- Shadows preserve shape, dtype, device, and memory format, require no gradient,
  do not alias live state, and are absent from optimizer and SAM inventories.
- Optimizer momentum, gradients, SAM snapshots, RNG state, modes, evaluator
  state, counters, and nonpersistent objects are excluded.

Sampling is eligible only when one-based `next_step % 31 == 0` and step-entry
charged progress is at least 0.75. It occurs after the sole optimizer step and
after exact SAM restoration, but before the existing CUDA synchronization and
charged-time increment, so all EMA work is charged. The first decay interval
runs from exactly 225.0 seconds to the first eligible sample time. Do not anchor
at time zero or reset the clock to the first sample.

Evaluation remains exactly once per epoch. Before the first valid sample,
evaluate the live model. After activation, evaluate only the corrected EMA via
an exception-safe swap of parameters and every persistent buffer, then restore
the live state exactly. Do not evaluate live and EMA in the same epoch. Do not
recompute BN statistics, replay training data, ensemble logits, select between
sources, or change the evaluator.

## Required Audits

The final summary must make the estimator and protocol independently
reconstructable:

- fixed start, cadence, half-life, activation time, and state inventory;
- update count, first/last step, progress and charged time, boundary-to-first
  interval, later interval min/mean/max, and ordinary/SAM sample counts;
- decay and normalized-alpha min/mean/max, terminal mass, residual missing
  mass, and agreement with the closed-form charged-time mass identity;
- exact reconstructed weight sum, oldest/newest/min/max weights, effective
  sample size `1/sum(w_i**2)`, and weighted mean state age;
- report-only reconstruction of the corresponding copy-in kernel on the same
  actual timestamps, isolating the implemented difference;
- finite, nonzero consecutive-sample parameter L2 and final corrected/live
  absolute and relative parameter L2;
- corrected/live BN mean and variance L2, BN variance-ratio min/mean/max,
  strictly positive corrected running variances, and newest integer equality;
- live/EMA evaluation counts, evaluator calls, swaps, restores, module modes,
  optimizer identity, and zero coverage, restore, nonfinite, alias, and RNG
  failures;
- unchanged CutMix/SAM exposure and first-activation semantics, epochs, steps,
  charged/total time, parameter count, train-loss diagnostic, and peak VRAM.

An FP64 arithmetic smoke must compare the online recurrence with a direct
weighted sum for irregular intervals, a very short first interval, one sample,
multiple samples, floating BN-like buffers, and latest-copy integer behavior.

## Accuracy-Blind GPU 0 Preflight

Run one complete, decisive parent/candidate preflight before the metric launch.
Materialize EXP-004 parent behavior and the candidate in the same harness on
physical GPU 0, BF16/channels-last. Monkeypatch the evaluator to raise before
any trace begins; preflight may use synthetic evaluation-shaped forwards only
and must never iterate the test loader or reveal accuracy.

Use five alternating-order rounds containing the parent's measured workload
mixture: early ordinary steps, CutMix steps, clean ordinary steps, period-two
SAM steps, cadence updates of both SAM parities, and synthetic full-state
swap/restore. The first complete numeric output is decisive. Retry only for an
exception, missing/malformed output, or assertion failure before any numeric
gate is emitted; never relabel a numeric failure as a harness error.

Proceed only if:

- parent drift is at most 3%, and parent median absolute deviation divided by
  parent median is at most 1%;
- median candidate/parent weighted charged latency is at most 1.02 and every
  round ratio is at most 1.05;
- projected optimizer steps are at least 25,200, projected EMA samples at least
  155, projected epochs at least 128, and projected total runtime below 600 s;
- candidate-only model, optimizer, SAM snapshots, corrected EMA, restore, and
  audit state peak allocation is below 1.30 GiB;
- at least 30 production-order candidate samples cover both cadence/SAM
  parities;
- direct kernel arithmetic, closed-form mass, parent online-state equality
  after equal work, state coverage, BN positivity, RNG preservation, optimizer
  identity, and exact success- and injected-failure-path restoration pass.

No cadence, start, half-life, coefficient, state policy, threshold, or other
training choice may change after any numeric preflight result.

## One Metric Run and Decision Rules

After the full preflight passes, launch exactly once:

```bash
timeout 600s env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 uv run train.py > run.log 2>&1
```

Base EXP-004 is 95.40, so the formal parent-relative threshold is
`best_test_acc >= 95.50%` (at least 9,550 of 10,000 test examples correct).
Retain these separate contextual readings:

- final-16 corrected-EMA mean `>=95.50%` is the preregistered strong stable-tail
  bar; report its range, final accuracy, and best-minus-tail premium;
- `best_test_acc >=95.61%` reaches the existing global-best level, while
  `>=95.71%` clears that global best by the goal's 0.10-point resolution;
- a result from 95.50 through 95.60 is a valid local improvement over EXP-004,
  but is neither a new global best nor evidence that bias correction improved
  on EXP-011.

Require exit 0, 299.5-301.0 charged seconds, total below 600 seconds, exactly
one evaluation line and one evaluator call per epoch, at least 25,200 optimizer
steps, at least 155 EMA updates, first sample at the first due cadence after
75%, last sample at or above 99.5% progress, sample parity difference at most
one, nonzero trajectory distance, complete audits, and only `train.py` changed.

- With all integrity conditions, `best_test_acc >=95.50%` is `improvement`.
- A formal improvement below the 95.50 final-16 mean bar retains its tree
  verdict but falsifies the stable-plateau mechanism claim.
- A trustworthy, fully dosed result below 95.50 is `no-improvement`; do not
  retry or switch to a blend or half-life variant in this experiment.
- A realized dose shortfall is `no-improvement`, not a metric-rerun trigger.
- Crash, timeout, wrong GPU, extra evaluation, state/RNG/restore failure,
  nonfinite state, BN nonpositivity, parent-path drift, or incomplete summary is
  invalid.

## Risks

- The normalization is mathematically determined and introduces no fitted
  scalar, but there is no workload-specific evidence that removing the anchor
  improves accuracy.
- Full-state EMA averages BN statistics rather than recalibrating them. Applying
  one kernel to parameters and floating buffers isolates the correction but
  does not make those statistics exact for averaged weights.
- A single fixed-seed run can establish protocol-valid local improvement, not
  isolate all trajectory noise. The formal 0.10-point bar is smaller than known
  max-selection variability, hence the explicit stable-tail and global bars.
- The expected effect is likely modest. This is a precise, low-cost estimator
  study, not a high-confidence route to a large global gain.

## Verification Checklist

1. Prove that only `train.py` differs from EXP-004.
2. Match normalized recurrence and exact weights to FP64 direct references.
3. Verify complete parameter/floating-buffer/integer-buffer semantics.
4. Verify post-update/post-SAM cadence parity, online-state identity, and RNG
   neutrality.
5. Verify one-source evaluation and exact restoration on success and failure.
6. Pass the decisive, accuracy-blind physical-GPU-0 preflight.
7. Run exactly one fixed-seed metric launch and apply the local, stable-tail,
   and global-context rules without retuning.
