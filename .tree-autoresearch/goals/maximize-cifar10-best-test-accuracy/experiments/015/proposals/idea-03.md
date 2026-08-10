# Proposal: Activation-Anchored Bias-Corrected Charged-Time EMA

## Summary

Keep EXP-011's successful cadence-31, 18.75-second-half-life, full-state EMA,
but remove the unintended point mass created by copy-initializing its first
sample. Initialize the EMA mass at zero at the exact charged-time activation
boundary and normalize every floating-state update by accumulated exponential
mass. This is a fixed estimator correction, not a half-life sweep or an
accuracy-selected live blend.

The one selected operating point is:

```python
EMA_START = 0.75
EMA_UPDATE_EVERY = 31
EMA_TAIL_HALF_LIVES = 4.0
EMA_HALF_LIFE_S = (1.0 - EMA_START) * TIME_BUDGET_S / EMA_TAIL_HALF_LIVES
EMA_ACTIVATION_TIME_S = EMA_START * TIME_BUDGET_S

# Initial state
mass = 0.0
last_sample_time = EMA_ACTIVATION_TIME_S

# At each cadence-selected sample_time after optimizer.step() and SAM restore:
decay = 2.0 ** (-(sample_time - last_sample_time) / EMA_HALF_LIFE_S)
new_mass = decay * mass + (1.0 - decay)
alpha = (1.0 - decay) / new_mass
corrected.lerp_(live, alpha)
mass = new_mass
last_sample_time = sample_time
```

At the first sample `alpha == 1` algebraically, so copy live state exactly to
avoid relying on uninitialized shadow contents. Thereafter use the normalized
`alpha`. Apply it identically to all parameters and persistent floating
buffers. Copy persistent integer buffers from the latest sample. Before the
first sample evaluate live state; after it evaluate only the corrected EMA,
exactly once per epoch, through EXP-011's exception-safe full-state swap.

Preserve everything else from parent commit `d68f73a`: PreAct WRN-16-4,
initialization, data stream, seed 42, front-loaded CutMix, drop path, clean-tail
period-two SAM, SGD/Nesterov, cosine LR, BF16/channels-last, the 300-second
charged budget, and physical GPU 0 via `CUDA_VISIBLE_DEVICES=0`.

## Why Bias Correction Is Selected

### Selected: activation-anchored bias correction

EXP-011's code copies the first sampled state into the EMA and then applies
time-decayed lerps. The terminal coefficient of that first state is therefore
the product of all later decays: `2**(-74.7736/18.75) = 0.063025`, or 6.30%.
That is not the approximately 0.12% oldest weight of a normalized exponential
point-sample kernel. Claude's EXP-014 adversarial review identified this exact
copy-in anchor and recommended bias correction as the cleanest isolation.

Bias correction preserves the already successful horizon, cadence, support,
sample parity, online optimization, and one-source evaluation policy. It adds
no accuracy coefficient. The normalization is forced by the estimator:

`mass_n = 1 - product(decay_1 ... decay_n)`

where the first interval runs from the fixed activation time `225.0 s` to the
first cadence sample. The normalized recurrence is algebraically equivalent to
maintaining a zero-initialized biased numerator and dividing it by `mass_n`,
but it requires no second floating-state shadow and no division during eval.

### Rejected: fixed EMA/live blend

A `0.5 EMA + 0.5 live` endpoint has no lineage evidence for 0.5 and would place
half the evaluation mass on a single stochastic endpoint. EXP-011 did not
measure live-tail accuracy, so even the sign of a useful live correction is
unknown. Blending BatchNorm statistics would introduce another nonlinear state
choice. It is less falsifiable than removing a known estimator artifact.

### Rejected: shorter half-life

Halving 18.75 seconds to 9.375 seconds would reduce lag, but it tunes an exposed
scalar using no parent trajectory comparison. The 1.51% final EMA/live
parameter distance does not establish that the parent is over-smoothed rather
than appropriately denoised. The NeurIPS EMA scaling evidence says horizons
must be exposure-aware; it does not select a shorter horizon here.

## Exact Kernel Implications

For EXP-011's realized trajectory, activation is `225.0 s`, the first sample is
at `225.1324 s`, the last at `299.9060 s`, there are 160 samples, mean cadence
interval is `0.470274 s`, and the later-sample span is `74.7736 s`.

The parent copy-in EMA has, under the observed near-regular cadence:

- oldest-state weight: `6.3025%`;
- newest-state weight: approximately `1.7235%`;
- effective sample size: approximately `79.18`;
- mean state age: approximately `25.13 s`.

The selected activation-anchored correction has total terminal mass
`1 - 2**(-(299.9060-225.0)/18.75) = 0.937282`. Using the observed mean spacing
for the descriptive approximation gives:

- oldest-state weight: approximately `0.0328%` because the boundary-to-first
  interval was only `0.1324 s`;
- newest-state weight: approximately `1.8388%`;
- effective sample size: approximately `101.47`;
- mean state age: approximately `21.80 s`.

Thus the intervention mainly removes a stale 6.30% first-state anchor and
redistributes it across the existing exponential tail. It does not make the
kernel aggressively endpoint dominated: the newest checkpoint remains below
2% and effective sample size increases by about 28%. Exact run values must be
reconstructed from all recorded intervals, not asserted from these rounded
parent statistics.

The scientific hypothesis is narrow: the first clean-tail state occurs at a
substantially higher cosine LR than the endpoint, so removing its accidental
point mass should reduce stale-state bias while retaining most of EMA's
variance reduction. This could lift the stable averaged tail. The candid
counter-hypothesis is that the 6.30% anchor was beneficial regularization;
normalization may follow late trajectory noise more closely and lose accuracy.
Cross-workload averaging evidence supports averaging with annealing but does
not establish the direction or a 0.10-point effect on this workload.

## Complete State Semantics

Build shadows from `model.named_parameters()` and `model.named_buffers()` and
assert their names exactly equal a fresh `model.state_dict()` key set.

- **Parameters:** all 44 trainable parameter tensors use the normalized
  recurrence. Shadows are same shape/dtype/device/memory format, non-gradient,
  non-aliased, and excluded from optimizer and SAM inventories.
- **Floating buffers:** all 26 persistent floating buffers, including every BN
  `running_mean` and `running_var`, use the identical normalized recurrence.
  This retains EXP-011's tested full-state convention and isolates only kernel
  normalization. It is not claimed to equal BN population-stat recomputation.
- **Integer buffers:** all 13 `num_batches_tracked` buffers copy the latest
  sampled live value. They are never averaged or divided by mass.
- **Excluded state:** optimizer momentum, gradients, SAM snapshots, RNG state,
  module modes, evaluator state, counters, and nonpersistent objects are not
  averaged.

Sampling remains one-based step divisible by 31 with step-entry progress at
least 0.75. It occurs after the sole optimizer step and after SAM restoration,
before the existing synchronization and charged-time increment. The first
decay interval begins at the derived charged-time boundary, not time zero and
not the first sample. Cadence 31 remains odd, so samples alternate ordinary and
SAM-derived states and must differ in counts by at most one.

No BN recalibration, extra data replay, live/EMA double evaluation, logit
ensemble, checkpoint selection, or evaluation-driven kernel update is allowed.

## Required Audits

The implementation must durably report:

- fixed configuration and derived activation/half-life values;
- state tensor/element inventory, complete key coverage, shape/dtype/device
  checks, alias/gradient checks, optimizer and SAM exclusion;
- update count, first/last step/progress/time, boundary-to-first interval,
  later interval min/mean/max, and ordinary/SAM counts;
- decay min/mean/max, correction `alpha` min/mean/max, terminal mass, residual
  missing mass, and agreement of mass with
  `1 - 2**(-(last_time-EMA_ACTIVATION_TIME_S)/EMA_HALF_LIFE_S)`;
- exact reconstructed checkpoint weights from recorded intervals: sum, oldest,
  newest, min/max, effective sample size, and mean state age; also reconstruct
  the parent copy-in kernel for a report-only paired comparison;
- finite nonzero consecutive-sample L2 and final corrected-EMA/live absolute
  and relative parameter L2;
- corrected/live BN mean and variance L2, variance-ratio min/mean/max, strictly
  positive corrected running variances, and latest integer-buffer equality;
- live/corrected evaluation counts, evaluator calls, swaps, restore checks,
  module-mode checks, and zero coverage/restore/nonfinite/RNG failures;
- unchanged CutMix/SAM counts and semantics, training loss diagnostic, epochs,
  steps, parameter count, charged/total time, and peak VRAM.

Synthetic arithmetic must compare the recurrence with an FP64 direct weighted
sum for irregular intervals, including a short first interval, one sample,
multiple samples, floating BN buffers, and integer latest-copy behavior.

## Accuracy-Blind Preflight

Materialize parent `d68f73a` and compare parent/candidate on physical GPU 0 in
one BF16/channels-last harness. Before every GPU command require physical GPU 0
and the sole visible CUDA device to be an NVIDIA H20 with approximately 97,871
MiB. Monkeypatch the evaluator before any trace so no accuracy or test batch is
queried.

Use five alternating-order rounds with the parent's measured workload mixture:
ordinary early steps, CutMix steps, clean ordinary steps, period-two SAM steps,
cadence updates across both parities, and synthetic evaluation swap/restore.
The first complete numeric result is decisive. A retry is allowed only for an
exception, missing/malformed output, or failed harness assertion before any
numeric gate is emitted; a numeric failure is not a harness error.

Proceed to the sole metric run only if all of these pass:

- parent round drift is at most 3%; parent median absolute deviation divided by
  parent median is at most 1%;
- median candidate/parent weighted charged latency is at most `1.02`, and every
  round ratio is at most `1.05`;
- projected optimizer steps are at least 25,200, projected EMA samples at least
  155, projected epochs at least 128, and projected total runtime below 600 s;
- candidate-only model + optimizer + SAM + corrected-EMA state peak allocation
  is below 1.30 GiB;
- at least 30 production-order candidate samples cover both SAM parities;
- direct kernel arithmetic, mass identity, online parent-state equality after
  equal work, state coverage, BN positivity, RNG preservation, optimizer
  identity, and exact success/failure-path restoration all pass.

No accuracy may be queried in preflight, and no coefficient, cadence, start,
half-life, threshold, or state policy may change after any numeric gate.

## One Metric Launch and Decision Rule

After the complete preflight passes, launch exactly once:

```bash
timeout 600s env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 uv run train.py > run.log 2>&1
```

Parent EXP-011 is 95.61%, so formal improvement requires
`best_test_acc >= 95.71%` (exactly at least 9,571 correct of 10,000). The
scientific stable-tail bar is final-16 corrected-EMA mean `>=95.69%`; report
best-minus-tail-mean premium separately. The bar is intentionally demanding
because the parent final-16 mean is only 95.493125% and the diagnosed need is a
stable gain, not another selected maximum.

Require exit 0, 299.5-301.0 charged seconds, total below 600 seconds, exactly
one evaluation line per epoch, at least 25,200 optimizer steps, at least 155 EMA
updates, first sample at the first due cadence after 75%, last sample at or
above 99.5% progress, parity difference at most one, nonzero trajectory
distance, complete summary/audits, and only `train.py` changed.

- `best_test_acc >=95.71%` with all hard constraints and integrity conditions is
  a formal improvement.
- A formal improvement with final-16 mean below 95.69% is retained formally but
  falsifies the stable-plateau mechanism claim.
- A trustworthy full-dose result below 95.71% is `no-improvement`; do not retry
  or then test a blend/shorter half-life on the same experiment.
- A dose shortfall is `no-improvement` and not a rerun trigger.
- Crash, timeout, extra evaluation, wrong GPU, state/RNG/restore failure,
  nonfinite state, BN nonpositivity, parent-mechanism drift, or incomplete
  summary is invalid.

## Risks and Interpretation

- **Coefficient evidence:** no new fitted coefficient is introduced. The
  18.75-second half-life is inherited from the successful parent; bias
  normalization is mathematically determined. This still does not supply
  workload-specific evidence that removing the anchor improves accuracy.
- **Over-smoothing versus under-smoothing:** the parent may benefit from its
  stale anchor, or the corrected kernel may be too responsive to noisy late SAM
  iterates. One seed and one-source evaluation cannot distinguish all
  bias/variance explanations; a null rejects only this fixed correction.
- **BatchNorm mismatch:** averaging BN statistics is an explicit package
  convention, not exact recalibration for averaged weights. Applying the same
  correction to parameters and floating buffers avoids an untested hybrid but
  does not eliminate this limitation.
- **Expected effect size:** the intervention is cheap and well isolated, but
  literature describes generalization gains from averaging as mild. Removing a
  6.30% anchor may be below the required 0.10-point formal resolution. It is a
  defensible mechanism test, not a high-confidence large-gain proposal.

## Verification Checklist

1. Compile and statically prove only `train.py` differs from `d68f73a`.
2. Verify the exact normalized recurrence and reconstructed weights against
   direct FP64 references under regular and irregular charged-time intervals.
3. Verify complete parameter/floating-buffer/integer-buffer semantics and
   exact first-sample behavior.
4. Verify cadence parity, post-optimizer/post-SAM ordering, unchanged online
   state and RNG, and no shadow optimizer/SAM ownership.
5. Exercise successful and injected-failure evaluation restoration with one
   evaluator call and no BN recalibration.
6. Pass the decisive accuracy-blind physical-GPU-0 preflight and all latency,
   dose, memory, kernel, and integrity gates.
7. Run exactly one fixed-seed metric launch, record the formal and final-16
   scientific outcomes, and never retune from intermediate or final accuracy.
