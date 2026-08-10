# Proposal: Uniform Full-State Clean-Tail SWA From EXP-004

## Summary

Add cumulative uniform checkpoint averaging directly to EXP-004's validated
WRN/CutMix/clean-tail-SAM package. Starting at charged progress `0.75`, sample
every 31st post-optimizer model state and maintain its arithmetic mean. The
averaging update is sparse, consumes no RNG, adds no forward pass, and is
performed inside the charged step before the existing CUDA synchronization.

```python
SWA_START = 0.75
SWA_UPDATE_EVERY = 31

# After optimizer.step(), with any SAM perturbation already restored.
if progress >= SWA_START and next_step % SWA_UPDATE_EVERY == 0:
    swa_updates += 1
    if swa_updates == 1:
        copy_full_state(swa_state, live_state)
    else:
        lerp_floating_state_(swa_state, live_state, 1.0 / swa_updates)
        copy_integer_buffers_(swa_state, live_state)
```

Before the first sample, the epoch's sole evaluation uses the live model. From
the first sample onward, it evaluates only the full SWA state through an
exception-safe swap and exact live-state restoration. No BatchNorm
recalibration, second evaluation source, EMA, live/SWA interpolation, changed
learning-rate schedule, or accuracy-conditioned checkpoint weighting is
allowed.

Everything else remains EXP-004 at commit `1a8d0de`: PreAct WRN-16-4,
2,748,890 trainable parameters, seed 42, random crop/flip, front-loaded CutMix,
drop path, period-two rho-0.05 clean-tail SAM, Nesterov SGD, cosine annealing,
BF16/channels-last, the fixed 300-second charged budget, and physical GPU 0 via
`CUDA_VISIBLE_DEVICES=0`.

## Why This Is a Distinct Test

EXP-011 proves that sparse full-state trajectory averaging can be integrated
into EXP-004 safely and cheaply, but it tested a particular copy-initialized,
time-decayed EMA. Uniform averaging tests a different estimator with no fitted
decay coefficient. It is not a repeat of EXP-011 and it is not a bias-corrected
EMA: every cadence-selected state receives exactly `1/N` terminal weight.

The classic SWA result motivates averaging late SGD states, and the ICML 2025
averaging study supports combining averaging with learning-rate annealing.
Neither establishes that uniform averaging is superior under this short,
strongly annealed, SAM-alternating tail. This is therefore a low-overhead but
uncertain mechanism test, not a high-confidence route to a large gain.

Rejected variants for this experiment are:

- a windowed or delayed SWA, because another start point would add an
  uncalibrated horizon and confound kernel shape with support;
- parameter-only SWA with latest BatchNorm buffers, because it changes both the
  kernel and parameter/buffer coupling relative to the operationally validated
  EXP-011 convention;
- BN recalibration, because it requires extra train-data exposure and either
  steals charged optimizer time or performs uncharged post-budget training;
- EMA/live interpolation or a shorter EMA half-life, because both introduce a
  coefficient not selected by this lineage;
- simultaneous EMA and SWA evaluation, because it violates the once-per-epoch
  evaluation ceiling and creates an extra checkpoint-selection channel.

## Correct Kernel Arithmetic

The earlier EXP-014 SWA proposal incorrectly treated EXP-011 as a normalized
exponential from the activation boundary. EXP-011 instead copied its first
sample into the EMA shadow, then applied decay updates to later samples. With
the realized 160 samples over 74.7736 seconds, the actual terminal weight of
the first sample was the logged product of subsequent decays, `0.063025`
(about 6.30%). Under the observed near-regular cadence, the newest sample
received approximately 1.7235%, the implemented kernel's effective sample size
was approximately 79.18, and its mean state age was approximately 25.13 seconds.
The exact non-first weights depend on all 159 recorded time intervals; they
must not be reconstructed from the logged mean interval alone.

For `N=160`, uniform SWA instead assigns every sample `1/160 = 0.00625`
(0.625%), has effective sample size exactly 160, and, under the observed nearly
regular 74.7736-second span, mean state age about `74.7736/2 = 37.3868` seconds.
Thus uniform SWA:

- removes the accidental 6.30% point mass on the first clean-tail state;
- reduces the newest state's weight from about 1.7235% to 0.625%;
- approximately doubles effective sample size from about 79.18 to 160;
- nevertheless shifts the estimator's overall center materially earlier, from
  about 25.13 seconds of age to about 37.4 seconds.

Uniform is therefore not simply a less-stale version of EXP-011's EMA. It is
less anchored to the very first state but gives the early and middle clean tail
more aggregate influence. At progress 0.75 the cosine learning rate is still
materially above its endpoint, and those early states may be inferior or may
not lie in a sufficiently tight connected basin. A negative result would be a
credible rejection of this full-support uniform kernel, not of weight
averaging generally.

## Mechanistic Hypothesis

Period-two SAM deliberately creates alternating ordinary- and
sharpness-aware updates in the final clean quarter. Cadence 31 is odd and
coprime to period 2, so consecutive samples alternate those two state classes.
Uniformly centering the complete alternating trajectory could increase the
effective diversity used by the evaluated model and reduce endpoint variance
without reducing optimizer exposure materially.

The counter-hypothesis is at least as plausible: the strongly decaying cosine
schedule makes early clean-tail states stale, classical SWA often relies on a
constant or cyclic late learning rate, and uniformly averaged BatchNorm
statistics are only an explicit state convention rather than exact population
statistics for the averaged parameters. The expected effect may be below the
0.10-point formal resolution of one fixed-seed run.

## Exact State Semantics

Build name-aligned shadow tensors from a fresh inventory of
`model.named_parameters()` and persistent `model.named_buffers()`. Require the
union of those names to equal a separately materialized `state_dict` key set,
with exact shape, dtype, device, and uniqueness checks. Shadows are detached,
non-gradient tensors and must not alias live state, optimizer state, SAM
snapshots, or one another.

The recurrence is:

- On sample 1, copy every parameter and persistent buffer exactly.
- On sample `n > 1`, apply `average += (live-average)/n` (or equivalent
  `lerp_(live, 1/n)`) to every parameter and every persistent floating buffer.
- On every sample, copy each persistent non-floating buffer from live. For this
  model these are BatchNorm `num_batches_tracked` counters. Integer counters
  are never averaged, rounded, or accumulated independently.
- Never include gradients, optimizer momentum, SAM snapshots, RNG states,
  module flags, or nonpersistent buffers.

For BatchNorm, uniformly average `running_mean` and `running_var` with the same
coefficient as parameters, and copy `num_batches_tracked` from the latest
sample. SAM's second pass retains EXP-004's disabled running-stat tracking, so
each optimizer step still contributes exactly one online BN update. Sampling
occurs only after BN flags and unperturbed weights have been restored.

No BN recalibration is performed. Averaged running variances are convex
combinations of nonnegative buffers, but they are not guaranteed to be the
population variances of the averaged parameters. Require all averaged floating
state to be finite and all SWA running variances strictly positive. Interpret
the result as a full-state SWA package; do not attribute it to parameter
averaging alone.

## Cadence and Ordering

Use the charged-time `progress` captured at step entry and the upcoming
one-based `next_step`, exactly as EXP-004 computes LR, CutMix/SAM eligibility,
and drop-path scale. A sample is due iff:

```python
progress >= 0.75 and next_step % 31 == 0
```

The exact production order is:

1. Run the unchanged first forward/backward.
2. On a scheduled SAM step, perturb, replay RNG, run the BN-suppressed second
   pass, and restore parameters and BN flags in `finally`.
3. Execute the sole `optimizer.step()`.
4. Update SWA from the resulting live state if the cadence predicate is true.
5. Synchronize CUDA, compute `dt`, and add it to charged training time.

Consequently SWA never samples perturbed SAM weights. The sparse averaging work
is charged. With odd cadence 31 and `SAM_PERIOD=2`, successive cadence multiples
alternate odd ordinary and even SAM-updated states. Require ordinary/SAM sample
counts to differ by at most one; an even terminal sample count must split
exactly evenly. Harness timing steps that deliberately select one parity must
not feed the production cadence audit.

Uniformity means equal weight per cadence-selected optimizer state, not per
image or per wall-clock interval. Do not time-weight samples or change cadence
when realized step spacing varies.

## Evaluation and Restoration

Keep `EVAL_EVERY=1` and exactly one frozen evaluator call per completed epoch.
Before SWA activation evaluate live state; afterward evaluate only SWA. For an
SWA evaluation:

1. Snapshot the complete live model state, per-module training flags, optimizer
   parameter/momentum identities, and RNG state around state-management work.
2. Copy all SWA parameters and persistent buffers into the existing model.
3. Invoke the frozen evaluator once.
4. Restore the complete live state and module flags in `finally`, including on
   an injected evaluator failure.
5. Prove exact restoration using a fresh `state_dict(keep_vars=True)`
   enumeration, and require unchanged optimizer identities and no RNG movement
   caused by swap/restore.

Evaluation itself may have its normal behavior; the RNG parity check isolates
the SWA state-management operations. Never evaluate live and SWA together,
ensemble logits, choose checkpoints by loss/accuracy, or feed evaluation output
back into the recurrence.

## Required Production Audits

Print compact terminal audits, outside decision logic, for:

- configuration: `uniform_full_state_swa`, start 0.75, cadence 31, and no EMA
  half-life/interpolation/window;
- inventory: parameter, floating-buffer, and integer-buffer tensor/element
  counts; full key coverage; shape/dtype/device; no aliases or gradients; and
  exclusion from optimizer and SAM storage;
- dose: update count, first/last step, progress and charged time, total span,
  interval min/mean/max, and ordinary/SAM counts;
- kernel: theoretical weight sum, min/max/final weight `1/N`, ESS `N`, and
  direct arithmetic-reference error from preflight;
- trajectory: finite nonzero consecutive-sample parameter L2 min/mean/max and
  final SWA/live absolute and relative parameter L2;
- BatchNorm: SWA/live running-mean and running-variance L2, per-channel
  variance-ratio min/mean/max, strictly positive SWA variances, and exact latest
  integer-buffer copies;
- evaluation: live/SWA source counts, evaluator calls, swaps, restoration
  checks, and zero coverage/restore/mode/RNG/nonfinite failures;
- inherited mechanisms: CutMix and SAM counts/ratios, SAM first step/progress,
  epochs, steps, parameter count, and complete final summary.

Any final audit failure must make the process exit nonzero after printing the
diagnostics and summary; metric grepping alone is not sufficient evidence.

## Accuracy-Blind Feasibility Gate

Materialize the exact EXP-004 parent from commit `1a8d0de` under `/tmp` and
compare parent and candidate in one bounded BF16/channels-last harness on
physical GPU 0. Before every GPU command verify physical index 0 and, under
`CUDA_VISIBLE_DEVICES=0`, exactly one visible NVIDIA H20 with matching UUID.
Monkeypatch evaluator calls immediately after import so the preflight cannot
query accuracy or iterate the test loader.

The harness must include five alternating-order paired rounds with the
production-weighted mix of early/ordinary steps, late ordinary steps, and late
SAM steps; a separate consecutive-step sequence must produce at least 30 SWA
samples across both parities. It must also exercise one full-state swap/restore
and synthetic evaluation-shaped forwards without reading evaluator data.
Reseed before each model construction and require candidate and parent online
model, optimizer, BN, and RNG states to match after identical finite step
prefixes.

Proceed to the one metric run only if the first complete valid measurement
passes all of:

- physical and visible GPU identity checks;
- parent round drift `<=0.03`;
- median absolute deviation of paired latency ratios divided by their median
  `<=0.005`;
- weighted median candidate/parent charged-step latency ratio `<=1.02`;
- projected optimizer steps `>=25,200` and projected total runtime `<600s`;
- candidate peak allocated memory `<1.30 GiB`;
- at least 30 cadence samples with ordinary/SAM imbalance at most one;
- exact FP64 arithmetic-mean agreement within a preregistered dtype-aware
  tolerance, full-state coverage, BN positivity, finite nonzero distance,
  optimizer/RNG identity, parent-online equality, and exact success/failure
  restoration.

A completed numeric gate failure is decisive: record a pre-metric failed leaf
without changing start, cadence, state policy, or thresholds. A rerun is
permitted only for an exception or malformed harness output before any numeric
gate is emitted; accuracy is never queried in preflight.

## One-Run Decision Rule

If and only if preflight passes, launch exactly once:

```bash
timeout 600s env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 uv run train.py > run.log 2>&1
```

The parent is EXP-004 at `95.40%`, so the formal parent-relative threshold is
`best_test_acc >=95.50%` (at least 9,550 correct of the fixed 10,000 examples).
The separate scientific stable-tail bar is the mean of the final 16 SWA-source
evaluations `>=95.48%`; report its min/max/final value and
`best_test_acc - final16_mean`. This bar sits 0.08 points above the parent's
final 95.40 and only 0.02 below the formal maximum threshold, demanding a
plateau rather than a fortunate selected checkpoint. EXP-004's full final-16
mean was not durably recorded, so do not claim a paired tail-mean delta against
the parent; the absolute bar is preregistered instead.

Require exit 0, charged seconds in `[299.5,301.0]`, total seconds `<600`, one
evaluation per epoch, a complete summary, unchanged `num_params=2,748,890`,
only tracked `train.py` changed, and no traceback/CUDA/OOM/nonfinite/audit
failure. Full-dose scientific interpretation additionally requires at least
25,200 optimizer steps, at least 155 SWA samples, ordinary/SAM imbalance at
most one, first sample at the first due cadence after progress 0.75, last sample
at or above progress 0.995, and finite nonzero trajectory distance.

Classify once, without retry or post-result tuning:

- A trustworthy `best_test_acc >=95.50%` is a formal local improvement over
  EXP-004, whether or not it exceeds the current goal-wide best of 95.61.
- A formal improvement whose final-16 SWA mean is below 95.48 retains its tree
  verdict but falsifies the stable-plateau hypothesis.
- A trustworthy full-dose result below 95.50 is `no-improvement` and rejects
  this exact uniform full-state package.
- A realized dose shortfall is `no-improvement`, not permission to rerun.
- Wrong GPU, scope/evaluation/RNG/state corruption, extra evaluation, failed
  restoration, nonfinite state, BN nonpositivity, incomplete summary, crash, or
  timeout is invalid/crash according to the goal protocol.

Do not retry with another seed, select a different tail window, fall back to
EMA, alter BN semantics, or interpolate the live endpoint after seeing any
accuracy.

## Risks and Interpretation Limits

- **Stale states are the primary risk.** Equal full-tail weighting produces an
  older average than EXP-011's actual EMA even though it removes the anomalous
  first-state anchor. The earliest clean-tail states may be too high-LR and too
  under-converged for uniform averaging.
- **BatchNorm is package-level.** Averaged running buffers are explicit and
  auditable but not recalibrated statistics for averaged parameters. A null
  cannot distinguish parameter-kernel harm from BN-state mismatch.
- **Classical SWA conditions differ.** The parent retains strongly annealed
  cosine LR rather than a constant/cyclic SWA tail to isolate averaging. This
  may collapse useful trajectory diversity.
- **Historical control only.** There is one fixed-seed candidate run and no
  contemporaneous full parent rerun. Tiny timing differences can change the
  number of updates, and test maxima vary across epochs. Report dose, final-16
  mean, and selection premium; do not claim statistical or causal superiority
  from a narrow pass.
- **Effect size may be too small.** Averaging literature suggests mild
  generalization gains. EXP-011 validates averaging as a package, but does not
  show that uniform weighting can clear even the lower local 0.10-point bar.

## Recommendation

Advance as a technically clean, low-cost candidate with medium confidence in
feasibility and low-to-medium confidence in formal improvement. Its value is a
direct test of whether EXP-004's alternating clean-tail SAM trajectory benefits
from maximal uniform smoothing. The proposal is scientifically candid: it may
recover EXP-011-like gains without an EMA horizon, but its substantially older
state center and unrecalibrated full-state BatchNorm convention could erase the
benefit.
