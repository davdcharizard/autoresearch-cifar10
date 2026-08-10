# Proposal: Uniform Full-State Clean-Tail SWA

## Summary

Replace EXP-011's exponential averaging kernel with cumulative, uniform
checkpoint averaging over exactly the same clean/SAM tail. Keep the averaging
start at 75% charged progress, keep cadence 31, sample the same post-optimizer
and post-SAM-restoration states, and keep one-source evaluation once per epoch.
The only intended change is how the sampled full-model states are weighted:
every state observed since activation has equal weight instead of exponentially
less weight with age.

The selected rule is:

```python
SWA_START = 0.75
SWA_UPDATE_EVERY = 31

# At the nth eligible sample:
if n == 1:
    average = live
else:
    average.lerp_(live, 1.0 / n)
```

Apply this recurrence to every model parameter and persistent floating buffer.
Copy persistent integer buffers from the latest sampled state. Do not average
optimizer state, SAM snapshots, gradients, or non-model objects. Before the
first sample, evaluate the live model; afterward, swap in the full SWA state for
the epoch's sole evaluation and restore the live state exactly.

Everything else remains EXP-011: PreAct WRN-16-4, random crop/flip, front-loaded
CutMix, drop path, clean-tail period-two SAM, cosine learning-rate annealing,
BF16/channels-last execution, seed 42, 300 charged seconds, and physical GPU 0.

## Candidate Rules Considered

### 1. Uniform full-state SWA over the existing clean tail - selected

This rule uses exactly the same support, samples, cadence, and evaluation
activation as EXP-011. It changes only the averaging kernel. It introduces no
new fitted coefficient: the start and cadence are inherited from the parent,
and `1/n` is forced by the definition of an arithmetic mean.

The classic SWA result motivates uniform late-trajectory averaging, while the
ICML 2025 averaging study supports pairing checkpoint averaging with learning-
rate annealing rather than replacing the existing decay. Neither paper proves
that uniform averaging beats EMA in this exact 75-second tail, so this remains
a one-shot mechanism test rather than a claimed theorem.

### 2. Fixed EMA/live interpolation - rejected

A state such as `0.5 * EMA + 0.5 * live` would reduce exponential lag, but the
blend coefficient has no calibration evidence in this lineage. It also gives a
single endpoint checkpoint a 50% point mass, sharply reducing smoothing, while
EXP-011 did not measure live-tail accuracy to show that endpoint bias is useful.
Applying the same interpolation to BatchNorm buffers would add another
nonlinear state-semantics choice. A midpoint coefficient is simple but still
arbitrary, making this closer to unvalidated coefficient tuning than a clean
test of a standard estimator.

### 3. Shorter-horizon EMA - rejected

Changing the 18.75-second half-life would stay within EXP-011's exponential
family and would mainly tune an existing scalar. The observed 1.51% final
EMA/live parameter distance does not reveal whether the parent is too stale or
too noisy, so no shorter horizon is evidence-selected.

### 4. Windowed or delayed uniform averaging - rejected

A final-half-tail window could reduce stale-state bias, but it would add an
unvalidated cutoff and either change evaluation support or require a ring of
full checkpoints. That would confound kernel shape with horizon. Cumulative
full-tail SWA gives the more controlled first comparison.

## Motivation and Mechanistic Hypothesis

EXP-011 is the 95.61% global-best parent. Its final 16 EMA evaluations average
95.493125%, range from 95.44% to 95.61%, and end at 95.46%. The bottleneck is a
stable late generalization gain, not throughput or memory. The current EMA is
sound and cheap, but one fixed-seed run does not establish that its exponential
recency bias is the best summary of the annealed SAM trajectory.

EXP-011 observed 160 cadence samples across 74.7736 charged seconds, with mean
sample spacing 0.470274 seconds and exactly 80 ordinary/80 SAM states. At the
terminal evaluation, the 18.75-second-half-life EMA gives the oldest sample
about 1/16 the unnormalized weight of the newest. Under the observed nearly
regular cadence, its normalized newest weight is about 1.84%, its normalized
oldest weight about 0.12%, and its effective sample size about 102. Uniform SWA
would assign all 160 samples 0.625%, for effective sample size 160.

This is materially different from a half-life micro-adjustment. Relative to
the parent, uniform averaging gives the oldest clean-tail states roughly five
times more normalized influence, gives the newest state roughly one third as
much influence, and increases effective sample count by about 57%. Its final
mean state age is about 37.4 seconds rather than the truncated exponential
kernel's approximately 22.1 seconds.

The hypothesis is that the clean-tail trajectory remains within a connected
useful basin while period-two SAM supplies local trajectory diversity. Uniform
weighting should center that trajectory more completely and reduce checkpoint
variance beyond EXP-011's EMA, improving the stable averaged-model plateau.
The explicit counter-hypothesis is stale-state bias: the early clean-tail
states have a materially higher cosine learning rate, so giving them equal
weight may pull the model away from the better-converged endpoint.

This direction is also distinct from EXP-011's failed children. EXP-012 added
spatial erasure, while EXP-013 replaced affine classifier geometry and, at full
preregistered dose, lowered best accuracy by 0.50 points and the final-16 mean
by about 0.42 points. Neither changed the parent averaging kernel, so their
negative results do not answer the uniform-versus-exponential question. This
proposal preserves the affine classifier and every online training mechanism.

## Evidence Basis

- `01-definition.md` fixes the 95.71% child threshold, physical-GPU-0 scope,
  300-second charged budget, and once-per-epoch validation ceiling.
- `02-system-understanding.md` identifies the stable 95.49 EMA plateau as the
  current limiter and confirms that throughput and memory are not limiting.
- `experiments/011/04-analysis.md` supplies the exact 160-sample, 74.7736-second,
  80/80-parity parent trajectory, its 18.75-second EMA horizon, final distance,
  tail statistics, state semantics, and negligible overhead.
- `experiments/013/04-analysis.md` establishes that the fixed-scale cosine
  classifier was a full-dose but mechanism-unrelated failure, leaving averaging
  shape unresolved.
- `experiments/014/papers/when-where-why-average.md` supports checkpoint
  averaging alongside annealing but does not select EMA, uniform averaging, or
  interpolation for this workload.
- `knowledge/papers/stochastic-weight-averaging.md` motivates uniform late
  checkpoint averaging and explicitly flags learning-rate diversity and
  BatchNorm handling as the two central risks.
- `knowledge/papers/how-to-scale-your-ema.md` explains why EMA horizons must be
  exposure-aware. It argues against importing a new decay scalar; uniform SWA
  avoids that extra horizon parameter while preserving the parent's cadence and
  activation support.

## Exact Averaging Semantics

Construct the averaging state from `model.named_parameters()` and
`model.named_buffers()` with the same complete key-set checks as EXP-011.
Preallocate non-gradient shadows using `torch.empty_like(...,
memory_format=torch.preserve_format)`.

On a one-based training step, sample if and only if both conditions hold:

```python
progress >= SWA_START
next_step % SWA_UPDATE_EVERY == 0
```

Use the charged-time progress captured at step entry, matching the LR, CutMix,
SAM, and parent EMA boundary semantics. Perform the update after the sole
`optimizer.step()` and after exact SAM restoration. Thus an eligible SAM sample
contains the model after the SAM-derived SGD update, never the perturbed model.

For sample count `n` after incrementing:

- At `n == 1`, copy all live parameters, floating buffers, and integer buffers.
- At `n > 1`, update each averaged parameter and floating buffer with
  `torch._foreach_lerp_(average, live, 1.0 / n)`.
- Copy each integer buffer from live on every sample.
- Record the pre-update live-to-previous-sample parameter distance, but do not
  synchronize inside the charged step merely to print it.

The recurrence yields an arithmetic mean in exact real arithmetic. Floating-
point roundoff is accepted only within a preregistered numerical tolerance
against a direct FP64 arithmetic-mean reference in unit tests. Do not add
momentum correction, bias correction, a burn-in, an EMA warm start, a live
interpolation, checkpoint weighting by accuracy/loss, or a second averaging
rule.

Uniformity here means uniform over cadence-selected optimizer states, not over
images or exact wall-clock intervals. Cadence intervals were narrow in EXP-011,
and retaining the parent's step cadence is necessary for a controlled kernel
comparison. Do not time-weight samples post hoc.

## Cadence Parity With Period-Two SAM

Keep cadence 31 because it is odd and therefore coprime to `SAM_PERIOD=2`.
Successive multiples of 31 alternate step parity, so cadence samples alternate
ordinary and SAM-trained states after the common 75% boundary. An even sample
count is exactly balanced; an odd count can differ by only one.

Log ordinary and SAM sample counts and require their absolute difference to be
at most one. The full run should produce approximately 158-160 updates. A
separate preflight sequence must use consecutive production step IDs; selected
even-only timing IDs must not be allowed to contaminate the cadence audit, as
that harness error was already found in EXP-011.

## Full-State BatchNorm Semantics

Average every persistent floating BatchNorm buffer, including `running_mean`
and `running_var`, with the same uniform coefficient as parameters. Copy each
`num_batches_tracked` integer buffer from the latest sampled checkpoint. At
evaluation, swap parameters and all persistent buffers together so the
evaluated state is one explicitly defined full-state arithmetic average.

Do not perform BatchNorm recalibration. Standard SWA often recomputes BatchNorm
statistics after parameter averaging, but an extra train-data pass would add
data exposure, consume RNG, and either steal charged optimizer time or become
uncharged post-budget work. It would make this a two-part intervention. The
parent already established full-state shadow averaging as an operationally
valid, low-cost convention; retaining that convention isolates exponential
versus uniform weighting.

Arithmetic means of running variances are convex combinations of nonnegative
buffers and therefore should remain nonnegative, but this does not make them
the exact population statistics of the averaged parameters. Treat full-state
SWA as the tested package and do not claim parameter-only SWA causality. Audit
all averaged BN buffers for finiteness, require every running variance to be
strictly positive, and report SWA/live running-mean L2, running-variance L2,
and per-channel variance ratios.

SAM's second forward continues to have BN tracking disabled exactly as in the
parent. The SWA sample is taken only after that tracking flag and the unperturbed
weights have been restored, so each optimizer step contributes one BN update.

## Evaluation Semantics

Keep `EVAL_EVERY=1` and invoke the frozen evaluator exactly once per completed
epoch. Before the first SWA sample, evaluate live state. After the first sample,
evaluate only SWA state. This matches EXP-011's activation boundary and avoids
giving the candidate an extra live checkpoint selection channel.

For every SWA evaluation:

1. Record SWA/live distance diagnostics without modifying model state.
2. Snapshot all live parameters and persistent buffers, optimizer identities,
   module training flags, and RNG state around state-management operations.
3. Copy the complete SWA state into the existing model and call the evaluator
   once.
4. In a `finally` block, restore the complete live state and module flags.
5. Verify a fresh `state_dict(keep_vars=True)` enumeration is bitwise equal to
   the restore snapshot; verify optimizer parameter and momentum-buffer
   identities are unchanged and swap/restore consumed no RNG.

Do not evaluate both live and SWA at an epoch, average logits, ensemble models,
select the averaging rule using intermediate accuracy, or use evaluation
results to alter the state recurrence.

## Parent Integrity

The online model and optimizer path must remain bit-identical to EXP-011 for an
identical finite step prefix. Preserve all model construction and initialization,
data transforms and generators, CutMix probability/geometry, drop-path draws,
learning-rate schedule, SAM rho/start/period/replay/BN semantics, optimizer
hyperparameters, and global seeds.

The shadow recurrence consumes no RNG and is excluded from the optimizer and
SAM parameter inventory. Its state update is placed before the existing CUDA
synchronization and charged-time increment, exactly where the parent EMA update
ran. Timing may change terminal dose, but paired tests must prove that parent
and candidate online parameters, optimizer state, BN state, and RNG remain
equal after the same synthetic workload.

## Compute, Memory, and Overhead

Uniform SWA needs the same kinds of shadows as the parent EMA: one averaged
full state, one exact restore state, and one previous-parameter state for
trajectory diagnostics. It removes time-decay exponentiation and performs the
same sparse foreach copy/lerp pattern. Parameter count remains 2,748,890 and no
trainable tensor is added.

Expected peak allocation is near EXP-011's 1,222.4 MiB and safely below 1.30
GiB. The update happens about twice per second and requires no model forward,
data transfer, or extra evaluator call. Expected charged latency ratio is near
1.00 and should retain roughly 25,500-25,800 optimizer steps. Total runtime
should remain near 448 seconds and below the 600-second outer limit.

## Required Audits

Log the following without changing training or evaluation decisions:

- configuration: averaging type `uniform_full_state_swa`, start 0.75, cadence
  31, and explicit absence of EMA half-life or interpolation coefficient;
- complete state inventory: parameter, floating-buffer, and integer-buffer
  tensor/element counts, shape/dtype/device coverage, alias checks, gradient
  flags, optimizer exclusion, and SAM-snapshot exclusion;
- dose: updates, first/last step, progress and charged time, span, ordinary/SAM
  counts, and interval min/mean/max;
- uniform weights: final `1/N`, weight sum 1 within tolerance, minimum and
  maximum theoretical checkpoint weight both `1/N`, and effective sample count
  `N`;
- trajectory: finite nonzero consecutive-sample parameter L2 min/mean/max and
  final SWA/live absolute and relative parameter L2;
- BatchNorm: SWA/live running-mean and running-variance L2, variance-ratio
  min/mean/max, strict-positive averaged variances, and copied latest integer
  buffers;
- evaluation: live/SWA source counts, swaps, restore checks, evaluator-call
  count, and zero restore, coverage, nonfinite, RNG, or mode failures;
- parent mechanisms: CutMix and SAM eligible/applied counts, SAM first
  step/progress, epoch count, step count, and unchanged parameter count.

The uniformity claim must be validated structurally and by synthetic arithmetic,
not inferred from terminal diagnostics alone.

## Accuracy-Blind Preflight

Materialize EXP-011 from commit `d68f73a` and compare parent/candidate on
physical GPU 0 in one BF16/channels-last harness. Query no accuracy. Reseed
before construction and require identical initial model/RNG states.

Use five alternating-order paired rounds that cover ordinary steps, clean
period-two SAM steps, cadence updates across both parities, and evaluation
swap/restore. Preserve the parent's workload weighting. Proceed to the sole
metric run only if:

- physical GPU 0 is the approximately 97,871 MiB NVIDIA H20 and is the only
  visible CUDA device;
- parent round drift is at most 0.075 and median absolute deviation divided by
  median is at most 0.005;
- candidate/parent weighted median charged latency is at most 1.02;
- projected optimizer steps are at least 25,200 and projected total runtime is
  below 600 seconds;
- peak allocated memory is below 1.30 GiB;
- at least 30 production-order SWA samples split ordinary/SAM by at most one;
- direct arithmetic-mean, full-state coverage, BN positivity, finite-distance,
  RNG, optimizer-identity, and exact-restore checks all pass;
- parent and candidate online model and optimizer states match exactly after
  equal workloads.

The first complete valid timing measurement is decisive. Correct a harness bug
only when no gate metric or accuracy was emitted; do not optimize the SWA rule
after observing a gate or metric.

## Testable Hypothesis and Decision Rule

Parent EXP-011 has `best_test_acc=95.61%`, so formal improvement requires
`best_test_acc >= 95.71%`. The preregistered hypothesis is:

> Uniform full-state averaging of all cadence-31 clean-tail states will use the
> period-two SAM trajectory more completely than the parent EMA, reach at least
> 95.71% best test accuracy, and raise the final-16 averaged-checkpoint mean
> from 95.493125% to at least 95.60%, without reducing realized optimizer dose
> below 25,300 steps.

Run one fixed seed and one averaging rule. Apply these classifications:

- `best_test_acc >= 95.71%` with all hard constraints satisfied is a formal
  improvement.
- A final-16 SWA mean below 95.60% does not invalidate a formal improvement,
  but it falsifies the stable-plateau mechanism claim.
- Fewer than 25,300 optimizer steps, fewer than 155 SWA updates, a first sample
  outside the first due cadence after 75% progress, a final sample below 99.5%
  progress, or zero trajectory distance is a mechanism-dose failure and must
  not be presented as a full-dose test.
- Parity difference above one, extra evaluation, state coverage/restoration/RNG
  failure, nonfinite state, BN nonpositivity, CutMix/SAM semantic drift, crash,
  timeout, or incomplete summary makes the run invalid.
- A trustworthy full-dose result below 95.71% is `no-improvement`; do not retry,
  delay the SWA start, introduce a window, interpolate live state, or change
  cadence after seeing accuracy.

## Risks and Mitigations

- **Stale-state bias:** uniform SWA gives early, higher-LR clean-tail states
  much more influence than EMA. Matching the parent's full support isolates
  this risk; a negative result rejects this exact uniform kernel.
- **Annealed trajectory collapse:** classic SWA often maintains a larger or
  cyclic LR to preserve diversity. This experiment keeps cosine annealing
  because changing LR would confound the comparison; nonzero consecutive and
  average/live distances audit whether useful movement exists.
- **BatchNorm mismatch:** averaged running statistics are not recalibrated
  population statistics for averaged weights. Full-state averaging is explicit,
  convex, audited, and directly comparable to the parent's full-state EMA.
- **Maximum-selection noise:** historical sub-0.30-point gains are not strongly
  resolved by one seed. Report the final-16 mean/range and final accuracy beside
  the formal maximum; do not claim broad superiority from a bare 0.10-point
  pass.
- **Throughput-dependent dose:** the wall-clock protocol changes realized
  samples with latency. Use paired feasibility gates and terminal dose gates;
  never metric-retry a low-dose realization.
- **Silent online interference:** shadow aliases, RNG movement, or failed
  restore could corrupt the parent path. Exact ownership, RNG, optimizer, and
  fresh-state restoration assertions fail loudly.

## Verification Checklist

1. Compile `train.py`, run `git diff --check`, and prove only `train.py` differs
   from parent commit `d68f73a`; inspect evaluator and summary data flow.
2. On scalar and tiny full-state fixtures, compare every online SWA update to a
   direct FP64 mean, including irregular sample values, first-copy behavior,
   floating buffers, and latest integer buffers.
3. Prove cadence 31 samples alternate period-two SAM parity, begin only at the
   first due post-boundary step, and never sample perturbed SAM weights.
4. Verify all shadows are finite, non-gradient, non-aliased, and excluded from
   optimizer/SAM ownership; require exact state-dict key coverage.
5. Exercise successful and injected-failure SWA evaluation. Require one
   evaluator call, exact full-state/mode/RNG restoration, and unchanged optimizer
   parameter and momentum identities.
6. Verify BN tracks once on a SAM batch, SWA running variances remain positive,
   and no BN recalibration or extra data forward occurs.
7. Pass the accuracy-blind parent-relative physical-GPU-0 preflight above.
8. Launch exactly once with
   `timeout 600s env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 uv run train.py > run.log 2>&1`.
9. Require exit 0, 299.5-301.0 charged seconds, total time below 600 seconds,
   one evaluation line per epoch, complete summary/audits, and the fixed dose
   and integrity gates.
10. Durably transcribe the complete summary, final-16 SWA values and statistics,
    state/dose diagnostics, preflight measurements, and adversarial result
    review before deleting transient logs.
