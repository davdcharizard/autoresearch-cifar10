# Proposal: CIFAR-Grounded High-Alpha Lookahead Without Evaluation EMA

## Summary

Wrap EXP-004's Nesterov SGD in parameter-only Lookahead from the first optimizer
step, using one fixed setting selected before any local measurement:

```text
LOOKAHEAD_K = 5
LOOKAHEAD_ALPHA = 0.8
LOOKAHEAD_START_STEP = 1
LOOKAHEAD_MOMENTUM_POLICY = retain
LOOKAHEAD_STATE = parameters_only
LOOKAHEAD_EVAL_SOURCE = slow_parameters_only
```

After every five completed Nesterov updates, move a persistent slow parameter
copy 80% of the way toward the fast endpoint and copy the result back into the
same optimizer-owned `Parameter` objects. Retain the Nesterov momentum buffers
exactly. Keep EXP-004's architecture, seed, data order, time-indexed learning
rate and drop-path schedules, front-loaded CutMix, clean-tail period-two SAM,
BatchNorm behavior, and once-per-epoch evaluator unchanged. Add no training or
evaluation EMA and no other averaging mechanism.

Evaluate exactly one source: the Lookahead slow parameters with the current
online BatchNorm buffers. A complete CIFAR epoch has 195 optimizer steps, which
is divisible by five, so ordinary epoch-end evaluations already have
`fast == slow`; an exception-safe parameter swap makes the source explicit and
also handles the budget-truncated final epoch. Never evaluate both slow and
fast in one epoch.

This is a **scientifically weak-to-moderate proposal**. Removing EXP-011's EMA
eliminates the old nested-smoothing objection, and `alpha=0.8` sharply reduces
the canonical `alpha=0.5` displacement loss. However, EXP-004's final accuracy
already equaled its best, so excessive online-iterate variance is not a
diagnosed limiter. Lookahead may complement late SAM by changing the trajectory
that SAM follows, but it may instead contract useful Nesterov/SAM travel. A
realistic preregistered effect range is roughly **-0.10 to +0.15 percentage
points**. It can clear the local `95.50%` threshold, but it is not a high-power
bet and is unlikely on present evidence to beat the `95.61%` global best.

## Literature-Grounded Fixed Choice

The original NeurIPS 2019 Lookahead paper defines the slow/fast recurrence,
shows compatibility with SGD and momentum-family inner optimizers, and reports
CIFAR evidence with small `k`, commonly `k=5`. Its analysis explicitly compares
Lookahead and SGD under the same inner learning rate; the paper's empirical
claim is that Lookahead can improve standard optimizers without requiring a
new inner optimizer.

Use `k=5, alpha=0.8`, not the earlier proposal's `k=5, alpha=0.5`. The NeurIPS
2021 follow-up *Towards Understanding Why Lookahead Generalizes Better Than SGD
and Beyond* uses `alpha=0.8` for its CIFAR-10/100 experiments, while identifying
five inner steps as the standard small-cycle setting. This is a direct
CIFAR-grounded pair, fixed without a local sweep. Relevant sources are:

- `knowledge/papers/lookahead-optimizer.md`
- <https://papers.nips.cc/paper_files/paper/2019/hash/90fd4f88f588ae64038134f1eeaa023f-Abstract.html>
- <https://papers.nips.cc/paper_files/paper/2021/hash/e53a0a2978c28872a4505bdb51db06dc-Abstract.html>

Do not try `alpha=0.5`, a late-only start, another `k`, momentum pullback, or a
conditional fallback. This proposal is one fixed point, not a parameter sweep.

## Effective Displacement and Learning-Rate Decision

The old EXP-015 Lookahead proposal did not account for the dominant first-order
effect. If a five-step fast path starts at slow state `s_j` and ends at `f_j`,
then the outer update is exactly

```text
s_(j+1) = s_j + alpha * (f_j - s_j).
```

With locally constant gradients and no momentum, the cycle therefore retains
only `alpha` times the five-step SGD chord. `alpha=0.5` discards half the chord;
`alpha=0.8` retains 80%. Momentum retention does not make this equality false
for the observed endpoint chord, but it makes the phrase "effective learning
rate is exactly alpha times LR" too strong: retained velocity, weight decay,
nonlinear gradients, CutMix, drop path, and SAM all change later fast paths.

Keep `PEAK_LR=0.2` and the full EXP-004 time-indexed schedule unchanged. Do
**not** compensate with `PEAK_LR / alpha = 0.25`. Such compensation only matches
a constant-gradient first-order chord; it raises every individual fast step by
25%, changes the points at which gradients and SAM perturbations are computed,
and creates a second unvalidated intervention. The same-LR decision follows the
original paper's comparison principle, whereas `0.25` has no local or cited
support for this WRN/CutMix/SAM package.

This leaves an explicit 20% outer-chord contraction. It is the intended
variance/stability tradeoff, not an implementation detail to hide. At the first
sync and first clean-tail sync, measure report-only FP32 norms of the pre-sync
chord, retained slow displacement, and discarded pullback. Verify the retained
ratio is `0.8` within a preregistered numerical tolerance. These observations
cannot select a new LR or alpha. A negative result may be caused by remaining
under-travel; it does not authorize an `alpha` or LR retry.

## Exact State and Update Semantics

After model initialization, construct the optimizer-owned trainable-parameter
inventory in stable named order. Allocate one `torch.empty_like` slow tensor per
parameter with preserved memory format and copy the live value into it before
the charged timer. Allocation and copying consume no RNG. Do not construct a
second model.

The slow inventory must exactly match the optimizer inventory in names, order,
shape, dtype, device, stride or preserved memory format, and element count.
Slow tensors have `requires_grad=False`, never receive gradients, never appear
in an optimizer parameter group, and never alias fast parameters, SAM
snapshots, evaluation backups, gradients, BatchNorm buffers, or optimizer
state.

On one-based completed optimizer update `t`, perform:

```python
optimizer.step()
if t % LOOKAHEAD_K == 0:
    torch._foreach_lerp_(slow_parameters, fast_parameters, LOOKAHEAD_ALPHA)
    torch._foreach_copy_(fast_parameters, slow_parameters)
```

Use prebuilt homogeneous foreach lists rather than per-parameter Python update
loops. All interpolation/copy work occurs before the parent's CUDA synchronize
and is charged to the 300-second training budget. There is no bias correction,
adaptive alpha, slow optimizer, decoupled decay, or uncharged synchronization.

Retain every inner `torch.optim.SGD` state exactly. In particular, momentum
buffers keep their values, storage identities, and ownership across a sync;
they are not reset, scaled by `alpha`, interpolated, copied to a slow buffer, or
replaced. The next Nesterov step acts from pulled-back parameters with the
accumulated fast-path velocity. Coupled weight decay remains applied by the
single inherited optimizer step. The parameter/velocity mismatch is canonical
retained-state Lookahead behavior and an acknowledged risk.

## SAM, BatchNorm, and RNG Ordering

Preserve EXP-004's exact scheduled-SAM sequence:

1. Save the CUDA RNG state at the inherited point and run the primary forward
   and backward on the current fast parameters; only this pass updates BN.
2. Snapshot and perturb only the fast parameters by normalized rho `0.05`.
3. Zero gradients, replay CUDA RNG, disable BN running-stat tracking, and run
   the second separately-autocast CE forward/backward.
4. In `finally`, restore BN flags and restore the exact unperturbed fast
   parameter snapshots.
5. Apply the sole Nesterov optimizer update.
6. If the completed one-based step is divisible by five, apply the Lookahead
   interpolation and copyback.

Thus a slow tensor never observes the temporary SAM perturbation. On a
Lookahead/SAM coincidence it interpolates toward the post-restoration,
post-Nesterov endpoint. There remains exactly one optimizer update and one BN
statistics update per batch.

In the clean tail, `SAM_PERIOD=2` and `LOOKAHEAD_K=5` have LCM 10. Syncs
alternate between odd ordinary updates and even SAM updates, so the tail counts
must differ by at most one. No sync may alter CPU RNG, global CUDA RNG, the
dedicated CutMix CPU generator, or the dedicated CutMix CUDA generator.

Lookahead owns parameters only. It must not read or write BN `running_mean`,
`running_var`, or `num_batches_tracked`. These buffers continue to summarize
the online fast path, while the evaluated parameters are the slow path. At a
normal epoch boundary `195 % 5 == 0`, fast and slow parameters coincide, making
this mismatch no worse than the inherent within-cycle path mismatch. At the
budget-truncated final epoch the slow parameters may lag by up to four updates
while BN buffers are current. No BN recalibration, buffer interpolation, extra
data pass, or alternate evaluation is allowed.

## Slow-Only Evaluation Semantics

Keep `EVAL_EVERY=1` and the frozen `Eval` implementation. Before every existing
evaluation call, outside the charged training timer:

1. copy current fast parameters into preallocated evaluation backups;
2. copy slow parameters into the same online `Parameter` objects;
3. invoke the frozen evaluator exactly once;
4. restore fast parameters in `finally`, even if evaluation raises.

The evaluator always sees current online buffers because only parameters are
swapped. The slow tensors and optimizer state are never swapped or overwritten.
On complete epochs the parameter copies are value-neutral because fast already
equals slow, but keep one code path so the partial final epoch cannot silently
change source. Evaluation and restoration must preserve optimizer parameter
identities, momentum-buffer identities/values, slow tensors, RNG states, and
the exact pre-evaluation fast model state.

`best_test_acc`, `final_test_acc`, and `final_test_loss` are computed solely from
these slow-parameter evaluations. Never run a fast evaluation, never choose
between sources, and never add EMA. This is canonical Lookahead model
evaluation, not checkpoint ensembling.

## Correctness and Integrity Audits

### Deterministic CPU smokes

1. Verify scalar and heterogeneous multi-tensor recurrences through steps
   1-11, with no sync on 1-4, exact 0.8 interpolation and copyback on 5 and 10,
   and the expected residual fast/slow difference on step 11.
2. Compare each sync against an independently computed FP64 recurrence, using
   fixed dtype-aware tolerances for production FP32 tensors.
3. Initialize distinct parameters, gradients, Nesterov momentum buffers,
   floating BN buffers, and integer BN counters; prove the sync changes only
   fast parameters and slow tensors.
4. Prove optimizer parameter identities and momentum-buffer values, storage
   pointers, and identities are unchanged by the sync itself.
5. Prove slow inventory coverage and non-aliasing against parameters, SAM
   snapshots, evaluation backups, buffers, and optimizer state.
6. Snapshot CPU, CUDA when available, and both CutMix generator states around
   slow construction, sync, successful evaluation swap, and exceptional
   evaluation swap; require exact RNG neutrality.
7. Force evaluator success and failure when fast differs from slow. Require
   exact restoration of fast parameters, unchanged slow parameters/buffers and
   optimizer state, one attempted source only, and balanced swap/restore counts.

### Physical-GPU-0 integration smoke

Use the production WRN, BF16 autocast, channels-last layout, real CIFAR batches,
CutMix generators, drop path, Nesterov state, SAM snapshots, and evaluator
guard. Run two production-faithful prefixes with replayed stochastic state:

- an early 10-step prefix exercising syncs 5 and 10, including CutMix and
  drop-path draws;
- a forced clean-tail 10-step prefix whose step IDs exercise ordinary sync 5
  and SAM sync 10, including perturbation, CUDA RNG replay, BN suppression,
  restoration, the optimizer update, and then Lookahead.

The parent and candidate must be identical through the pre-sync endpoint of
step 5. At each candidate sync require the FP32 reference recurrence, exact
fast/slow equality after copyback, unchanged BN buffers *across the sync
itself*, unchanged momentum and parameter identities, exact RNG parity, and
proof that the SAM snapshot was restored before interpolation. Future parent
and candidate weights need not match after the first sync; their data order,
augmentation decisions, schedule predicates, and RNG stream positions must.

Exercise a slow-only evaluation both at a synchronized endpoint and after one
unsynchronized fast step. Monkeypatch the evaluator to record the observed
parameter source without accessing test data, and separately force an
exception. Require exact fast restoration and zero state/RNG mutation.

### Production audit contract

Print a compact final Lookahead audit containing:

```text
k=5 alpha=0.8 momentum=retain state=parameters_only eval=slow_only
syncs=floor(num_steps/5) first_sync=5 last_sync=5*floor(num_steps/5)
tail_ordinary_syncs and tail_sam_syncs differ by at most 1
eval_swaps == eval_restores == num_epochs
inventory/alias/RNG/momentum/BN/order/restore failures == 0
```

Also report the number of fast steps since the last sync at final evaluation
(`num_steps % 5`) and the two preregistered displacement diagnostics. Avoid
per-sync `.item()`, full-model norms, or audit synchronizations. Counter updates
and foreach operations are charged; scalar formatting and final validation are
outside the charged timer.

## One Accuracy-Blind Timing Gate

After correctness passes, run one complete paired preflight on physical GPU 0,
confirmed as the approximately 97,871 MiB NVIDIA H20, with
`CUDA_VISIBLE_DEVICES=0`. Materialize exact parent EXP-004 from commit
`1a8d0de`. Use real CIFAR batches, BF16, channels-last, the production optimizer,
SAM and Lookahead state, identical batch and stochastic replay, and five rounds
with alternating parent/candidate order. Guard the evaluator to raise before it
can iterate test data or emit accuracy.

Each arm's timed mixture contains a 200-step production-faithful trace:

```text
early steps 1-150:       150 primary paths, 30 sync opportunities
tail steps 20663-20712:  25 ordinary + 25 SAM paths, 10 sync opportunities
                         (5 ordinary-sync + 5 SAM-sync)
```

Replay the same observed CutMix decisions and stochastic states in both arms.
The parent arm performs no Lookahead work; the candidate arm performs the exact
charged foreach recurrence. Time from the production `t0` through the same CUDA
synchronization boundary. Initialization and excluded evaluation swaps are not
charged. Measure candidate-only peak allocation after constructing model,
optimizer momentum, SAM snapshots, slow parameters, and evaluation backups.

The first complete numeric preflight is decisive. Require:

```text
parent round drift <= 0.03
median absolute deviation / median paired ratio <= 0.01
median candidate/parent weighted latency ratio <= 1.02
maximum round ratio <= 1.03
projected num_steps >= 25,000 from the EXP-004 dose of 25,560
projected total runtime < 600 seconds
candidate peak allocated memory < 2,048 MiB
all correctness and finite-state checks pass
```

Only an exception, missing output, or malformed harness assertion before any
numeric gate result permits repairing and repeating the harness. Once numeric
latency or dose is emitted, a failed gate ends this leaf before metric launch.
Do not change `k`, `alpha`, LR, momentum semantics, foreach implementation,
start time, evaluation source, or audits in response.

## One Metric Run and Decision Rule

After the accuracy-blind gate passes, reconfirm physical GPU 0, remove any stale
`run.log`, and launch exactly once:

```bash
timeout 600s env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 uv run train.py > run.log 2>&1
```

There is no seed repeat, metric retry, coefficient fallback, LR compensation,
fast/slow source fallback, or early stopping based on intermediate test output.
Preserve the raw log until durable transcription and adversarial result review
are complete.

Formal improvement over parent EXP-004 (`95.40%`) requires:

```text
best_test_acc >= 95.50%   # at least 9,550 / 10,000 correct
```

Validity also requires exit 0, approximately 300 charged training seconds,
total runtime below 600 seconds, physical GPU 0, fixed seed 42, unchanged
2,748,890-parameter architecture, only `train.py` modified, one slow-only
evaluation per epoch, a complete summary, at least 25,000 optimizer updates,
the inherited CutMix/SAM dose and parity audits, exact Lookahead sync arithmetic,
balanced evaluation restoration, and zero state-integrity failures.

For interpretation, report the last eight slow-evaluation accuracies, their
mean/range/final value, and `best - last8_mean`; do not turn these report-only
statistics into a second acceptance criterion. A formal pass with a depressed
late sequence is weak evidence for a stable Lookahead benefit. A result below
95.50 is one no-improvement result for this exact high-alpha package; do not try
`alpha=0.5`, `alpha=1`, LR `0.25`, late-only Lookahead, or fast evaluation in
the same experiment.

## Evidence, Expected Value, and Falsification

EXP-004 provides a clean parent: `95.40%` best and final accuracy, 25,560
optimizer steps, 2,449 late SAM updates, and no evaluation averaging. That is a
better causal base for Lookahead than EXP-011 because the evaluated quantity is
not already an ESS-79 full-state EMA. Lookahead can now change future gradients
and the basin reached, rather than merely adding a second smoothing kernel to
an already averaged evaluator.

The evidence remains limited. The cited papers support the method and the
`k=5, alpha=0.8` CIFAR operating point, but not this time-budgeted WRN,
time-indexed cosine schedule, CutMix, stochastic depth, or late period-two SAM.
The old reviewer correctly identified chord contraction as a serious issue;
high alpha mitigates rather than removes it. It also correctly identified
kernel-launch overhead, which the foreach implementation and strict paired gate
must measure instead of assuming away. Parameter-only slow weights retain a BN
path mismatch, and retained momentum can overshoot after pullback.

The positive hypothesis is that modest five-step trajectory contraction reduces
harmful Nesterov endpoint variance while preserving 80% of useful travel, and
that late SAM then explores from a more stable sequence of basin anchors. The
counter-hypothesis is stronger than for a representation intervention: EXP-004
does not show unstable late checkpoints, and the contraction may simply reduce
effective progress or blunt SAM's useful motion.

The accuracy hypothesis is falsified by `best_test_acc <95.50%`. The experiment
is invalid, rather than merely negative, if it violates scope, GPU, budget,
evaluation count/source, state ownership, sync/SAM ordering, RNG neutrality,
restoration, or bounded execution. A valid miss narrows only this fixed
same-LR, `k=5, alpha=0.8`, retained-momentum, parameter-only package; it does not
prove that all Lookahead variants fail.

## Effort and Risk

**Estimated effort: medium.** The recurrence is small, but the slow-only
exception-safe evaluator, optimizer/SAM ordering, state ownership, and paired
timing proof require careful implementation.

**Risk: medium for implementation and high for scientific impact.** The method
should fit comfortably in memory and likely in time with foreach operations.
The dominant risk is not code but insufficient or negative effect: the local
threshold is attainable, yet there is no direct diagnosis that Lookahead's
online variance reduction is the missing generalization mechanism.
