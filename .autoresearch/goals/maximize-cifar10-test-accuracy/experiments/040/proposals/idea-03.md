# Proposal: One-Time Nesterov Reset at the First Hard-Label Step

## Claim and Scope

The accepted learner changes its target distribution abruptly at 65% counted
time: batch-shared mixup stops and the first hard-label update begins, while
the global cosine, coupled weight decay, and Nesterov state continue. Test the
narrow claim that velocity accumulated under mixed images and paired soft
targets is harmful immediately after this target switch. Clear every live SGD
momentum buffer exactly once before the first hard-label forward/backward, with
the accepted global cosine fully restored after EXP039's normal-exposure miss.

This is an optimizer-state intervention, not a momentum-coefficient change or
a two-phase restart. Preserve accepted `a7c42dc` model bytes, pooled head,
optimizer construction and groups, `MOMENTUM=0.9`, `nesterov=True`,
`dampening=0`, matrix-only coupled `WEIGHT_DECAY=5e-4`, LR function and floor,
data/RNG, temporal thresholds, evaluator, seed, and budget. Do not rephase LR,
change momentum, reset parameters or BN running state, shift either cutoff, or
repeat the reset.

## Exact Implementation and Ordering

Put the reset only inside the existing one-way
`mixup_enabled and not use_mixup` transition. The production order for the
crossing step must be:

1. start the counted step and transfer the accepted batch;
2. compute and write the accepted global-cosine LR from the pre-step counted
   time;
3. compute `progress` and `use_mixup`; on the first observed
   `progress >= MIXUP_END_FRACTION`, set `mixup_enabled = False`;
4. traverse both existing optimizer groups in their existing order, require a
   live `momentum_buffer` for every parameter, and zero each buffer in place;
5. report the reset count in the existing transition message;
6. run accepted `zero_grad(set_to_none=True)`, the hard-label forward and CE,
   backward, and one accepted Nesterov step.

The minimal body is:

```python
momentum_buffers_reset = 0
expected_buffers = sum(len(group["params"]) for group in optimizer.param_groups)
for group in optimizer.param_groups:
    for parameter in group["params"]:
        buffer = optimizer.state.get(parameter, {}).get("momentum_buffer")
        if buffer is None:
            raise RuntimeError("Missing momentum buffer at hard-label transition")
        buffer.zero_()
        momentum_buffers_reset += 1
if momentum_buffers_reset != expected_buffers:
    raise RuntimeError("Incomplete momentum-buffer reset")
```

The accepted model has 52 trainable parameter tensors and 1,003,482 trainable
FP32 elements, so the crossing must report exactly 52 buffers and zero about
3.83 MiB of state. Use `optimizer.state.get` rather than indexing the
defaultdict so the audit cannot create missing state. Do not reconstruct the
optimizer or delete buffer keys: in-place zeroing preserves groups, defaults,
state identity, and makes the intended transition directly inspectable.

The reset occurs before the hard forward rather than after it so the complete
first hard update is defined from zero inherited state. It consumes no RNG and
does not touch gradients or parameter bytes. It is aligned to the target
transition, not the later image-policy transition: the current exhausted
iterator still supplies RandAugmented images until its normal boundary.

## What Nesterov Actually Changes

Let `b_0` be a parameter's live buffer just before the first hard step, `g_k`
the accepted current gradient including coupled decay where applicable, and
`mu=0.9`. With PyTorch SGD's zero dampening, the first no-reset update uses

```text
b_1 = 0.9 b_0 + g_1
d_1 = g_1 + 0.9 b_1 = 1.9 g_1 + 0.81 b_0.
```

Because the candidate retains the buffer tensor but zeros it first, it uses
`b_1=g_1` and `d_1=1.9 g_1`. Thus it removes exactly `0.81 b_0` from the first
Nesterov direction without suppressing the current hard-label gradient. Under
the counterfactual of identical subsequent gradients, the inherited term in
hard update `k` is `0.9^(k+1) b_0`; real trajectories diverge after the first
step, so this decomposition is an oracle, not a claim of long-run equality.

The memory is short and precisely bounded:

- half-life: `ln(0.5)/ln(0.9) = 6.58` updates, about 7 steps;
- below 10%: 22 updates; below 5%: 29 updates; below 1%: 44 updates;
- at the accepted EXP036 cadence, 44 batches are 11,264 examples, 0.2253 data
  passes, and roughly 0.5 counted seconds;
- EXP036 had 9,114 hard-label steps, so 44 steps are only 0.483% of the hard
  tail; its RandAugment iterator also exhausted exactly 44 steps after the
  mixup crossing, meaning nearly the entire directly removed-memory window was
  the staggered hard-label-plus-RandAugment phase.

At nearly constant boundary LR `~0.06123`, the infinite-horizon sum of the
removed old-buffer coefficients is `mu^2/(1-mu)=8.1`; 44 steps capture 8.02 of
that coefficient, or an approximate cumulative displacement of `0.491 b_0`
before accounting for LR drift and trajectory divergence. This can perturb the
later basin through path dependence, but it is not sustained tail adaptation.
The buffer also contains coupled-decay history for matrix parameters, so a
success cannot be attributed purely to mixed-target data gradients.

## Expected Ceiling and Risks

The honest mechanistic ceiling is 44 directly material updates, 0.225 passes,
and under 0.5% of the hard tail. No defensible top-1 ceiling follows from those
counts because a transient can redirect all later updates, but local evidence
supports a low-upside prior: EXP039's much stronger intervention changed tail
LR area by 39.46% over roughly 9,000 steps and still lost 0.50 points, while the
accepted run already finishes within 0.03 points of its best. The reset must
flip at least 10 additional examples on the 10,000-image test set merely to
reach the required +0.10-point margin. Treat an acceptance-sized gain as the
plausible ceiling, not a reason to expect a large jump; any improvement larger
than that would require nonlinear path amplification rather than persistent
optimizer-state correction.

The strongest contrary hypothesis is that the recent mixed-target velocity is
useful feature-learning signal. Clearing all 52 buffers also resets backbone,
pooled head, classifier, BN affine, and biases indiscriminately. The first hard
gradient may already align with `b_0`, so reset can reduce productive motion at
LR 0.061 rather than eliminate conflict. Conversely, a measured whole-model
buffer/gradient cosine can hide damaging minority tensors; it is diagnostic
only and must not motivate a post-result selective reset. The one fixed seed
cannot estimate average treatment effect.

## Semantic and Systems Gates

Use an evaluator-free harness with an independent `git show a7c42dc:train.py`
oracle. Before timing or scoring require:

- production diff is confined to the existing mixup-transition block;
  `prepare.py` and every other production file are byte-identical;
- topology, all initial parameter/buffer bytes, 1,003,482 parameters,
  construction CPU/CUDA RNG, optimizer groups/options/empty initial state,
  transforms, accepted `learning_rate()`, losses, thresholds, and cadence are
  exact;
- cloned early steps and the last synthetic step below 65% are bitwise equal
  end to end, including mixup draw/permutation, logits, loss, gradients,
  parameter/BN/optimizer state, and terminal RNG;
- seed all 52 accepted/candidate buffers with distinct nonzero deterministic
  tensors, then cross the boundary: immediately before reset, require exact
  model, buffer, input, and RNG equality; immediately after, require all and
  only candidate momentum buffers bitwise zero, accepted buffers unchanged,
  exact 52/52 coverage, unchanged parameter/BN/gradient bytes, and unchanged
  CPU/CUDA RNG;
- independently reproduce first and second hard Nesterov updates. The first
  candidate oracle must use `1.9*g_1`; the accepted oracle must include
  `0.81*b_0`. On the second step verify buffer recurrence from each arm, finite
  state, and deterministic replay from restored snapshots;
- boundary probes below, at, and above 65% prove one reset only, before the
  first hard forward/backward, using the accepted LR computed for that exact
  pre-step time. Prove no reset at the later exhausted-iterator RandAugment
  transition and preserve its ordering;
- print per-group buffer counts/elements, aggregate and per-group `||b_0||`,
  `||g_h||`, cosine `b_0` versus `g_h`, predicted first-step delta, and maximum
  oracle/replay errors before assertions. These are interpretation diagnostics
  only and cannot select tensor subsets, reset strength, or whether to score.

Steady-state candidate work is source-identical to accepted outside one
crossing. Do not benchmark a reset on every timing step. Measure the isolated
one-time reset latency over restored 52-buffer snapshots with CUDA events and
synchronization, printing all counterbalanced windows, then verify ordinary
early and hard complete-step timing remains stable. Require finite timings,
population CV at most 5%, candidate peak below 2,048 MiB, and account for the
one-time median reset cost in exposure:

```text
projected_passes = 130.304 * (300 - reset_seconds) / 300
```

Require projected passes at least 127. The expected 3.83 MiB clear should make
this gate trivial, but the 52 launches and its inclusion inside counted `dt`
must be measured rather than assumed. A stable systems miss closes this
implementation only and must not be rescued by clearing fewer buffers.

## Sole Score and Falsifiable Closure

After gates pass, reconfirm baseline 94.48% at `a7c42dc`, accepted source and
global cosine, one idle local H20, local CIFAR-10, frozen evaluator, no stale
log, and exact diff. Run exactly once at seed 42:

```bash
timeout 600s uv run train.py > run.log 2>&1
```

Require exit zero, one finite complete summary, 300.0-300.1 counted seconds,
wall below 600, 1,003,482 parameters, exactly one transition reporting 52
buffers at the first pre-step progress at or above 65%, the later
exhausted-iterator RandAugment transition, unique every-fifth-epoch evaluations
plus final partial epoch, and no traceback, OOM, worker, non-finite, or source
violation. Record realized passes as `num_steps*256/50000`.

Success is only `best_test_acc >=94.58%`. Realized exposure at least 127,
`final_test_acc >=94.45%`, and `final_test_loss <=0.2456` corroborate a
normal-exposure, stable-boundary interpretation but neither endpoint metric can
rescue or veto the primary verdict.

A valid >=127-pass miss falsifies the narrow claim that fully clearing inherited
whole-model Nesterov state at the first hard-label step improves this accepted
trajectory. Retain the accepted uninterrupted buffers and close immediate
rescues: partial reset factors, tensor/layer-specific clears, deleting buffers,
optimizer reconstruction, reset at the RandAugment boundary, shifted or
repeated resets, momentum tapering, and combination with EXP039's failed LR
rephase. Two isolated misses do not justify combining them. The result does not
close a separately diagnosed full-run momentum coefficient, another optimizer,
or unrelated classifier/loss geometry. A completed run below 127 passes remains
the sole valid score but closes only this exact implementation because exposure
confounds mechanism attribution; a pre-score semantic/timing failure supplies
no accuracy evidence.

## Falsifiable Hypothesis

If inherited Nesterov velocity from the accepted mixed-target phase impedes the
first hard-label updates enough to redirect later refinement, then one complete
52-buffer reset immediately before the first hard-label forward/backward will
retain at least 127 projected and realized passes and raise fixed-seed
`best_test_acc` from 94.48% to at least 94.58%, with final accuracy at least
94.45% and final evaluator loss at most 0.2456 as corroboration. A
normal-exposure miss closes this isolated transition-state program without
reopening the now-bracketed hard-tail LR curve.

Local evidence: `experiments/036/03-execute.md`,
`experiments/036/04-analysis.md`, `experiments/039/04-analysis.md`,
`experiments/039/proposals/idea-01.md`, `02-system-understanding.md`, and
`03-experiment-learnings.md`.
