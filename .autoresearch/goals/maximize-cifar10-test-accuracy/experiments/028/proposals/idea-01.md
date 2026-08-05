# Proposal: Freeze Stem and Stage 1 After the Exhausted 65% Boundary

## Thesis

Train the complete accepted `(2,2,3)` WRN normally through the early
mixup-plus-RandAugment phase, then stop computing parameter gradients for the
stem and both stage-1 blocks at the already established exhausted epoch boundary
after 65% counted time. Keep their complete forward functions and BatchNorm
running-statistic updates active. Spend the saved high-resolution backward time
on more hard-label updates to stage 2, stage 3, final BN, and the classifier.

The testable causal claim is narrow: high-resolution features are essential to
the representation, but their weights may be sufficiently established by about
196-197 counted seconds that late gradients are less valuable than additional
low-resolution optimizer decisions. This directly targets the measured 74%
backward bottleneck without changing the accepted early trajectory or deleting
any feature transformation.

## Exact Intervention

Preserve the accepted production model, data path, and training policy from
commit `67c8e98`:

- stage blocks `(2,2,3)`, widths `[32,64,128]`, and 987,098 parameters;
- RNG-isolated `RandAugment(num_ops=1, magnitude=5)` through the first exhausted
  epoch ending at or after 65% counted time;
- batch-shared alpha-0.2 mixup until exactly 65% counted time, followed by hard
  labels;
- batch 256, FP32 SGD/Nesterov, matrix decay `5e-4`, LR `0.2 -> 0.002` on the
  accepted time-based warmup/cosine, seed 42, loader, and evaluator cadence.

At the same exhausted epoch boundary where production sets the shared
RandAugment flag to zero, and only after the current DataLoader iterator has
fully ended:

1. call `optimizer.zero_grad(set_to_none=True)` to remove the just-completed
   step's gradients;
2. set `requires_grad_(False)` on every parameter in `model.conv1` and
   `model.layer1` exactly once;
3. leave those parameters in their original optimizer groups with their
   momentum buffers intact but dormant;
4. never unfreeze them during the run.

The frozen set is exact: 432 stem-convolution weights plus 32,992 stage-1
parameters, for 33,424 frozen parameters and 953,674 still trainable parameters.
Freeze both convolution weights and all stage-1 BatchNorm affine weights and
biases. Do not freeze stage 2, stage 3, final `model.bn`, or `model.fc`. Do not
rebuild optimizer groups, delete momentum state, move modules, detach tensors
manually, wrap the prefix in `no_grad`, or change any forward method.

Keep `model.train()` on the whole network. Stage-1 BatchNorm `running_mean`,
`running_var`, and `num_batches_tracked` are buffers, not frozen parameters, so
they must continue updating from each hard-label training batch. Evaluation
continues to use those live running statistics through the unchanged evaluator.
Because inputs and every prefix parameter no longer require gradients, ordinary
autograd naturally returns a non-grad stage-1 output and omits the prefix
backward graph while preserving its exact training-mode forward computation.

Log one transition line with epoch, step, counted seconds, percent, frozen and
remaining parameter counts, and `iterator_exhausted=true`. Require the freeze
and RandAugment-disable events to occur at the same boundary. Mixup still ends
per batch at 195.0 seconds, so the accepted short lag of at most the remainder
of one epoch is preserved; in EXP-027 the exhausted boundary was 196.7 seconds,
148 steps after mixup ended.

## Expected Timing and Exposure

The system probe attributes 73.7-74.2% of an accepted step to backward. The stem
and stage 1 account for 41.5% of measured forward time (1.9% + 39.6%). If their
share of backward cost is roughly proportional, freezing the prefix removes
about 4.2 ms from a 13.77 ms isolated hard-label step, giving an approximate
9.5 ms frozen-tail step and 1.44x tail throughput. Applying that ratio only to
the final 34.4% after the historical exhausted boundary projects:

`133.00736 * (0.656 + 0.344 * 1.44) ~= 153.2 passes`.

This estimate is deliberately non-authoritative: convolution backward shares
need not follow forward shares, and isolated CUDA timings differ from scored
means. A reasonable expected range is 145-154 passes. The matched production-
path H20 timing gate below decides feasibility. Forward FLOPs, total parameters,
the first 65% step time, loader wall time, and evaluation cost remain accepted;
VRAM may fall slightly because prefix activations no longer need backward
retention, but memory is not the objective.

The additional work is not a neutral replay. The time-based LR gives every
extra tail batch the LR associated with its wall-time position, so upper layers
receive more low-LR optimizer decisions and see more crop/flip examples. Frozen
prefix weights receive neither gradients, momentum updates, nor weight decay
after the boundary. That combined temporal compute allocation is the treatment.

## Semantic Preflight

Use an ignored evaluator-blocked harness and the exact production helper before
the scored run. It must establish all of the following:

1. Candidate construction has the exact accepted topology, 987,098 parameters,
   state dict, optimizer groups/state, initial logits, CPU/CUDA RNG, and data
   policy. Merely defining the freeze helper must consume no RNG or change early
   computation.
2. Run accepted and candidate copies on identical synthetic minibatches and
   private RNG streams through a shortened boundary. Require bitwise-equal model
   state, optimizer state, BN buffers, logits, losses, gradients, updates, and
   RNG immediately before the boundary.
3. Freeze only after a simulated exhausted iterator and after clearing gradients.
   Require exactly 33,424 parameters to change from trainable to frozen, exactly
   953,674 to remain trainable, no parameter to change value, and original
   optimizer group order/membership plus all momentum buffers to remain intact.
4. On the first matched hard-label batch after freezing, require candidate and
   accepted pre-update logits and loss to match. Candidate `conv1` and `layer1`
   parameters must have `grad is None`; their output must not require grad; all
   eligible upper parameters must have finite gradients and update normally.
5. Require every frozen parameter and its momentum buffer to remain bitwise
   unchanged across multiple candidate tail steps. Require no weight decay or
   stale-gradient update on the frozen set.
6. Keep both copies in training mode and require every stage-1 BN running mean,
   variance, and batch counter to update. On the first matched batch these
   buffers must equal the unfrozen accepted reference, proving the candidate did
   not accidentally call `.eval()` or freeze buffers.
7. Verify exactly one ordered mixup transition, exhausted RandAugment transition,
   and freeze transition; freeze can occur only at the latter boundary and can
   never be reversed. Test/evaluator data must never be constructed or read.

Abort before timing on any semantic mismatch. In particular, do not rescue a
failed gradient check by detaching the prefix output, rebuilding the optimizer,
leaving BN affine trainable, freezing BN statistics, or moving the cutoff.

## Throughput and Exposure Gate

Benchmark the complete hard-label production timed region for accepted and
frozen-tail models on the one local H20: pinned-host transfer, LR assignment,
zeroing, forward, loss, backward, SGD, and synchronization. Use identical saved
boundary state and batches, private RNG streams, at least 25 warmup steps, and
three balanced windows of at least 50 steps in alternating accepted/candidate
order. Both paths remain `model.train()` so BN behavior and kernels match
production.

Report every window, mean/median hard-step milliseconds, CV, and tail throughput
ratio. Project whole-run exposure conservatively with:

`projected_passes = 133.00736 * (0.656 + 0.344 * accepted_hard_ms / frozen_hard_ms)`.

Proceed to scoring only if all timing CVs are at most 5%, frozen-tail throughput
is at least 1.20x accepted, projected exposure is at least 145.0 passes, the
candidate has finite loss/gradients, and no path errors or OOMs. These gates
require a material compute reallocation while staying below the analytic upper
estimate. If any gate fails, classify the intervention as preflight-infeasible
and do not lower a threshold, narrow the frozen set, move the boundary, or score
a fallback.

Loader timing need not be repeated because the transform, persistent workers,
prefetch, and exhausted cutoff are unchanged from EXP-027; confirm those source
paths are untouched. The historical 345.3-second wall time leaves ample margin
under 600 seconds, and a faster counted tail should not increase excluded loader
or evaluation time enough to threaten it.

## Hypothesis and Decision Rule

The current accepted baseline is 94.32% at `67c8e98`; therefore the formal
success threshold is **94.42% `best_test_acc`**. The hypothesis is that preserving
the fully learned high-resolution forward representation and its live BN
statistics while reallocating late backward time will reach at least 94.42% and
at least 145 realized data passes within the same 300 counted seconds.

After all preflights pass, launch exactly one fixed-seed scored command under the
600-second timeout. Accept only if it exits zero with a complete finite summary,
one H20, 300 counted seconds, total wall time below 600 seconds, exact topology
and 987,098 parameters, one valid exhausted freeze event, unique evaluations no
more than once per epoch, at least 145 realized passes, and
`best_test_acc >= 94.42%`. Report final accuracy/loss and the best-final gap, but
they cannot rescue a sub-threshold primary metric. Never rerun a valid score.

A valid score below 94.42 closes this exact whole-prefix late freeze. If exposure
passes but accuracy or loss regresses, conclude that late adaptation of early
filters/affines is more valuable than the extra upper-layer updates. If the run
misses 145 passes despite a passed timing gate, inspect attribution but retain
the preregistered no-improvement verdict; do not tune the freeze boundary or
subset from the result.

## Evidence and Distinction From EXP-016

EXP-027 provides the 94.32% accepted interaction and exact exhausted boundary.
The current system profile identifies backward as 74% of step time and stage 1
as the dominant high-resolution stage. The Time Matters distillation supports
temporal specialization after an early critical period, while the accepted
mixup/RandAugment schedule supplies a natural predeclared handoff.

EXP-016 is contrary evidence only to deleting early representation capacity. It
removed `layer1[1]` before training, added a newly seeded stage-3 block, changed
the topology from `(2,2,2)` to `(1,2,3)`, and trained the altered graph for the
entire run. Despite 171.7 passes, it scored 93.82, proving that the second
high-resolution block's forward transformation and early learning are
essential.

This proposal retains both stage-1 blocks, the stem, their exact accepted
initialization, all of their first-65% learning, and every one of their forward
calls in the tail. It adds or removes no module and keeps the accepted third
stage-3 block. Only 33,424 parameter gradients cease after the data/loss phase
handoff. Thus a negative result answers whether late prefix adaptation is
necessary; it does not repeat the EXP-016 claim that the prefix block itself is
dispensable.

## Risks

- The distribution changes precisely at the boundary: RandAugment and mixup end,
  so early filters may need late gradients to adapt to clean hard-label inputs.
- Frozen stage-1 affine parameters may not compensate for continued BN-statistic
  drift. Conversely freezing BN statistics would introduce a different, less
  evaluator-consistent treatment and is prohibited.
- Stopping prefix weight decay and momentum updates is part of freezing. EXP-007
  showed that broad late decay removal is harmful, raising the prior that static
  early weights may regress.
- More tail batches can overfit or over-optimize the upper network even under the
  LR floor; extra exposure has repeatedly failed to guarantee top-1 gains.
- Faster epochs produce more legal evaluations under the unchanged cadence,
  increasing opportunities for a best-only spike. Preserve cadence, report the
  final endpoint, and judge formal success only by the preregistered threshold.

Implementation effort is low-to-medium, but causal risk is high. The experiment
is worthwhile only because it converts a directly measured backward bottleneck
into a tightly controlled temporal intervention with a fail-closed speed gate.
