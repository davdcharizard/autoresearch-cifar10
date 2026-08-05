# Proposal: Worker-Safe Early-Only One-Operation RandAugment

## Thesis

Add one standard torchvision RandAugment operation during the early training
phase, then remove it for a genuinely RandAugment-free hard-label tail. Preserve
the accepted WRN-16-2, batch-shared alpha-0.2 mixup cutoff, optimizer, cosine
schedule, seed, crop/flip pipeline, and evaluator. The intervention tests an
orthogonal source of image-space invariance after capacity, exposure, attention,
and optimizer refinements plateaued below the required margin.

This is not the previously proposed always-on treatment. The local reviews
identified always-on RandAugment as conflicting with the useful late refinement
phase. Here the worker policy changes only after a complete epoch iterator has
been exhausted, so persistent-worker prefetch cannot leak augmented samples into
the terminal phase.

## Exact Image Policy

Use the installed torchvision 0.24.1 implementation with exactly:

```python
transforms.RandAugment(
    num_ops=1,
    magnitude=5,
    num_magnitude_bins=31,
    interpolation=transforms.InterpolationMode.BILINEAR,
    fill=[125, 123, 114],
)
```

Place it after the accepted 32x32 random crop and horizontal flip and before
`ToTensor` and normalization. Bilinear interpolation avoids nearest-neighbor
artifacts for geometric operations. The fill is the rounded CIFAR-10 mean in
uint8 space, so exposed borders are approximately zero after mean subtraction.

`N=1, M=5` is a fixed standard policy, not a claim that every sampled operation
is weak. In the installed 14-operation space, AutoContrast and Equalize are
magnitude-independent, while Identity is a no-op. Do not filter the operation
space, add a probability wrapper, add Cutout, change mixup, or tune `N`, `M`,
fill, interpolation, or duration after observing the result.

## Exact Temporal Policy

Define `RANDAUGMENT_END_FRACTION = 0.65`. RandAugment is active from startup
through the entire epoch in which counted training time first reaches 195
seconds. At the end of that epoch, after the `for ... in train_loader` iterator
has been completely exhausted, set its shared active flag to false. Only then
may the next epoch iterator be created. RandAugment remains off for every later
training sample.

The accepted mixup behavior remains byte-for-byte in policy: it switches per
batch when `total_training_time / TIME_BUDGET_S >= 0.65`. Consequently, if the
threshold is crossed mid-epoch, there is an intentional and bounded interval of
hard-label batches that still use RandAugment. The lag is less than the
remainder of one 195-batch epoch, roughly under 0.7% of the 300-second budget at
accepted speed. Every subsequent epoch is both RandAugment-free and hard-label.
Here "clean tail" means the accepted crop/flip distribution with hard targets;
crop and flip remain active throughout, as in the accepted run.

## Persistent-Worker Implementation

Create one shared one-byte flag in the parent process using the active
`forkserver` multiprocessing context, for example an unlocked shared `c_byte`
initialized to one. A top-level, picklable transform wrapper owns that flag and
the fixed RandAugment object:

```python
class EarlyRandAugment:
    def __init__(self, active):
        self.active = active
        self.transform = transforms.RandAugment(...fixed arguments...)

    def __call__(self, image):
        return self.transform(image) if self.active.value else image
```

The production loader remains one shuffled loader with eight persistent workers,
batch 256, pinning, and `drop_last=True`. Keep the default prefetch depth
explicitly at `prefetch_factor=2` for auditability. Do not mutate
`train_set.transform` in the parent, because persistent workers own copied
dataset state. Do not rebuild or alternate loaders; that would add worker
startup, sampler, and base-seed changes.

The no-leak invariant follows from iterator ownership: PyTorch does not enqueue
the next epoch until the parent requests the next iterator. Exhausting the
current iterator consumes all tasks from its bounded prefetch queue. The parent
then flips the shared byte before the next `iter(train_loader)` reset can enqueue
new work, so no batch produced under the old policy can appear in the next
epoch. Never flip the flag while an iterator is live.

Log one transition line containing epoch, step, counted seconds, fraction, and
the statement that the prior iterator was exhausted. Do not add per-sample
counters, marker tensors, locks, or diagnostic reductions to the scored path.

## Required Semantic and Cutoff Preflight

Before scoring, run an evaluator-free local preflight against the exact
production wrapper and multiprocessing context:

1. Confirm torchvision 0.24.1 exposes the expected 14-operation RandAugment
   space and that the fixed constructor arguments and placement match this
   proposal.
2. Seed fresh accepted and candidate constructions with 42. Verify transform and
   shared-state construction consume no torch CPU or CUDA RNG and that model
   state, optimizer grouping, parameter count (691,674), and initial logits are
   exact accepted matches. The model and optimizer must remain unchanged.
3. Use a marker-only training dataset with the production worker count,
   persistent workers, and prefetch depth. Consume one whole active epoch,
   exhaust its iterator, flip the same kind of shared flag, and assert every
   marker in the next full epoch reports inactive. Also assert the parent never
   flips the flag while an iterator is live.
4. On real training data only, compare fixed-seed accepted and candidate label
   order for matched batches. Labels and sampler order must match. Candidate
   pixels should differ while active and use the accepted transform distribution
   after cutoff. Do not inspect test data or evaluator output.
5. Verify there is exactly one production transition, it can occur only at an
   epoch boundary after 195 counted seconds, no later path can re-enable it, and
   evaluation remains at most once per epoch.

RandAugment's random operation and sign draws occur in the worker CPU torch
streams. They therefore advance those worker streams and change later crop/flip
realizations, including after RandAugment is disabled. That stochastic stream
shift is part of the fixed augmentation treatment, not a seed reroll. It does
not consume the parent CPU stream, CUDA mixup/Beta stream, model initialization
stream, or shuffle sampler stream. Preserve `torch.manual_seed(42)`,
`torch.cuda.manual_seed(42)`, all loader settings, and the existing main-process
shuffle behavior.

## Required Wall-Time Preflight

CPU transform work occurs before the scored timer starts, so it may leave
`training_seconds` and optimizer exposure unchanged while increasing total wall
time. Measure operational feasibility before the single score:

- Benchmark accepted and candidate real-data loaders with the production batch,
  eight persistent workers, pinning, drop-last behavior, and prefetch depth.
- Use balanced `A-B-B-A` order, fresh workers per arm, one complete warmup epoch,
  and at least three complete measured epochs per arm. Pace consumption at the
  accepted approximately 10.8 ms GPU-consumer interval so prefetch overlap and
  starvation are represented.
- Time each whole epoch from iterator creation through the final yielded batch;
  do not infer feasibility from instantaneous `next()` latency on a prefilled
  queue. Report each epoch, median, coefficient of variation, and any worker
  error.
- Separately time the active-to-inactive boundary plus the first inactive epoch
  and require no restart or transition stall beyond normal epoch variance.

Project total wall time conservatively as:

`341.2s + max(0, candidate_epoch_median - accepted_epoch_median) * 143`.

Proceed only if both arms have CV at most 5%, all tensors are correctly shaped
and finite, workers exit cleanly, and projected total wall time is at most 500
seconds. This preserves 100 seconds of hard-timeout margin. Otherwise classify
the CPU policy as preflight-infeasible and do not score, weaken, move, or retry
it. Because data loading is outside the counted timer, no optimizer-pass floor
is needed beyond requiring the scored run to remain in the accepted normal
exposure regime; report realized steps and passes for attribution.

## Hypothesis and One-Run Decision Rule

The hypothesis is that one early image-space transformation teaches useful
photometric and geometric invariances that crop/flip and convex mixup omit,
while the RandAugment-free hard-label tail avoids the additive-regularization
failure of an always-on policy. The preregistered prediction is
`best_test_acc >= 94.17%`, at least +0.10 percentage points over the accepted
94.07%.

After both preflights pass, run exactly one fixed-seed scored command under the
required 600-second timeout. Accept only if the summary is complete, the run
uses one H20, counted training reaches 300 seconds, total wall time is below 600
seconds, evaluation occurs no more than once per epoch, the single cutoff is
valid, and `best_test_acc >= 94.17%`. Lower loss, normal exposure, or a favorable
intermediate evaluation cannot rescue a sub-threshold score.

A valid result below 94.17% closes this exact `N=1, M=5`, early-through-first-
boundary policy. Do not respond with a stronger policy, a different cutoff, an
operation filter, another seed, or a rerun. Interpret a normal-exposure accuracy
regression as evidence that even temporally bounded image augmentation compounds
the accepted mixup unfavorably. Interpret a timeout despite a passed preflight
as operational failure, not an accuracy verdict. Report best/final accuracy,
final loss, steps, passes, epochs, total wall time, and the exact transition
point so the mechanism remains auditable.

## Evidence and Risk

The saved RandAugment note supports the compact one-operation/shared-magnitude
family, and the saved Time Matters note supports removing augmentation after an
early critical period. EXP-002 validates early regularization followed by hard-
label refinement, while EXP-003, EXP-005, and EXP-006 make additive
regularization the principal accuracy risk. EXP-009's proposal and reviews
identify CPU wall time, magnitude-independent operations, and full-tail exposure
as the unresolved defects; EXP-018 specifically recommends stopping RandAugment
with mixup and requiring a worker-throughput preflight. EXP-025 and EXP-025's
analysis reserve early-only mild RandAugment as the broader orthogonal next
lever after the attention family closed.

Estimated implementation effort is medium and accuracy risk is high. The shared
flag and epoch-exhaustion protocol make the temporal treatment feasible and
exact, but they cannot make the underlying regularization prior favorable. This
is therefore a disciplined one-shot test, not the start of a RandAugment sweep.
