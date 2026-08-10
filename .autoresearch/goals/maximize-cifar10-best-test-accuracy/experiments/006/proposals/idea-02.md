# Proposal: RandAugment Magnitude 8 on the Accepted Phase Recipe

## Summary

Increase only the accepted EXP-004 RandAugment magnitude from 7 to 8. Preserve
`num_ops=1`, the strong-to-weak loader switch at exactly 80% of counted training
time, the 80% learning-rate hold, hard-label loss, model, optimizer, worker
lifecycle, seed, and evaluation schedule byte-for-byte.

EXP-004 established that one-operation magnitude-7 RandAugment during the high-LR
plateau, followed by crop/flip training during the low-LR tail, raises
`best_test_acc` from `91.83%` to `92.30%` while retaining 99.3% of the accepted
optimizer exposure. EXP-005 then showed that switching to weak augmentation at
75% regresses to `92.12%`, so the 80% phase boundary should no longer be moved.
The narrow remaining question is whether the accepted representation-learning
phase benefits from one additional RandAugment magnitude bin.

## Exact Change

In the existing `strong_train_tf` definition in `train.py`, replace exactly:

```python
transforms.RandAugment(num_ops=1, magnitude=7),
```

with:

```python
transforms.RandAugment(num_ops=1, magnitude=8),
```

There must be no other training-code change. In particular, retain:

- `num_ops=1` and torchvision's existing 31-bin operation space, default nearest
  interpolation, and default fill;
- the exact transform order: crop, horizontal flip, RandAugment, tensor
  conversion, normalization;
- `randaugment_enabled = True` initially and the break/worker replacement when
  `total_training_time >= 0.8 * TIME_BUDGET_S`;
- the weak crop/flip loader for the final 20% and the current explicit forkserver
  worker shutdown/recreation lifecycle;
- the learning-rate switch at the same 80% boundary, with `lr=0.1` before it and
  the unchanged `0.01`-to-`1e-4` cosine tail after it.

The intended experimental diff is one numeric literal. Do not introduce a
RandAugment constant, magnitude schedule, in-run choice between 7 and 8, or any
other cleanup/refactor in the same commit.

## Mechanism

RandAugment selects one operation per training image and looks up that operation's
strength at the configured magnitude index. Moving from M7 to M8 preserves the
operation distribution and number of transformations while slightly enlarging
the neighborhood around each plateau-phase training example. The hypothesis is
that EXP-004 has not yet reached the useful-invariance boundary: a small increase
in geometric and photometric perturbation can discourage remaining sensitivity
to nuisance variation, while the unchanged weak hard-label tail restores clean
image statistics and refines the decision boundary.

Under torchvision 0.24.1 on 32x32 inputs, the adjacent-bin changes are:

| Operation family | M7 | M8 | Increment |
|---|---:|---:|---:|
| Rotation | 7 degrees | 8 degrees | +1 degree |
| X/Y translation | 3.384 px | 3.867 px | +0.483 px |
| X/Y shear | 0.07 | 0.08 | +0.01 |
| Brightness/color/contrast/sharpness | 0.21 | 0.24 | +0.03 |
| Solarize threshold | 195.5 | 187.0 | -8.5 |
| Posterize | 7 bits | 7 bits | unchanged after quantization |
| Identity/AutoContrast/Equalize | operation-defined | operation-defined | unchanged |

Ten of the fourteen operation choices become slightly stronger; four are
identical between these bins. This makes M8 a conservative local test rather than
a new augmentation policy. It also bounds the potential gain: many examples see
only a small difference, and roughly four-fourteenths of operation selections see
no magnitude change at all.

## Evidence and Local Rationale

Cubuk et al., *RandAugment: Practical Automated Data Augmentation with a Reduced
Search Space* (NeurIPS 2020), identifies the number of operations and a shared
magnitude as the policy's central tunable dimensions and reports strong CIFAR-10
results. The local paper distillation recommends tuning shared magnitude for the
target model and dataset while combining it with standard crop and flip:
`knowledge/papers/randaugment.md`.

The strongest evidence is goal-local:

- EXP-004's N1 M7 plateau-only policy achieved `92.30%`, a +0.47-point gain over
  the prior moving baseline, in 38,358 steps and 340.7 total seconds.
- Its final strong checkpoint was only `84.60%`, followed by a rapid jump after
  the weak-loader switch and a peak at epoch 98. The weak tail is therefore part
  of the validated mechanism, not removable overhead.
- EXP-005 moved only the augmentation switch to 75% and lost 0.18 points while
  preserving throughput. That result argues for keeping strong-view exposure
  aligned with the 80% LR transition and tuning magnitude instead.

This proposal follows those findings rather than reopening the schedule or phase
duration.

## Hypothesis and Expected Gain

The testable hypothesis is that M8 provides a slightly more useful invariance
pressure during the accepted 240-second high-LR phase and raises
`best_test_acc` from `92.30%` to at least `92.40%`, the required +0.10-point
improvement. A realistic target is approximately `92.40-92.55%`; an improvement
larger than 0.25 points would be welcome but should not be expected from a
one-bin change.

The expected gain is deliberately smaller than EXP-004's +0.47 points. EXP-004
introduced a qualitatively new phase recipe, whereas EXP-006 changes the strength
of only those selected operations whose M7 and M8 lookup values differ. The
primary value of the experiment is a clean local direction test around the
accepted magnitude, not a claim that accuracy increases monotonically with M.

## Semantic-Distortion Risk

CIFAR-10 images are only 32x32. At M8, a translation can move content by nearly
four pixels after the random crop, solarization affects a wider intensity range,
and signed photometric factors reach 0.24 rather than 0.21. These changes remain
moderate individually, but a single operation can still erase class-relevant
detail, introduce default-fill borders, or shift BatchNorm statistics away from
clean evaluation images.

The proposal contains that risk in three ways:

- keep `num_ops=1`, avoiding compounded distortions;
- apply M8 only during the 80% exploratory phase;
- retain the full accepted weak tail so clean crop/flip images resettle BatchNorm
  statistics and train the final hard-label objective.

Do not compensate for a regression by changing interpolation, fill, operation
weights, or the switch boundary. Such changes could be worthwhile later, but
they would prevent attribution to the M7-to-M8 step.

## RNG and Single-Run Noise Caveat

Changing the magnitude index does not intentionally change how many random draws
`torchvision.transforms.RandAugment` makes. Operation identity and sign are
selected with worker-side random draws; magnitude is then a deterministic lookup.
With the same seed, shuffle order, worker count, forkserver lifecycle, operation
count, and switch point, EXP-006 should therefore be more closely paired to
EXP-004 than EXP-005 was: selected operation identities and signs are expected to
align, while their strengths change.

That alignment must not be overstated. The accepted M7 metric and the new M8
metric come from separate multiprocessing/GPU executions, and the project does
not store a per-sample trace proving identical worker assignment, operation
selection, or CUDA execution. RandAugment necessarily participates in the worker
RNG stream, and normal fixed-run variation may be material relative to a 0.10-
point threshold. A successful result demonstrates that the fixed-seed M8
operating point beats the accepted fixed-seed M7 run under the declared protocol;
it does not precisely identify the causal effect size or prove a globally
monotonic magnitude response.

The correct control is to retain seed 42 and run once. Do not reroll, repeat until
success, choose between M7 and M8 after observing multiple seeds, or alter worker
seeding to manufacture paired draws. Record the RNG limitation in the analysis
regardless of verdict.

## Throughput and Wall-Time Feasibility

M8 uses the same operation pool, exactly one transform call per image, identical
image sizes, and the same worker lifecycle as M7. Increasing the magnitude lookup
does not add another PIL operation, GPU kernel, loss computation, model tensor,
or evaluation. Operation runtime should therefore be effectively unchanged.
All RandAugment work remains in DataLoader workers before a batch reaches the
synchronized GPU step timer.

EXP-004's preflight measured M7 strong-loader throughput at 165.5-175.8 batches/s,
well above its approximately 128 batches/s GPU consumption rate. The full run
retained 38,358 steps, 99 epochs, and finished in 340.7 seconds. EXP-006 should
remain near those values; the expected operational band is at least 37,590 steps
(within 2% of EXP-004) and below 360 total seconds.

Run a lightweight M8 loader-only preflight in a fresh process before the GPU run:
use the exact strong transform, batch size 128, eight workers, pinned memory,
`drop_last=True`, persistent forkserver workers, one warmup epoch, and three
timed epochs. Require the slowest timed epoch to sustain at least 140 batches/s
and require all worker processes to shut down cleanly. This threshold is below
the historical M7 minimum but still produces 38,358 batches in about 274 seconds,
fast enough to overlap the 300-second GPU path without expected starvation. If
it fails, treat the node or candidate as preflight-infeasible and do not change
worker count or retry the policy within EXP-006.

The preflight runs in a separate process and performs no model training, so its
RNG consumption cannot alter the later fixed-seed experiment. The mandatory
600-second supervisor still governs the full run even after preflight passes.

## Failure Modes

1. **M7 is already near the local optimum.** The stronger perturbations may add
   no useful invariance or may reduce accuracy below 92.30%.
2. **Semantic damage exceeds the marginal regularization gain.** Small-object
   detail can be lost through translation, solarization, or photometric changes
   even with one operation.
3. **The weak tail cannot fully undo the stronger train/eval shift.** BatchNorm
   and classifier refinement may need more than the fixed final 20% at M8.
4. **The true gain is below run noise.** A 0.10-point boundary is only ten CIFAR-10
   test examples, so a marginal pass or failure must not be described as a
   precise magnitude-response estimate.
5. **Worker throughput changes unexpectedly.** Theoretical operation count is
   unchanged, but node contention or operation-dependent PIL cost can starve
   prefetch and inflate total time.
6. **An implementation edit changes more than magnitude.** Refactoring the
   transform constants or lifecycle creates avoidable scope and RNG confounds;
   the diff must remain one literal.
7. **Final accuracy differs from best.** The current protocol selects
   `best_test_acc` across pre-registered evaluations; do not alter evaluation
   density to favor M8.

## Confound Controls and Excluded Interventions

Use the accepted EXP-004 implementation as the direct parent and preserve all of
its code except the single magnitude literal. Specifically exclude:

- changing `num_ops`, augmentation order, fill, interpolation, magnitude bins,
  weak transforms, or RandAugment operation selection;
- moving the augmentation switch away from 80%, changing the crossing-batch
  behavior, or changing worker shutdown/recreation;
- changing the 80% LR hold, anneal endpoints, optimizer, momentum, weight decay,
  batch size, model, initialization, or normalization;
- adding label smoothing, Mixup, CutMix, random erasing, Cutout, stochastic
  depth, weight averaging, or another regularizer;
- changing persistent workers, `multiprocessing_context`, worker count, pinning,
  shuffling, `drop_last`, or seed handling;
- changing checkpoints, dense-tail evaluation, the test transform, evaluator,
  `prepare.py`, dependency files, or any tracked experimental code besides
  `train.py`;
- running more than one trial or changing seed 42.

The M7 result is the moving baseline, not a component to ensemble with M8. No
checkpoint averaging, multi-run selection, or test-time augmentation is in scope.

## Verification

Before execution:

1. Confirm the moving baseline in `04-results.tsv` is `92.30%` at commit
   `11f8469` and the working `train.py` is the accepted EXP-004 implementation.
2. Confirm the complete code diff is exactly `magnitude=7` to `magnitude=8` in
   `strong_train_tf`; all other `train.py` lines and all out-of-scope files must
   be unchanged.
3. Run static compilation, Ruff/pre-commit, and a transform-constructor smoke
   check.
4. Run the M8 loader preflight above and require at least 140 batches/s in its
   slowest timed epoch plus clean termination of all workers.
5. Confirm exactly one idle NVIDIA H20 with approximately 98 GB VRAM is selected
   and no stale completed run-log variant remains.

Execute exactly one fixed-seed run with redirected output:

```bash
uv run train.py > run.log 2>&1
```

Monitor it without streaming the full log and kill it at 600 seconds if it has
not exited. Verify one and only one augmentation switch at 80%, all eight strong
workers stopped, the post-switch loader is weak crop/flip, and evaluation occurs
no more than once in any epoch.

A valid summary must contain unique finite values for all expected fields, about
300 seconds of counted training, fewer than 600 total seconds, 269,722 model
parameters, and the terminal evaluation at the summary epoch. Record steps,
epochs, total time, peak VRAM, switch progress, late accuracy trajectory, and
final-vs-best gap.

Decision criteria:

- **Improvement:** `best_test_acc >= 92.40%` with all scope, runtime, lifecycle,
  hardware, and evaluation checks passing.
- **No improvement:** valid completion below `92.40%`; do not retry M8 with a new
  seed or pair it with another change.
- **Preflight-infeasible:** slowest M8 timed loader epoch below 140 batches/s or
  incomplete worker shutdown; skip the full GPU run and record the blocker.
- **Failure:** crash, timeout, wrong hardware, scope violation, incomplete/invalid
  summary, counted-budget violation, duplicate per-epoch evaluation, or an
  incorrect augmentation switch.

If M8 wins, accept only this one-bin change on top of EXP-004. If it is neutral
or worse with healthy throughput, revert to M7 and treat the result as evidence
that stronger augmentation is not beneficial at this adjacent operating point;
do not leap directly to M9 in the same experiment.
