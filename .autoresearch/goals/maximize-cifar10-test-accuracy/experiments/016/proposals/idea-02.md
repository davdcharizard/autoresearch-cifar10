# Proposal: One-Operation Mild RandAugment

## Recommendation

Add exactly one worker-side torchvision transform to the accepted training
pipeline:

```python
train_tf = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=1, magnitude=5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)
```

Use torchvision 0.24.1's constructor defaults for every unspecified field:
`num_magnitude_bins=31`, nearest-neighbor interpolation, and `fill=None`
(zero fill for these PIL RGB images). Keep RandAugment active for the entire
run. Preserve the accepted batch-shared `Beta(0.2, 0.2)` mixup through 65% of
counted training time and the subsequent hard-label loss exactly.

This is one fixed policy, not a search. Do not tune magnitude, operation count,
interpolation, fill, probability, ordering, or activation window after
preflight or scoring. In particular, do not use the common stronger
`num_ops=2, magnitude=9` defaults.

## Diagnosis And Rationale

The accepted WRN-16-2 fits the training objective within the 300-second budget
but remains at 94.07% test accuracy. More update exposure alone did not help:
BF16 reached 159.07 passes and scored 93.81%. Low-resolution capacity produced
small positive movements, while stronger mixup, CutMix, residual dropout,
coefficient decorrelation, EMA, and residual endpoint initialization all
failed to cross 94.17%. The remaining error is therefore more consistent with
missing useful invariances or decision-boundary generalization than raw
optimization capacity.

The accepted image pipeline contains only padded crop and horizontal flip.
Neither changes color, contrast, local sampling geometry, or mild orientation.
One low-magnitude RandAugment operation can expose those nuisance directions
without adding parameters, another loss, a second forward pass, or a package.
It is materially different from increasing mixup alpha: mixup constrains
linearity between examples and softens paired targets, while RandAugment keeps
each example's class target and applies a label-preserving transformation.

The proposal is intentionally conservative because local history strongly
warns against additive regularization. In torchvision's 31-bin operation
space, magnitude 5 corresponds to approximately 0.05 shear, 2.42 pixels of
translation on a 32-pixel image, 5 degrees of rotation, or a 0.15 signed
photometric adjustment. Posterize retains seven bits and Solarize uses a
threshold near 212.5. Only one of the 14 uniformly sampled operations is
applied. `Identity`, `AutoContrast`, and `Equalize` are magnitude-independent,
so the policy is not uniformly weak; that is an explicit accuracy risk rather
than a reason to silently edit torchvision's standard operation space.

## Exact Transform Semantics And Order

Keep CIFAR-10's PIL RGB image representation through all three stochastic
operations. Apply them in this exact order:

1. accepted `RandomCrop(32, padding=4)`;
2. accepted `RandomHorizontalFlip()`;
3. candidate `RandAugment(num_ops=1, magnitude=5)`;
4. accepted `ToTensor()`;
5. accepted mean subtraction with `std=(1, 1, 1)`.

Putting RandAugment after crop and flip makes its magnitude operate on the
actual 32x32 training field and avoids augmenting pixels that are immediately
discarded by the crop. It also leaves the accepted crop and flip calls first in
the per-sample random sequence. RandAugment must remain before `ToTensor`
because this proposal uses its established PIL path; moving it after
normalization is invalid, and moving it between crop and flip is a different
treatment.

Do not replace the operation with `AutoAugment`, `TrivialAugmentWide`, or a
hand-written subset. Do not add an outer random-application probability. One
sample always invokes RandAugment, although `Identity` is one possible sampled
operation. Evaluation remains entirely controlled by frozen `prepare.py` and
receives no RandAugment.

## Temporal Choice And Interaction With Mixup

Run RandAugment for all training samples, including the final 35% hard-label
phase. This is the simplest auditable implementation with persistent workers:
workers own and execute the transform while batches are prefetched ahead of
the main process. A precise counted-time cutoff would require shared
multiprocessing state, would take effect at different prefetched samples, and
could not align exactly with the GPU-side mixup transition. An epoch cutoff
would drift with throughput. Either mechanism adds a second treatment and is
excluded from EXP-016.

The choice is not evidence that full-run augmentation is theoretically
optimal. The time-local regularization note suggests an early critical period
may be sufficient, and the accepted mixup result supports late removal of a
strong regularizer. If full-run RandAugment loses accuracy at normal exposure,
the result closes only this exact always-on policy; it does not reject a future
worker-safe early-only augmentation design.

RandAugment stacks with crop/flip and with input mixup during the first 65%.
The sequence is per-example augmentation in DataLoader workers, batch transfer
to CUDA, then the unchanged single batch-shared mixup coefficient and random
permutation. Thus mixed endpoints may have different sampled operations. Do
not share RandAugment choices across paired examples, apply RandAugment to the
already mixed tensor, or alter the coefficient distribution. During the final
35%, labels become hard exactly as accepted while mild image augmentation
continues.

## PIL And Worker RNG Determinism

Retain `torch.manual_seed(42)`, `torch.cuda.manual_seed(42)`, eight workers,
shuffle, `drop_last=True`, and `persistent_workers=True`. Torchvision's crop,
flip, and RandAugment use PyTorch RNG calls. DataLoader seeds each worker from
its deterministic base seed, and the persistent worker then advances its own
stream across samples and epochs. RandAugment's operation index and optional
sign draw therefore occur in the worker that owns the sample, not on CUDA and
not in the main process.

Adding those draws necessarily changes later crop/flip draws within each
worker. That changed training-data trajectory is intrinsic to the treatment;
bitwise equivalence to the accepted augmentation stream is neither expected
nor a valid gate. It does not change model initialization, main-process epoch
shuffling, or CUDA mixup randomness because worker-local draws do not advance
the main CPU or CUDA generators. Do not add a private seed, `worker_init_fn`,
cached policy list, or generator realignment. Those would change the policy or
randomness plumbing and risk becoming seed optimization.

Determinism means two fresh candidate processes with the same environment,
seed, construction order, worker count, and requested batches must produce the
same augmented tensor/label trace. It does not mean accepted and candidate
traces must match, nor does it promise invariance to a different worker count
or torchvision version.

## CPU, Counted-Time, And Wall-Time Cost

RandAugment executes on PIL images in the eight DataLoader workers. It adds no
GPU kernels, parameters, optimizer state, or persistent device allocation.
Peak VRAM should remain near the accepted 1,094 MiB and the model must remain
691,674 parameters.

The timing placement in accepted `train.py` is important: the loop obtains
`inputs, targets` from the DataLoader before setting `t0`. Consequently, batch
fetch and worker-transform wait are excluded from `total_training_time`.
RandAugment should not directly consume the fixed 300 counted GPU seconds or
reduce dataset-equivalent passes, provided worker CPU contention does not slow
the timed body. It can still starve the consumer between steps, add epoch-boundary
latency, and increase `total_seconds` toward the external 600-second ceiling.
That wall overhead must be measured rather than inferred from CUDA step time.

Expected counted-step retention is 98-100%. Expected end-to-end training-loop
throughput retention is 90-100% with eight persistent workers, corresponding
to roughly 142 counted passes and a total run around 345-410 seconds. CPU cost
has medium uncertainty because PIL operation costs vary: `Identity` is cheap,
while `Equalize`, rotation, and geometric resampling are more expensive.

## Predicted Metric Impact

Predict `best_test_acc` in the **93.85-94.35%** range, centered near 94.15%,
with low-to-medium confidence. Standard RandAugment evidence gives positive
prior support, and the accepted pipeline lacks most of its invariances. The
wide interval reflects the stronger local evidence that extra regularization
often harms this already calibrated WRN. Crossing 94.17% is plausible but not
the modal outcome by a large margin.

The decision remains exact: a valid fixed-seed score of at least 94.17% is an
improvement; 94.16% or lower is no-improvement regardless of test loss. One
scored run is allowed. Do not retry a near miss, select a different magnitude,
or interpret operation randomness as justification for a reroll.

## Implementation Scope

Modify only `train.py` by inserting the single constructor shown above after
`RandomHorizontalFlip()` and before `ToTensor()`. An optional single startup
line may print the fixed policy, but no per-sample or per-step logging is
allowed. Preserve all other behavior:

- WRN-16-2, `[2,2,2]`, Kaiming initialization, and 691,674 parameters;
- batch 256, FP32 SGD, momentum 0.9, Nesterov, and matrix-only `5e-4` decay;
- LR 0.2, 5% warmup, cosine decay to the 0.002 floor, and time-based progress;
- batch-shared alpha-0.2 mixup through exactly 65% counted time;
- the accepted hard-label loss after the single transition;
- seed 42, eight persistent workers, local CIFAR-10, and no dependency/network
  operation;
- existing evaluation cadence, at most one evaluation per epoch, and the
  frozen evaluator.

## Evaluator-Free Preflight Gates

Run all preflight checks locally without calling, constructing output from, or
inspecting the test evaluator. If importing `train.py` is necessary, replace
`prepare.Eval` with a fail-closed stub before import. Use only the local CIFAR
training split or synthetic PIL images; no test labels, evaluator metrics,
package install, or network access are permitted.

### Static And Semantic Gates

1. Inspect the `Compose` children and require the exact ordered types
   `RandomCrop`, `RandomHorizontalFlip`, `RandAugment`, `ToTensor`, `Normalize`.
   Require `num_ops == 1`, `magnitude == 5`, `num_magnitude_bins == 31`, nearest
   interpolation, and `fill is None`.
2. Assert the accepted crop padding/size, flip probability, normalization
   mean/unit standard deviations, batch size, worker settings, and all model and
   optimizer constants are unchanged.
3. On a nonuniform synthetic RGB PIL image, run the RandAugment object over at
   least 256 fixed-seed draws. Require every result to remain PIL RGB and 32x32,
   every pixel to remain in `[0,255]`, finite tensor conversion, more than one
   distinct output hash, and both changed and unchanged outputs. This catches a
   dead transform, accidental tensor/normalized input, and an operation count
   greater than zero without selecting on accuracy.
4. Verify directly from the instantiated augmentation space that magnitude 5
   yields the expected approximate shear, translation, rotation, photometric,
   posterize, and solarize values. This catches an altered bin count or wrong
   magnitude while allowing magnitude-free operations.
5. Start two fresh processes that reproduce candidate main-process seeding,
   DataLoader/model construction order, worker count, and persistent-worker
   iteration. Hash at least the first 16 full training batches, including labels,
   and require identical traces between processes. Also require a different
   trace from the accepted transform under the same seed. A failed candidate
   replay blocks scoring; do not repair it with another seed.

### Loader And Production-Timing Gates

Benchmark accepted and candidate pipelines in fresh, order-balanced processes
so their worker pools do not contend. Each process must use the local training
split, eight persistent workers, pinned memory, batch 256, and the exact
production model/optimizer step. Measure `next(iterator)` wall time separately
from the existing timed CUDA body and total batch-to-batch wall time. Exercise
both the mixup regime and hard-label regime, warm workers and cuDNN before
measurement, include at least one epoch boundary, and collect at least three
windows per path. Never run evaluation.

For each path report median window means and population CVs, accepted/candidate
counted-step retention, end-to-end retention, mean and p99 fetch wait, and peak
VRAM. Use order-balanced windows or fresh subprocess order
`accepted,candidate,candidate,accepted`; do not keep both worker pools alive
while timing one path.

Proceed to the sole scored run only if all of these fixed gates pass:

- all semantic and deterministic replay assertions pass;
- every window-level timing CV is at most 0.05;
- counted CUDA-body retention is at least 0.98 and its projection from 141.9
  accepted passes is at least 139.0;
- end-to-end batch throughput retention is at least 0.85;
- a conservative projection from the accepted approximately 341-second total
  run is below 480 seconds;
- there is no repeated post-warmup fetch stall above one second, no worker
  exception, no non-finite loss, no OOM, and no parameter/VRAM change beyond
  ordinary allocator noise.

The 85% end-to-end gate is intentionally looser than the counted-step gate:
worker wait is outside the 300-second training timer and the accepted run has
substantial headroom below 600 seconds. The separate 480-second projection
still leaves at least two minutes for variance and sparse validation. Do not
weaken a failed gate, reduce operation count, or change magnitude in EXP-016.

## Full-Run Verification

After preflight passes, remove stale `run.log` and execute exactly once:

```bash
timeout 600s uv run train.py > run.log 2>&1
```

Require exit code 0 on one NVIDIA H20, a complete summary, approximately 300
counted training seconds, total wall time below 600 seconds, finite loss, and no
more than one validation per epoch. Confirm exactly one mixup transition near
195 counted seconds, unchanged model parameter count, and unchanged evaluation
cadence. Record best/final accuracy, final loss, epochs, steps, realized passes,
startup and total wall time, peak VRAM, and the mixup transition step/time.

Accuracy is authoritative after a valid run even if realized wall overhead
differs from preflight. A timeout, worker crash, missing summary, semantic drift,
or invalid evaluation cadence is a failed experiment, not a score and not a
reason to run a cheaper variant under the same ID.

## Failure Modes And Interpretation

- **Additive over-regularization:** crop/flip, RandAugment, and mixup may jointly
  distort early inputs too strongly. A lower score with normal counted exposure
  rejects this exact stacked policy.
- **Hard-label-tail interference:** always-on RandAugment may prevent the clean
  margin refinement that makes the final 35% useful. This would motivate only a
  separately designed, worker-safe temporal policy, not an EXP-016 rescue.
- **Magnitude-free strong operations:** Equalize or AutoContrast can be
  substantial despite magnitude 5. Removing them would create a custom policy
  and is forbidden here.
- **Nearest-neighbor artifacts and zero fill:** geometric operations may add
  aliasing or artificial borders. Bilinear interpolation or mean-colored fill
  are separate treatments and cannot be adopted after seeing the result.
- **Worker starvation:** PIL transforms may lengthen uncounted fetch time enough
  to threaten 600 seconds even while counted exposure looks normal. The paired
  loader benchmark and conservative wall projection guard this explicitly.
- **RNG trajectory change:** later crop/flip choices differ because RandAugment
  consumes worker RNG. This is expected treatment behavior, not seed rerolling
  and not evidence for another run.
- **Effect smaller than test granularity/noise:** one weak operation may not
  move enough top-1 boundaries. A valid result below 94.17% closes the fixed
  standalone policy even if it is numerically near the baseline.

## Falsifiable Hypothesis

Adding exactly one torchvision `RandAugment(num_ops=1, magnitude=5)` operation
after accepted crop/flip and before tensor conversion, while preserving
batch-shared alpha-0.2 mixup through 65% and every other accepted choice, will
retain at least 98% counted-step throughput, project below 480 seconds total
wall time, and raise fixed-seed `best_test_acc` from 94.07% to at least 94.17%.

A valid lower score rejects always-on standard mild RandAugment as a sufficient
standalone improvement for this recipe. Do not rescue it by changing order,
magnitude, operation count, operation space, interpolation, fill, probability,
activation window, mixup, or seed.

## Evidence Base

- `knowledge/papers/randaugment.md`: RandAugment reduces learned augmentation to
  operation count and shared magnitude and reports CIFAR gains without policy
  search; it also identifies strength and CPU cost as the local risks.
- `knowledge/papers/time-matters-regularization.md`: early augmentation effects
  may persist after removal, motivating caution about always-on augmentation
  and reserving temporal removal as a distinct future experiment.
- `experiments/002/04-analysis.md`: batch-shared alpha-0.2 mixup through 65%
  reached the accepted 94.07% and defines the recipe that must remain intact.
- `experiments/003/04-analysis.md`, `experiments/005/04-analysis.md`, and
  `experiments/006/04-analysis.md`: CutMix, stronger mixup, and residual dropout
  regress, establishing a strong prior against stacked regularization.
- `experiments/015/04-analysis.md`: per-example coefficient diversity also
  regresses, reinforcing the decision to leave accepted mixup untouched.
- `04-results.tsv`, `03-experiment-learnings.md`, and accepted `train.py`:
  fixed metric threshold, local failures, RNG/timer placement, loader settings,
  and implementation constraints.
