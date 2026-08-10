# Proposal: Plateau CutMix Composed with RandAugment and a Weak Hard Tail

## Summary

Add canonical input-level CutMix to the accepted EXP-004 strong phase while
retaining `RandAugment(num_ops=1, magnitude=7)`. Perform CutMix in the training
DataLoader's worker-side `collate_fn`, not inside the synchronized GPU step. At
the existing 80% boundary, shut down the strong workers exactly as EXP-004 does
and rebuild the existing weak crop/flip loader with ordinary hard integer labels.

Use the installed `torchvision.transforms.v2.CutMix` implementation with
`alpha=1.0` and `num_classes=10`. It replaces a rectangular region with pixels
from another shuffled minibatch example and returns labels mixed in proportion
to the rectangle's actual clipped area. This directly answers EXP-006's failure:
fixed 16x16 Cutout discarded 25% of every image and kept the original label at
weight one, whereas CutMix fills the region with class-bearing pixels and assigns
the donor class the exact corresponding target mass.

The proposal composes CutMix with RandAugment rather than replacing RandAugment.
Removing N1/M7 would simultaneously discard the only validated broad-invariance
intervention and add a new occlusion intervention. EXP-006 already showed that
this replacement framing can preserve throughput yet lose 0.67 accuracy points.
Composition changes one conceptual lever relative to the accepted parent: every
strong-phase source image still receives the established crop, flip, and N1/M7
transform, and CutMix then creates a labeled regional composition from those
views. The full weak hard-label tail is retained to remove the compounded
augmentation and probability-target distribution before final evaluation.

## Why CutMix Is Different from Failed Cutout

EXP-006 replaced RandAugment with a normalized-zero 16x16 square on every strong
view. It retained 38,028 steps, so infrastructure and optimization exposure were
not the cause of failure, but its final strong checkpoint was 1.45 points below
EXP-004 and its first weak checkpoint remained 0.96 points behind. The weak tail
inherited a worse representation. The likely mechanism was a combination of
three issues:

- 25% of every input was made uninformative;
- the target still asserted that the occluded source class explained 100% of the
  image;
- replacing N1/M7 removed the broad geometric and photometric invariances that
  had already produced the best local result.

CutMix changes all three properties. The pasted rectangle contains real pixels
from another transformed CIFAR-10 image; both visible classes receive target mass
according to area; and N1/M7 remains active before batch composition. The ICCV
paper's central argument is precisely that regional dropout can retain its
partial-view regularization without wasting the erased region. Its CIFAR-10
experiment improved top-1 error from 3.85% to 2.88%, outperforming both Cutout
and Mixup in that paper's much larger and longer-trained PyramidNet setting. That
result establishes mechanism plausibility, not an expected effect size for this
small 300-second ResNet-20.

## Replace versus Compose Decision

**Decision: compose CutMix with the accepted RandAugment strong phase.**

Replacing RandAugment would be a head-to-head regularizer substitution, but it is
not the most controlled test against the moving baseline. It changes both the
removal of a successful method and the addition of an untested method. A loss
could mean CutMix is ineffective, or merely that removing broad invariance costs
more than CutMix adds. That ambiguity is especially serious after EXP-006.

Composition has a clearer comparison:

```text
accepted EXP-004 plateau: crop + flip + RandAugment N1/M7 -> hard label
proposed EXP-007 plateau: crop + flip + RandAugment N1/M7 -> CutMix -> area label
accepted/proposed tail:  crop + flip -> hard label
```

The risk is stronger overall regularization. Each source and donor view may carry
a different RandAugment operation before patching, and a 0.27M-parameter model
may not fit the combined distribution in 240 seconds. The unchanged 60-second
weak tail is the predeclared mitigation, not permission to weaken CutMix after
seeing the result.

## Worker Collate versus Timed GPU Decision

**Decision: perform CutMix in DataLoader worker collation.**

The current training timer begins after the DataLoader yields a complete batch.
A GPU-side implementation after `t0` would charge the following to the fixed
300-second optimization budget: Beta and rectangle parameter handling, a batch
clone, batch roll, rectangular assignment, larger target transfer, and the
probability-target loss. EXP-003 demonstrated that even a loss-only change can
reduce fixed-budget steps by 6.7% in this tiny-model regime.

With automatic batching and `num_workers=8`, PyTorch executes `collate_fn` in the
workers. Running torchvision CutMix there moves sampling, cloning, rolling, and
patch assignment outside the synchronized GPU step. The only remaining timed
difference is transferring a `[128, 10]` float target rather than `[128]` integer
targets and executing probability-target cross-entropy. Both costs must still be
microbenchmarked; worker placement does not justify calling the method free.

The worker path also integrates cleanly with the accepted phase lifecycle. The
strong loader owns both N1/M7 and CutMix. Its prefetched batches and all eight
workers are destroyed at 80%, then the new weak loader uses default collation and
hard targets. No mixed batch can leak into the tail if the current iterator-break
and verified shutdown sequence remains intact.

## Installed API Semantics

The installed environment contains torchvision `0.24.1+cu128` with:

```python
v2.CutMix(*, alpha: float = 1.0, num_classes: int | None = None,
          labels_getter="default")
```

For a collated image tensor `[B, C, H, W]` and integer labels `[B]`, the installed
implementation:

1. samples one scalar `lambda ~ Beta(alpha, alpha)` for the whole batch;
2. samples one rectangle center on the 32x32 spatial grid;
3. sets nominal rectangle width and height proportional to
   `sqrt(1 - lambda)` and clips the box to image boundaries;
4. pairs examples by rolling the already shuffled batch by one position;
5. clones the batch and replaces the same rectangle in every item with the
   corresponding rectangle from its rolled donor;
6. recomputes `lambda_adjusted = 1 - pasted_area / 1024` after clipping;
7. one-hot encodes labels to `[B, 10]` and returns
   `lambda_adjusted * y + (1 - lambda_adjusted) * roll(y, 1)`.

The deterministic roll is an installed implementation detail, not a random
permutation. It is acceptable here because the DataLoader already shuffles the
training set and uses `drop_last=True`, so rolled neighbors are random training
pairs without another permutation kernel. The same lambda and box for all 128
items matches the minibatch-level form used by standard CutMix implementations.

Set `alpha=1.0`. The primary paper used `alpha=1` in all experiments and found it
best among its reported alpha ablation. Do not add a separate application
probability in EXP-007: installed `v2.CutMix` applies exactly one CutMix operation
per batch, and adding a custom probability would create a second unvalidated
regularization-strength parameter. The plateau-only schedule already limits its
application to the first 80% of counted time.

## Exact Code Mechanics

Add the installed batch-transform and collate imports:

```python
from torch.utils.data import DataLoader, default_collate
from torchvision import datasets, transforms
from torchvision.transforms import v2
```

After `NUM_CLASSES` is defined, add the fixed CutMix configuration:

```python
CUTMIX_ALPHA = 1.0
cutmix = v2.CutMix(alpha=CUTMIX_ALPHA, num_classes=NUM_CLASSES)
```

Define a module-level collator so it is picklable under the existing forkserver
workers:

```python
def cutmix_collate(batch):
    inputs, targets = default_collate(batch)
    with torch.random.fork_rng(devices=[]):
        return cutmix(inputs, targets)
```

The CPU `fork_rng` context is a confound control. RandAugment and CutMix both use
the worker's PyTorch RNG. Without the context, CutMix's Beta and rectangle draws
would advance that state and change all subsequent RandAugment choices in the
worker. Saving and restoring the state lets CutMix sample parameters from a
batch-dependent worker state while preserving the later RandAugment RNG stream.
No CUDA RNG is accessed in worker collation, hence `devices=[]`.

Extend the loader factory without changing any existing loader option:

```python
def make_train_loader(transform, collate_fn=None):
    train_set = datasets.CIFAR10(
        DATASET_DIR, train=True, download=True, transform=transform
    )
    return DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        multiprocessing_context="forkserver",
        collate_fn=collate_fn,
    )
```

Construct only the strong loader with CutMix:

```python
train_loader = make_train_loader(strong_train_tf, collate_fn=cutmix_collate)
```

Keep the current tail construction unchanged:

```python
train_loader = make_train_loader(weak_train_tf)
```

`collate_fn=None` selects PyTorch's default collation, so weak-tail targets return
to integer shape `[B]`. Update only the provenance print to make the transition
auditable:

```python
"augmentation_switch: randaugment+cutmix->base ..."
```

No forward-loop branch is needed. The existing statement handles both target
formats:

```python
loss = F.cross_entropy(outputs, targets)
```

During the strong phase, targets are float probability vectors `[B, 10]` and
cross-entropy computes the mean soft-target negative log likelihood. During the
weak tail, targets are integer class indices `[B]` and the same API computes
ordinary hard-label cross-entropy. Do not add label smoothing, a second weighted
cross-entropy call, manual lambda reconstruction, or a custom soft-label loss in
this experiment.

## Target and Loss Correctness

For each mixed example, the target must exactly reflect the observed rectangle,
not the initially sampled Beta value. Boundary clipping can make the pasted area
smaller than nominal. The installed transform's adjusted lambda prevents the
label mismatch that a naive implementation would introduce on 32x32 inputs.

The following invariants must hold in preflight:

- strong images have shape `[128, 3, 32, 32]` and strong targets `[128, 10]`;
- each probability target is finite, nonnegative, and sums to one within
  floating-point tolerance;
- at most two classes have nonzero mass for ordinary distinct-label pairs;
- using synthetic constant-valued source images, the donor target weight agrees
  with the observed pasted-pixel fraction within `1 / 1024`;
- weak-loader targets after the switch have shape `[128]`, dtype `torch.int64`,
  and no probability targets remain prefetched;
- `F.cross_entropy` accepts both formats and produces a finite scalar with finite
  gradients in the installed environment.

The logged training loss changes semantics at the switch. Plateau loss includes
soft regional targets; tail loss is hard-label CE. Its absolute level cannot be
compared directly with EXP-004 or across the 80% boundary. Test loss remains the
unchanged fixed evaluator's hard-label metric.

## RNG and Worker Lifecycle

The main process retains `torch.manual_seed(42)` and
`torch.cuda.manual_seed(42)`. No seed is added, changed, rerolled, or selected.
DataLoader shuffling and worker base seeds remain governed by the existing main
RNG and forkserver construction.

Within each strong worker, per-sample crop, flip, and RandAugment run before
collation. `cutmix_collate` then saves the CPU RNG state, samples CutMix, and
restores the state. This design aims to keep subsequent N1/M7 operation draws
aligned with the accepted stream while still obtaining varied CutMix parameters
because the saved state differs after each batch's per-sample transforms.

Preflight must explicitly assert that the CPU RNG state before and after a direct
`cutmix_collate` call is identical. It must also show that consecutive batches do
not all receive the same lambda/rectangle effect. If either assertion fails, do
not run the full experiment; fix the collator as a planning defect without
changing the declared algorithm.

At 80%, retain the current order:

1. break immediately after the crossing strong batch;
2. release the strong iterator;
3. perform the scheduled evaluation at most once for that epoch;
4. stop and join all eight strong workers;
5. delete the strong loader and collect garbage;
6. create a weak default-collate loader;
7. set the phase flag false and resume the cosine tail.

This guarantees that RandAugment, CutMix, and probability targets end together at
the accepted LR transition. Do not reuse the strong workers with a mutable flag;
prefetch could otherwise leak mixed batches into the hard tail.

Even with `fork_rng`, the EXP-004 and EXP-007 metrics come from separate
multiprocessing and CUDA executions. Exact operation pairing is not recorded, and
the CutMix intervention necessarily changes inputs, targets, gradients, and later
model trajectory. A marginal delta cannot be presented as a precise causal effect
size. The fixed one-run policy still governs: do not rerun or change seeds to
resolve noise.

## Mandatory Throughput Preflights

Run diagnostics in disposable fresh processes so they cannot consume the final
run's RNG state. Do not run a partial or full training experiment during proposal
development.

### Worker-Collate Gate

Benchmark the exact strong loader with crop, flip, N1/M7, `cutmix_collate`, batch
128, eight workers, pinning, `drop_last`, persistence, and forkserver. Consume one
full warmup epoch, then time three full epochs and record each rate. Also measure
the real strong-worker shutdown and first weak batch.

Required gates:

- exactly 390 batches per full epoch;
- slowest warm timed strong epoch at least 140 batches/s;
- no timed epoch more than 20% slower than the median;
- all eight strong workers stop and join successfully;
- strong-to-weak transition plus first weak batch at most 5 seconds;
- first weak batch has hard integer labels and no CutMix state leakage.

EXP-004 measured 165.5-175.8 N1/M7 batches/s and consumed about 128 GPU batches/s.
The 140-batch/s gate leaves prefetch margin while allowing worker-side batch copy
cost. If this gate fails, do not move CutMix onto the GPU to rescue the idea; mark
this implementation preflight-infeasible because that would change the counted
compute exposure.

### Timed GPU Loss Gate

In a separate idle-H20 process, microbenchmark the exact synchronized training
step with hard `[128]` targets versus probability `[128, 10]` CutMix targets. Use
the current ResNet-20, FP32 inputs, SGD, H2D copies from pinned host tensors, loss,
backward, optimizer step, and `torch.cuda.synchronize()`. Warm both paths before
collecting at least 500 interleaved measurements so CUDA startup and ordering do
not favor one target format.

Let `t_hard` and `t_soft` be median full-step times. Because soft targets apply
for 80% and hard targets for 20%, estimate relative full-run update exposure as:

```text
predicted_step_ratio = 0.8 * (t_hard / t_soft) + 0.2
predicted_steps = 38_358 * predicted_step_ratio
```

Proceed only if:

- `predicted_step_ratio >= 0.965`;
- `predicted_steps >= 37_015`;
- both paths produce finite losses/gradients; and
- the soft path adds no unexpected synchronization or material VRAM increase.

This gate permits at most about 3.5% loss of EXP-004 update exposure across the
whole run, roughly half the 6.7% loss observed with EXP-003 label smoothing. If
the installed probability-target CE misses the gate, do not substitute a custom
loss after observing the benchmark. Record the candidate as preflight-infeasible
and evaluate a separately reviewed sparse-target implementation later.

## Runtime and Resource Expectations

CutMix's image clone, roll, and patch assignment occur in workers and should not
enter synchronized `dt`. The strong loader must nevertheless keep pace or total
wall time will rise while `training_seconds` remains 300.0. Passing 140 batches/s
means producing 38,358 batches takes about 274 seconds, fast enough to overlap the
GPU path. EXP-004 finished in 340.7 seconds, so there is substantial margin below
the 600-second hard timeout even with worker and phase-switch variation.

GPU memory impact is small: probability targets add only a few kilobytes per
batch, while image tensors and model shapes are unchanged. Worker memory rises
because CutMix clones a collated batch of roughly 1.5 MiB before pinning. Eight
workers and prefetch can multiply that transient allocation, so preflight should
also confirm no worker failure or runaway host memory, though H20 VRAM remains
near EXP-004's 330.1 MiB.

If both gates pass, expect approximately 37,000-38,400 optimizer steps, 95-99
epochs, 300 seconds counted training, and roughly 340-380 seconds total. A result
outside those bands is not automatically invalid, but it is a throughput confound
that must be reported before attributing accuracy to CutMix.

## Hypothesis and Expected Impact

The testable hypothesis is that labeled regional composition during the accepted
strong phase preserves the broad invariance benefit of N1/M7, avoids Cutout's
uninformative 25% hole, and teaches the network to recognize class evidence from
partial views. The unchanged weak hard tail then converts those features to the
clean test distribution. Under the preflight exposure gates, this will raise
`best_test_acc` from `92.30%` to at least `92.40%`.

A plausible successful range is `92.40-92.70%` (+0.10 to +0.40 points). The paper
reports larger gains, but those results use far larger models and 300 epochs; this
run has a small ResNet-20 and roughly 100 epochs. The expected effect should be
discounted accordingly. The main diagnostic trajectory is the weak tail: the
final strong checkpoint may be substantially below EXP-004 because probability
targets and composite images change clean evaluation, but the first several weak
epochs should close that gap faster than EXP-006 did if information retention is
working.

## Failure Modes

1. **Compounded augmentation underfits.** N1/M7 plus always-on alpha-1 CutMix may
   exceed the capacity or convergence speed of the 0.27M-parameter ResNet-20.
2. **Soft-target GPU overhead costs too many updates.** Worker-side image mixing
   does not remove probability-target CE from the synchronized step.
3. **The weak tail is too short.** Sixty seconds may not fully recover clean
   BatchNorm statistics and hard-target confidence after combined augmentation.
4. **Area labels are semantically imperfect.** Object pixels are not uniformly
   distributed, so rectangle area is only an approximation to class evidence,
   especially on 32x32 images.
5. **Patch boundaries become shortcuts.** Combining separately RandAugmented
   views can create edges, fill artifacts, or photometric discontinuities that the
   model exploits rather than learning robust object parts.
6. **Small or same-class pairs weaken the intervention.** Clipped rectangles can
   be small, and rolled neighbors can share a class, yielding little label change.
7. **Worker collation starves prefetch.** CPU batch clones and patch copies can
   increase total wall time despite being excluded from synchronized `dt`.
8. **RNG isolation is incomplete.** If CutMix advances worker RNG outside the
   saved context, later RandAugment draws become an unplanned confound.
9. **Best accuracy reflects run noise.** A 0.10-point threshold is only ten test
   images; one fixed run cannot quantify an exact causal gain.
10. **Training loss is misread.** Soft plateau loss and hard tail loss have
    different semantics and should not be compared numerically.

## Confound Controls and Excluded Interventions

Use accepted EXP-004 commit `11f8469` as the parent. Add only the installed CutMix
collator, its loader plumbing, RNG isolation, and truthful phase log. Preserve:

- ResNet-20 depth, widths, Option-A shortcuts, initialization, normalization, and
  269,722 parameters;
- batch size 128, SGD, standard momentum, weight decay `1e-4`, and all optimizer
  code;
- `lr=0.1` through 80%, the immediate `0.01` tail start, cosine decay to `1e-4`,
  and elapsed-time schedule math;
- N1/M7 RandAugment, crop/flip order, means/stds, and all transform parameters;
- the exact 80% iterator break and strong-to-weak worker transition;
- eight forkserver workers, persistence, pinning, shuffling, and `drop_last`;
- seed 42, one-run policy, model evaluation mode, fixed test preprocessing,
  evaluator, checkpoints, and at-most-once-per-epoch validation;
- `prepare.py`, dependency files, and all tracked experimental code except
  `train.py`.

Do not add a CutMix application probability, Mixup, label smoothing, Cutout,
RandomErasing, magnitude change, custom interpolation/fill, operation blacklist,
custom loss, second forward pass, test-time augmentation, weight averaging, or
architecture change. Do not replace RandAugment, move CutMix to the GPU, alter
the weak tail, or tune alpha after preflight. Any such change requires a separate
proposal.

## Verification Procedure

Before the full run:

1. Confirm the moving baseline is `92.30%` at accepted commit `11f8469`.
2. Confirm the diff touches only `train.py` and matches the declared imports,
   CutMix configuration/collator, loader argument, strong-loader call, and phase
   log. Review every other line as unchanged.
3. Run Python compilation, Ruff/pre-commit, and installed-API smoke tests.
4. Run the target-area, RNG-state, worker-lifecycle, worker-throughput, and timed
   GPU loss gates above in fresh processes. Do not continue after a failed gate.
5. Confirm exactly one idle NVIDIA H20 with approximately 98 GB VRAM is visible.
6. Confirm no stale completed `run.log` or renamed run-log variant remains.

Run exactly one experiment with all output redirected:

```bash
uv run train.py > run.log 2>&1
```

Monitor without streaming the full log and kill the process at 600 seconds if it
has not exited. Do not retry a crash with altered hyperparameters or rerun a valid
metric with another seed.

Post-run integrity checks:

- one finite, unique final summary with about 300 seconds counted training and
  fewer than 600 seconds total;
- exactly one `randaugment+cutmix->base` switch at 80.0%;
- all eight strong workers reported stopped and no worker survived the switch;
- no more than one evaluation in any epoch, with a terminal evaluation matching
  the summary epoch;
- 269,722 parameters and no unexpected VRAM growth;
- strong phase used probability targets, weak tail used integer targets, and no
  mixed prefetched batch crossed the switch;
- step count and total runtime compared explicitly with EXP-004's 38,358 and
  340.7 seconds.

Decision criteria:

- **Improvement:** `best_test_acc >= 92.40%` and every scope, hardware, budget,
  lifecycle, target, and evaluation check passes.
- **No improvement:** valid completion below `92.40%`; do not rerun or weaken
  CutMix post hoc.
- **Preflight-infeasible:** either worker rate below 140 batches/s, transition
  failure, RNG-state failure, target-area mismatch, or predicted step ratio below
  0.965; skip the full run and record the measured blocker.
- **Failure:** crash, timeout, wrong hardware, invalid summary, scope violation,
  incorrect switch, probability-target leakage into the tail, duplicate epoch
  evaluation, or counted-budget violation.

If the candidate improves, accept the complete composed plateau recipe and retain
the weak hard tail. If it fails with healthy throughput, revert all CutMix code
and keep EXP-004; the result would show that retaining pixels is not sufficient
to overcome compounded augmentation or soft-target convergence in this finite
horizon. Do not interpret a failure as support for returning to fixed Cutout.

## Sources

- Yun et al., *CutMix: Regularization Strategy to Train Strong Classifiers with
  Localizable Features*, ICCV 2019:
  <https://openaccess.thecvf.com/content_ICCV_2019/html/Yun_CutMix_Regularization_Strategy_to_Train_Strong_Classifiers_With_Localizable_Features_ICCV_2019_paper.html>
- Installed torchvision 0.24.1 `v2.CutMix` source and signature, inspected in the
  active environment.
- Local Mixup distillation: `knowledge/papers/mixup.md`.
- EXP-006 failure analysis:
  `experiments/006/04-analysis.md`.
