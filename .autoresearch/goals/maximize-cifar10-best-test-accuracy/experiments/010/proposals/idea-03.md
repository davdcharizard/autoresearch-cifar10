# Proposal: Conservative Plateau CutMix on the Width-2 N1/M7 Recipe

## Summary

Compose conservative batch CutMix with the accepted width-2 strong phase. Keep
the existing crop, horizontal flip, and `RandAugment(num_ops=1, magnitude=7)`
pipeline, then apply `torchvision.transforms.v2.CutMix(alpha=1.0,
num_classes=10)` to 50% of collated plateau batches in DataLoader workers. At the
existing 80% elapsed-training boundary, destroy the strong loader and rebuild the
accepted weak crop/flip loader with ordinary hard integer labels.

This is not a retry of EXP-006. That experiment removed N1/M7, erased a fixed 25%
of every image with no donor content, and kept 100% target mass on the occluded
class. The proposed method preserves the validated broad invariances, fills each
selected rectangle with pixels from another real training image, assigns target
mass according to the rectangle's actual clipped area, applies the intervention
to only half of plateau batches, and retains the full hard-label weak tail.

The current moving baseline is the EXP-007 width-2 model at `93.55%` and 27,143
steps. Success requires at least `93.65%`. Because width 2 already loses 29.2% of
the width-1 update exposure, worker throughput and probability-target GPU cost
are explicit preflight gates rather than assumed negligible.

## Experimental Motivation

The local sequence establishes a narrow composition point:

- EXP-004 validated N1/M7 through the 80% high-LR boundary followed by a weak
  hard-label tail, reaching `92.30%` while preserving optimizer exposure.
- EXP-006 replaced N1/M7 with fixed 16x16 Cutout and regressed by 0.67 points even
  though it retained 99.14% of the accepted steps. Its lower strong checkpoint
  and persistent weak-tail deficit identify representation quality, not runtime,
  as the failure.
- EXP-007 retained the complete EXP-004 data lifecycle and doubled model width.
  It reached `93.55%` despite only 27,143 steps, showing that added capacity can
  absorb the difficult strong-view phase and is worth preserving.
- EXP-009 matched EXP-007 exposure but did not improve by relaxing BN/bias decay.
  Together with EXP-008, this brackets all-parameter `1e-4` decay and motivates a
  different generalization mechanism.

CutMix is a mechanism-specific response to Cutout's information loss. It keeps a
regional-occlusion objective while ensuring that the replaced region remains
class-bearing and that supervision represents both visible sources. The wider
model has more capacity to fit these composite examples than the 0.27M-parameter
model considered in the earlier warning against stacking strong augmentation.
That makes a conservative composition plausible now, but still higher risk than
a compute-neutral change.

## Compose, Do Not Replace, N1/M7

Use CutMix after the accepted per-sample N1/M7 transform. Replacing RandAugment
would remove the only validated input-invariance component while adding an
untested regional method, making any outcome a net comparison of two changes.
EXP-006 already showed that retaining throughput does not compensate for losing
N1/M7's broad geometric and photometric pressure.

The controlled comparison is:

```text
EXP-007 plateau:
    crop + flip + N1/M7 -> hard label

EXP-010 candidate plateau:
    crop + flip + N1/M7 -> 50% CutMix batch / 50% unchanged batch
                        -> area target / hard label

Both tails:
    crop + flip -> hard label
```

Relative to the moving baseline, the sole conceptual addition is conservative
regional mixing. N1/M7 strength, operation order, phase duration, width, loss API,
optimizer, and tail remain fixed.

## Probability and Alpha Choice

Use:

```python
CUTMIX_ALPHA = 1.0
CUTMIX_PROBABILITY = 0.5
```

`alpha=1.0` is the canonical CutMix setting, gives a uniform pre-clipping lambda,
and was the best setting in the primary paper's reported alpha ablation. Lower
alpha is not necessarily more conservative: a Beta distribution below one is
U-shaped and creates more nearly empty or nearly dominant patches. Raising alpha
concentrates area around balanced composites, which would strengthen rather than
weaken the soft-label burden. Keep alpha at the evidence-backed value.

The 0.5 batch probability is the conservative lever. Always-on CutMix composed
with N1/M7 would expose every strong-phase view to two augmentation mechanisms
and every plateau optimizer step to dense probability-target cross-entropy. At
`p=0.5`, only half of plateau batches are mixed. Since the plateau occupies 80%
of counted time, approximately 40% of total optimizer steps use probability
targets; the other strong steps and the entire weak tail retain the accepted hard
objective.

Do not tune probability or alpha after preflight. Do not make probability depend
on elapsed progress, class pairing, lambda, or validation results. One fixed
operating point is necessary for attribution.

## Worker-Side Implementation Decision

Perform image mixing in DataLoader worker collation, not after `t0` on the GPU.
The current timer starts only after a batch is yielded. Worker placement therefore
keeps Beta sampling, batch clone/roll, bounding-box computation, and rectangular
copy outside the synchronized 300-second GPU budget. A GPU implementation would
charge all of those operations against update exposure and repeat the central
fixed-time risk exposed by EXP-003.

Worker placement does not make CutMix free. Dense `[128, 10]` targets are larger
than integer `[128]` targets, their H2D transfer is timed, and probability-target
cross-entropy takes a different GPU path. The proposed preflight measures that
remaining cost on the accepted width-2 model.

## Installed Torchvision Semantics

The installed environment is torchvision `0.24.1+cu128`, with:

```python
v2.CutMix(*, alpha: float = 1.0, num_classes: int | None = None,
          labels_getter="default")
```

For collated images `[B, C, H, W]` and integer labels `[B]`, the installed
implementation:

1. samples one `lambda ~ Beta(alpha, alpha)` for the batch;
2. samples one rectangle center and derives width/height from
   `sqrt(1 - lambda)`;
3. clips the rectangle to the 32x32 image;
4. pairs examples by rolling the already shuffled batch by one position;
5. clones the batch and pastes the rolled donor rectangle;
6. recomputes `lambda_adjusted = 1 - pasted_area / 1024` after clipping; and
7. returns one-hot probability targets mixed with the adjusted lambda.

The batch roll is deterministic, but DataLoader shuffling makes its neighbors
random training pairs. One lambda and rectangle are shared across the batch. The
area correction is important on 32x32 images: using the sampled rather than
clipped area would give incorrect target mass near image boundaries.

CutMix runs after per-sample normalization in collation. Because source and donor
use the same channel-wise affine normalization, pasting normalized donor pixels
is equivalent to pasting before that shared normalization. No mean-valued blank
region is introduced.

## Exact Code Mechanics

Add only the necessary imports:

```python
from torch.utils.data import DataLoader, default_collate
from torchvision import datasets, transforms
from torchvision.transforms import v2
```

After `NUM_CLASSES`, add fixed configuration and a module-level transform:

```python
CUTMIX_ALPHA = 1.0
CUTMIX_PROBABILITY = 0.5
cutmix = v2.CutMix(alpha=CUTMIX_ALPHA, num_classes=NUM_CLASSES)
```

Define a module-level collator so forkserver workers can import it:

```python
def cutmix_collate(batch):
    inputs, targets = default_collate(batch)
    with torch.random.fork_rng(devices=[]):
        if torch.rand(()).item() < CUTMIX_PROBABILITY:
            return cutmix(inputs, targets)
    return inputs, targets
```

The CPU `fork_rng` context saves and restores the worker RNG state. Crop, flip,
and RandAugment consume that state while constructing individual samples. If the
CutMix gate and parameter draws advanced it permanently, all subsequent N1/M7
choices would differ from the accepted stream. Forking confines the new draws to
collation while still varying them batch to batch because the saved state after
each batch's per-sample transforms differs. No CUDA RNG is accessed in workers.

Extend the existing factory with an optional collator while retaining every
loader setting:

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

Use CutMix only for the initial strong loader:

```python
train_loader = make_train_loader(strong_train_tf, collate_fn=cutmix_collate)
```

Keep the weak loader construction unchanged so `collate_fn=None` restores
default collation and integer labels:

```python
train_loader = make_train_loader(weak_train_tf)
```

The current loss statement needs no branch:

```python
loss = F.cross_entropy(outputs, targets)
```

Mixed plateau batches carry floating probability targets `[B, 10]`; unselected
plateau batches and all tail batches carry integer targets `[B]`. Installed
`F.cross_entropy` supports both. Do not add a second CE call, manual lambda loss,
label smoothing, or custom soft-label kernel in this experiment.

For provenance, count formats before `t0` without touching tensors:

```python
strong_batch_count = 0
cutmix_batch_count = 0

# At the beginning of each yielded strong batch, before t0:
strong_batch_count += 1
cutmix_batch_count += int(targets.ndim == 2)
```

At the existing switch, print both counts and update the message to
`randaugment+cutmix->base`. This makes the realized probability and phase
boundary auditable without GPU synchronization. Stop counting after the phase
flag becomes false.

## Target and Loss Invariants

Preflight and execution checks must establish:

- every batch retains image shape `[128, 3, 32, 32]`;
- mixed targets are float `[128, 10]`, finite, nonnegative, and sum to one per
  row within floating-point tolerance;
- hard targets remain `torch.int64` `[128]`;
- for synthetic constant source images, donor target weight agrees with observed
  pasted-pixel fraction within `1 / 1024`;
- probability-target and hard-target CE both produce a finite scalar and finite
  gradients on width 2;
- after the 80% loader replacement, the first and every subsequent tail target is
  hard, with no prefetched mixed batch crossing the boundary.

Plateau loss is a stochastic mixture of soft-target and hard-target CE, while
tail loss is entirely hard-target CE. Its absolute EMA is not directly comparable
with EXP-007 or across the switch. The unchanged evaluator's hard-label test loss
and top-1 accuracy remain authoritative.

## RNG and Determinism Controls

Retain `torch.manual_seed(42)` and `torch.cuda.manual_seed(42)` exactly. Do not
reroll, add a user-selectable seed, or repeat a valid run. The CutMix gate is part
of the training method, not a source to select post hoc.

The collator's `fork_rng` strategy must pass two tests in a fresh worker process:

1. CPU RNG state before and after one collator call is byte-identical.
2. Consecutive collated batches do not all make the same gate decision or produce
   the same area target.

This preserves subsequent RandAugment draws as closely as the installed worker
model allows. It does not make EXP-007 and EXP-010 exactly paired: they are
separate multiprocessing/CUDA executions, CutMix changes inputs and gradients,
and the project stores no per-sample augmentation trace. A marginal result cannot
support a precise causal effect-size claim. The protocol decision is only whether
this fixed-seed operating point exceeds the moving baseline by the required
margin.

## Strong-to-Weak Transition

Preserve the accepted transition order and the 80% boundary exactly:

1. The crossing batch still uses the strong N1/M7 loader and the plateau LR.
2. Break the strong iterator immediately after that batch.
3. Perform at most the scheduled single evaluation for the epoch.
4. Stop and join all eight persistent strong workers.
5. Delete the strong loader and collect garbage.
6. Build the existing weak crop/flip loader with default collation.
7. Resume with hard targets and the unchanged `0.01` to `1e-4` cosine tail.

Do not switch CutMix independently from RandAugment. A single loader boundary
prevents prefetched probability targets or composite images from leaking into the
tail and preserves the lifecycle validated by EXP-004 and EXP-007.

## Mandatory Fixed-Time Preflights

Run all diagnostics in fresh disposable processes so they cannot consume the
eventual training RNG stream. A failed gate makes this candidate preflight-
infeasible; it does not authorize changing alpha, probability, loss, workers, or
placement within EXP-010.

### Worker Collate Gate

Benchmark the exact width-independent strong loader: crop, flip, N1/M7, the 0.5
CutMix collator, batch 128, eight forkserver workers, pinning, `drop_last`, and
persistence. Consume one warmup epoch, then time three complete epochs. Also run
the real shutdown and first weak-batch transition.

Require:

- exactly 390 batches per epoch;
- slowest warmed epoch at least 120 batches/s;
- no timed epoch more than 20% slower than their median;
- realized mixed-batch fraction across the timed epochs between 0.45 and 0.55;
- all eight strong workers terminate cleanly;
- transition plus first weak batch no more than 5 seconds; and
- the first weak target is hard integer `[128]`.

Width 2 consumed about 90.5 optimizer batches/s in EXP-007, while the historical
N1/M7 worker rate was 165.5-175.8 batches/s. The 120-batch/s gate leaves meaningful
prefetch headroom after adding half-batch CutMix collation. If it fails, moving
CutMix onto the GPU would change the fixed-time intervention and is not allowed.

### Probability-Target GPU Gate

On one idle H20 in a separate process, compare the exact accepted width-2 hard
step with a probability-target step. Include pinned-host H2D copies, forward,
`F.cross_entropy`, backward, SGD step, and `torch.cuda.synchronize()`. Use the
same batch shape and model state, warm both paths, and collect at least 500
interleaved observations so order and CUDA startup do not favor either path.

Let median synchronized times be `t_hard` and `t_soft`. Half the plateau batches
are soft, so average plateau step time is `(t_hard + t_soft) / 2`. Estimate
full-run update retention relative to EXP-007 as:

```text
predicted_step_ratio = 0.8 * (2 * t_hard / (t_hard + t_soft)) + 0.2
predicted_steps = 27_143 * predicted_step_ratio
```

Proceed only if:

- `predicted_step_ratio >= 0.97`;
- `predicted_steps >= 26_329`;
- the soft path has finite loss and gradients; and
- peak VRAM remains well below the H20 limit with no unexpected synchronization.

This caps predicted exposure loss at 3%, materially below EXP-003's 6.7% loss.
If probability-target CE misses the gate, do not substitute a custom sparse loss
after observing the measurement. That is a different implementation requiring a
new proposal.

### Functional Gate

On synthetic collated tensors, force both gate outcomes and verify image/target
shapes, area-label equality, loss acceptance, RNG restoration, and format counts.
Run at least 1,000 collator calls in a disposable process and verify the fixed
0.5 gate is statistically active rather than stuck. This is a correctness and
distribution check, not a seed sweep.

## Fixed-Time and Resource Feasibility

EXP-007 completed 27,143 steps, 71 epochs, 300.0 counted seconds, and 333.0 total
seconds with 598.7 MB peak VRAM. Passing the GPU gate predicts at least 26,329
steps. Passing the worker gate means host production remains faster than the GPU
consumer, so worker CutMix should not add large uncounted wall time. Fewer steps
may reduce completed epochs and dense-tail evaluator calls slightly.

Expected valid-run bands are approximately:

- 26,300-27,200 optimizer steps;
- 68-71 epochs, with a usable 20% hard tail;
- 300 seconds counted training;
- 333-370 seconds total; and
- peak VRAM close to the 598.7 MB width-2 baseline.

CutMix clones a roughly 1.5 MiB CPU batch in selected workers and probability
targets add only a few kilobytes per transfer. Host transient memory rises under
prefetch, but GPU model/image shapes are unchanged. The mandatory 600-second
supervisor remains authoritative even if all preflights pass.

## Testable Hypothesis and Expected Impact

**Hypothesis:** applying alpha-1 CutMix to 50% of N1/M7 batches during the 80%
high-LR plateau will improve partial-object and regional feature use without
discarding input information or sacrificing more than 3% of EXP-007's update
exposure. The accepted weak hard-label tail will then restore clean-image
BatchNorm statistics and classifier confidence, raising `best_test_acc` from
`93.55%` to at least `93.65%`.

A plausible successful range is `93.65-93.90%` (+0.10 to +0.35 points). This is
far smaller than the primary paper's gains because the current baseline already
uses strong RandAugment, the run has roughly 70 rather than 300 epochs, and only
half of plateau batches are mixed. The prediction is nevertheless testable and
large enough to meet the project gate.

Trajectory expectations are secondary diagnostics, not acceptance conditions:
the final strong checkpoint may be lower than EXP-007's `90.08%` because clean
evaluation follows composite probability-target training. If CutMix retains
useful information, the first weak checkpoint should recover near or above
EXP-007's `92.96%`, and the late tail should exceed `93.65%` rather than merely
continuing an unfinished ascent.

## Risks and Failure Modes

1. **Compounded regularization is too strong.** N1/M7 plus CutMix may underfit even
   width 2 within 240 seconds of high-LR training.
2. **Probability-target CE costs too many updates.** Worker-side images do not
   eliminate timed dense-target loss overhead.
3. **The hard tail is too short to remove mismatch.** Approximately 14-15 tail
   epochs may not fully resettle BatchNorm or sharpen hard-label boundaries.
4. **Area is an imperfect semantic label.** CIFAR objects are not uniformly
   distributed, so pixel fraction can misstate class evidence.
5. **Patch boundaries become shortcuts.** Pasting independently RandAugmented
   views can create artificial local edges or photometric discontinuities.
6. **Same-class or tiny patches dilute the method.** Rolled pairs can share labels,
   and clipped boxes can have little donor area.
7. **Variable target format perturbs throughput.** Mixed and hard batches take
   different CE paths and can add timing variance even if average cost passes.
8. **Worker collation starves prefetch.** CPU clones/copies can increase total
   wall time while remaining outside `training_seconds`.
9. **RNG isolation is incomplete.** A faulty collator could advance worker state
   and confound subsequent RandAugment choices.
10. **A marginal gain is run noise.** A 0.10-point threshold is ten test examples;
    one run establishes the protocol result, not a precise general law.
11. **Training loss is misinterpreted.** Mixed and hard losses have different
    targets and cannot be compared as a single fit measure.

## Confound Controls and Excluded Changes

Use accepted EXP-007 commit `8faf0f3` as the parent. Preserve:

- `WIDTH_MULTIPLIER=2`, all 1,073,962 parameters, post-activation blocks, Option-A
  shortcuts, initialization, and batch size 128;
- all-parameter SGD weight decay `1e-4`, momentum, LR values, and elapsed-time
  schedule;
- crop/flip/N1-M7 transform order and parameters;
- the exact 80% strong/LR boundary and weak hard-label tail;
- eight persistent forkserver workers, pinning, shuffle, and `drop_last`;
- seed 42, the one-run policy, evaluator, test preprocessing, checkpoints, and
  at-most-once-per-epoch validation;
- `prepare.py`, dependencies, and every tracked file except `train.py`.

Do not replace RandAugment, change width, alter decay groups/scalar, add label
smoothing or Mixup, add Cutout/RandomErasing, tune magnitude, use CutMix in the
tail, add test-time augmentation, or change evaluation density. Do not move the
mixing operation to the GPU, change alpha/probability after preflight, or implement
a custom loss. Each would add a second causal lever.

## Verification Procedure

Before execution:

1. Confirm `04-results.tsv` reports the moving baseline `93.55%` at `8faf0f3`.
2. Confirm the diff touches only `train.py` and contains only the declared CutMix
   import/configuration, collator, loader plumbing, phase counters, and provenance
   print. Review all model, optimizer, schedule, transform, and evaluator lines as
   unchanged.
3. Run compilation, Ruff/pre-commit, installed API, functional, RNG, worker, and
   throughput checks. Stop if any preflight gate fails.
4. Confirm exactly one idle NVIDIA H20 with approximately 98 GB VRAM is selected.
5. Confirm no stale completed `run.log` or renamed log variant remains.

Execute exactly one fixed-seed run with redirected output:

```bash
uv run train.py > run.log 2>&1
```

Monitor without streaming the full log and kill it at 600 seconds if it has not
exited. Do not retry a valid run or change the seed.

Post-run checks:

- `best_test_acc >= 93.65%` for improvement;
- one complete finite numeric summary, about 300 counted seconds, and fewer than
  600 total seconds;
- exactly 1,073,962 parameters and expected VRAM scale;
- exactly one `randaugment+cutmix->base` switch at 80.0%;
- all eight strong workers stopped before weak loading;
- strong-phase mixed fraction reasonably close to 0.5 and reported from target
  format counts;
- no probability target after the switch;
- no more than one evaluation per epoch and terminal evaluation matching the
  summary epoch;
- step count and runtime explicitly compared with EXP-007's 27,143 and 333.0s.

Classify outcomes as:

- **Improvement:** `best_test_acc >= 93.65%` and every integrity check passes.
- **No improvement:** a valid run below `93.65%`; do not rerun or tune CutMix.
- **Preflight-infeasible:** any RNG/target/lifecycle failure, worker throughput
  below 120 batches/s, or predicted step ratio below 0.97; skip full training.
- **Failure:** crash, timeout, wrong hardware, invalid summary, scope violation,
  incorrect phase switch, soft-target leakage, duplicate epoch evaluation, or
  counted-budget violation.

If successful, accept the complete conservative CutMix composition on top of
EXP-007. If unsuccessful with healthy throughput, revert the CutMix additions and
retain EXP-007; that would indicate that class-bearing regional mixing still does
not repay compounded augmentation and finite-horizon convergence cost, rather
than rehabilitating EXP-006's fixed information-deleting Cutout.

## Evidence

- `experiments/004/04-analysis.md`: validated N1/M7 plateau plus weak tail.
- `experiments/006/04-analysis.md`: fixed 25% Cutout information-loss failure.
- `experiments/007/04-analysis.md`: accepted width-2 frontier and exposure.
- `experiments/009/04-analysis.md`: decay relaxation fit/generalization result.
- `knowledge/papers/cutmix.md`: ICCV CutMix mechanism, evidence, and installed API
  relevance.
