# Proposal: One-Operation Conservative RandAugment

## Summary

Add one conservative worker-side RandAugment operation to the accepted EXP-002
training pipeline while preserving every other training choice. Use
`transforms.RandAugment(num_ops=1, magnitude=7)` after the existing random crop
and horizontal flip and before `ToTensor`. Keep the EXP-002 hard-label loss,
80%-hold elapsed-time learning-rate schedule, standard momentum, model, loader,
seed, and evaluation cadence unchanged.

The proposal targets the remaining generalization gap through input invariance
rather than another target-side regularizer. EXP-003's label smoothing reduced
test NLL but did not improve top-1 accuracy, and its built-in loss path reduced
fixed-budget steps by 6.7%. RandAugment instead runs in the eight DataLoader
workers before the batch is yielded. Its CPU/PIL cost is outside the synchronized
GPU step timer, so it can preserve optimizer exposure if prefetch keeps pace. The
tradeoff is that slow transforms can starve the GPU and increase total wall time
even though `total_training_time` still records 300 seconds. A mandatory loader
throughput preflight therefore gates the full experiment.

## Diagnosis and Mechanism

EXP-002 improved the moving baseline to `91.83%` by retaining a long high-rate
exploration phase and guaranteeing terminal low-rate refinement. EXP-003 kept
that schedule and softened only the targets. It lowered fixed-evaluator test
loss from `0.2843` to `0.2740` but left best top-1 at `91.83%`, while completing
36,039 rather than 38,629 steps. This suggests that confidence regularization
alone is not the best next lever and reinforces that throughput is part of the
statistical experiment under a fixed compute-time horizon.

RandAugment randomly selects image operations from torchvision's predefined
space and applies them at a shared magnitude. It exposes the network to mild
geometric, photometric, contrast, and quantization perturbations, encouraging
features that are invariant to nuisance variation while retaining hard class
targets. This is mechanistically distinct from label smoothing and Mixup: the
label remains exact, and regularization comes from a richer local neighborhood
around each observed image.

One operation at magnitude 7 is intentionally below torchvision's default policy
of two operations at magnitude 9. On the 31-bin magnitude scale, this setting is
strong enough to create meaningful input variation but limits compounding and
destructive transforms on 32x32 images. For representative geometric operations,
the setting corresponds to roughly 7 degrees of rotation, 0.07 shear, or about
3.4 pixels of translation; only one is selected per image.

## Exact Proposed Change

Change only the training transform composition in `train.py`:

```python
train_tf = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=1, magnitude=7),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)
```

The order is deliberate:

1. `RandomCrop(32, padding=4)` retains the accepted spatial crop policy.
2. `RandomHorizontalFlip()` retains the accepted class-preserving reflection.
3. `RandAugment(num_ops=1, magnitude=7)` operates on the final 32x32 PIL image.
4. `ToTensor()` and `Normalize(mean, std)` remain last and unchanged.

Placing RandAugment after crop/flip means the current image's established crop
and flip decisions are made before RandAugment consumes its worker RNG draws.
Placing it before `ToTensor` uses the transform's PIL path and avoids adding a
GPU-side intervention. Use the installed torchvision 0.24.1 defaults for all
unspecified arguments: `num_magnitude_bins=31`, nearest-neighbor interpolation,
and default fill. Do not add custom operation weights, remove operations from the
space, set a separate application probability, or tune interpolation/fill in
this experiment.

No new top-level hyperparameter is necessary because this is a single fixed
candidate. If readability conventions favor constants, `RANDAUGMENT_NUM_OPS = 1`
and `RANDAUGMENT_MAGNITUDE = 7` are acceptable, but they must feed exactly the
constructor above and must not enable an in-run sweep.

## Evidence

Cubuk et al., *RandAugment: Practical Automated Data Augmentation with a Reduced
Search Space* (NeurIPS 2020), replaces expensive augmentation-policy search with
two interpretable parameters: number of operations and shared magnitude. It
reports strong CIFAR-10 results and supports composing a small fixed number of
random transforms with standard crop/flip augmentation. The goal knowledge-base
distillation is at `knowledge/papers/randaugment.md`.

The local history supplies two additional constraints:

- EXP-002's 80% high-LR plateau plus low-LR cosine refinement raised accuracy by
  0.16 points and is the accepted optimizer baseline; it must remain fixed.
- EXP-003 showed that an apparently cheap regularizer can reduce step exposure by
  6.7%. RandAugment must therefore pass a measured throughput check rather than
  being assumed free merely because it runs on the host.

## Hypothesis and Expected Impact

The testable hypothesis is that one-operation magnitude-7 RandAugment will add
useful input-space invariance without materially reducing the 38,629-step
optimization exposure of EXP-002, raising `best_test_acc` from `91.83%` to at
least `91.93%`. A plausible result range is `91.95-92.25%`, with most of the gain
expected to emerge during the existing 20% low-rate refinement tail.

The expected benefit is modest because the policy is intentionally conservative
and ResNet-20 is small. Its advantage over label smoothing is not necessarily a
larger regularization strength; it is better alignment with top-1 error modes and
moving the added work outside the synchronized GPU step interval.

## Mandatory Preflight Throughput Diagnostic

Before the full training run, benchmark DataLoader production in a fresh Python
process without constructing or training the model. The diagnostic must use the
same CIFAR-10 training set, batch size 128, shuffle, eight workers, pinned memory,
`drop_last=True`, and `persistent_workers=True` as `train.py`. Run two separate
fresh processes:

1. **Control loader:** existing crop, flip, tensor conversion, and normalization.
2. **Candidate loader:** the identical pipeline with
   `RandAugment(num_ops=1, magnitude=7)` at the proposed position.

For each process, set the same seed, record first-batch latency, consume one full
epoch as worker/cache warmup, then time three complete epochs. Record each timed
epoch's batch rate and image rate, not only their aggregate. The benchmark must
consume batches without GPU copies or synthetic sleeps. Running it in a separate
process ensures its RNG consumption cannot affect the later fixed-seed training
run.

Use the slowest of the three warm timed candidate epochs as the conservative
loader rate `R_candidate`. Estimate the full-run wall time as:

```text
projected_total_seconds =
    336.0 + max(0, 38_629 / R_candidate - 300.0)
```

Here `336.0s` and 38,629 steps are the accepted EXP-002 total runtime and step
count. The formula treats GPU compute and worker production as ideally overlapped
and charges any loader time beyond the 300-second GPU path as added wall time.
It is only a feasibility estimate, so it deliberately requires margin.

Proceed to the full experiment only if all of the following hold:

- `R_candidate >= 80 batches/s` (at least 10,240 images/s);
- the slowest timed epoch is no more than 20% slower than the candidate median,
  ruling out unstable starvation;
- `projected_total_seconds <= 540s`, leaving at least 60 seconds before the hard
  timeout for first-batch latency, evaluator variation, and imperfect overlap;
- the candidate process exits cleanly with exactly 390 batches per full epoch.

At 80 batches/s, producing 38,629 batches takes about 483 seconds; the formula
projects roughly 519 seconds total, still leaving about 81 seconds before the
600-second limit. If the candidate misses any gate, do not launch the full run.
Record RandAugment at this operating point as preflight-infeasible and return to
a cheaper input transform or a revised worker-throughput experiment. Do not
increase `NUM_WORKERS`, reduce evaluation, or alter the policy in the same
experiment to force it through the gate.

The control measurement is diagnostic rather than a second training baseline.
Its candidate/control ratio identifies transform cost, while the absolute gate
determines wall-limit feasibility. The full training process must restart fresh
with seed 42 after the preflight.

## Runtime and Resource Implications

RandAugment executes when CIFAR-10 samples are transformed in DataLoader workers.
The current training timer starts only after `for inputs, targets in train_loader`
has yielded a batch, so transform and queue-wait time is not included in the
synchronized per-step `dt`. If workers stay ahead of the H20, optimizer step count
should remain near EXP-002's 38,629 and avoid EXP-003's loss-path penalty. If they
fall behind, the GPU waits between timed steps: `training_seconds` can still be
300 while `total_seconds` grows substantially.

The transform adds no model parameters, no material GPU memory, and no extra
evaluation. Worker CPU time and transient PIL image allocations increase. Eight
persistent workers and the bounded EXP-002 evaluation scheme must remain in
place. Since EXP-002 finished at 336 seconds, a candidate that passes the 540-
second projection has meaningful headroom under the mandatory 600-second
supervisor.

## Failure Modes and Risks

1. **Worker starvation.** PIL transforms may not keep the prefetch queue full for
   a tiny ResNet-20 on H20, extending total runtime despite unchanged counted
   training time.
2. **Policy too weak.** One magnitude-7 operation may not produce enough new
   invariance to move top-1 by the required 0.10 points.
3. **Policy too destructive.** Even one posterize, solarize, geometric, or
   contrast operation can damage a 32x32 example or introduce default-fill and
   nearest-interpolation artifacts.
4. **Optimization underfits stronger examples.** The accepted 300-second horizon
   may be enough for clean crop/flip samples but not for the wider augmented
   distribution, particularly through the long high-LR plateau.
5. **BatchNorm distribution shift.** BatchNorm statistics are learned from
   augmented images while evaluation uses clean images. Conservative magnitude
   and one operation limit but do not eliminate this mismatch.
6. **RNG sequence changes.** RandAugment necessarily consumes worker RNG and
   changes later crop/flip draws. This is part of the fixed augmentation method,
   not permission to reroll seeds; the run remains single-seed and single-trial.
7. **Preflight optimism.** A loader-only benchmark may overestimate throughput
   under concurrent main-process and GPU activity. The 60-second projection
   margin is intended to absorb this gap, but the 600-second timeout still governs.
8. **Test loss may worsen while top-1 improves.** Unlike smoothing, stronger
   hard-target augmentation does not directly optimize calibration. The declared
   metric remains best accuracy, not NLL.

## Confound Controls and Excluded Interventions

Treat the accepted EXP-002 `train.py` as the moving baseline. The only training
change is the single RandAugment constructor inserted into `train_tf`. Preserve:

- hard-label `F.cross_entropy(outputs, targets)` with no label smoothing, Mixup,
  CutMix, distillation, or auxiliary loss;
- `LR=0.1`, `ANNEAL_START_LR=0.01`, `MIN_LR=1e-4`,
  `LR_HOLD_FRACTION=0.8`, ordinary momentum, weight decay, and elapsed-time
  schedule math;
- ResNet-20 architecture, initialization, batch size 128, and all model code;
- existing crop, horizontal flip, normalization, worker count, persistent
  workers, pinning, shuffling, and `drop_last` behavior;
- seed 42, one-run policy, clean test transform, fixed evaluator, checkpoint list,
  dense-tail evaluation, and once-per-epoch maximum;
- dependency files, `prepare.py`, and every tracked file except `train.py`.

Do not stack RandAugment with the failed EXP-003 smoothing setting. Do not add a
probabilistic gate, magnitude schedule, operation blacklist, custom fill,
bilinear interpolation, random erasing, or extra worker tuning. Any one of those
may be reasonable later, but each would prevent EXP-004 from answering whether
the standard one-operation policy improves the accepted schedule.

## Full-Run Verification

After the preflight passes:

1. Confirm the current moving baseline is `91.83%` at commit `5016cc4` and the
   diff touches only `train.py` with the single transform insertion.
2. Run static compilation, Ruff/pre-commit, and a constructor/API smoke check.
3. Confirm exactly one idle NVIDIA H20 with approximately 98 GB VRAM is visible.
4. Confirm no stale completed `run.log` or renamed run-log variant remains.
5. Launch exactly one fixed-seed run with all output redirected:

```bash
uv run train.py > run.log 2>&1
```

Monitor without streaming the full log and kill the process at 600 seconds if it
has not exited. A valid run must complete normally, print one finite numeric
summary, record approximately 300 seconds of counted training, and evaluate no
more than once in any epoch.

Decision criteria:

- **Improvement:** `best_test_acc >= 91.93%` with all integrity and runtime checks
  passing.
- **No improvement:** valid completion below `91.93%`; do not rerun with another
  seed or magnitude.
- **Preflight-infeasible:** candidate loader fails any throughput gate; skip the
  full run and record the measured blocker.
- **Failure:** crash, timeout, wrong hardware, scope violation, changed evaluator,
  duplicate evaluation within an epoch, missing/nonfinite summary, or counted
  training-budget violation.

Record accuracy, test loss, counted and total seconds, epochs, optimizer steps,
peak VRAM, plus the control and candidate preflight rates. Compare step count
against EXP-002's 38,629 and total time against 336.0 seconds. If step count falls
substantially despite host-side placement, inspect whether H2D/copy timing or
system contention changed; do not attribute the entire accuracy change to
RandAugment without reporting that confound.

## Follow-Up Interpretation

- If accuracy improves, retain the exact one-operation magnitude-7 policy as a
  validated component. Any move to two operations, magnitude tuning, custom
  interpolation, or composition with other regularization is a later experiment.
- If top-1 is unchanged but throughput is healthy, the policy is statistically
  ineffective at this strength; a stronger magnitude is possible but should be
  separately justified rather than silently retried.
- If accuracy regresses with healthy throughput, revert RandAugment and interpret
  the result as excessive or misaligned invariance for this model/horizon.
- If the preflight or full run is wall-time limited, retain EXP-002 and pursue a
  cheaper tensor transform or explicit loader optimization in a separately
  scoped experiment.
