# Proposal: CIFAR-Mean Constant Fill for Random Crops

## Recommendation

Test one exact, low-cost input-distribution change: retain the accepted
four-pixel constant-padded `RandomCrop`, but replace its raw-black RGB fill
with the quantized CIFAR mean `(125, 123, 114)`. Preserve the crop size,
padding width and mode, transform order, crop/flip draws, accepted early-only
`EarlyRandAugment`, batch-shared alpha-0.2 mixup through 65%, model, pooled
head, optimizer, schedules, seed, time budget, and evaluation cadence.

This is not an immediate reflection-padding rescue. EXP032 changed the
boundary geometry and entered torchvision's NumPy reflection path; it was not
scored because candidate active-loader CV reached 11.20%. The proposed change
keeps the accepted constant-padding implementation and changes only the value
assigned to synthetic crop-border pixels. It has a distinct normalization
mechanism and should use the same PIL `ImageOps.expand` path as accepted. Its
prior is plausible but modest: mean fill removes a large normalized black
artifact, yet the accepted black border may itself provide useful occlusion,
position, or contrast regularization. One strictly qualified score is
warranted; a fill-value or schedule sweep is not.

## Exact Production Change

Change exactly one transform construction in `make_train_transform`:

```python
operations = [
    transforms.RandomCrop(32, padding=4, fill=(125, 123, 114)),
    transforms.RandomHorizontalFlip(),
]
```

Do not change `padding_mode`; its accepted default remains `"constant"`.
Do not move the crop relative to flip or RandAugment, convert images to tensors
before cropping, share a new constant through unrelated code, modify
`EarlyRandAugment.transform.fill`, change normalization, or gate the fill by
the RandAugment/mixup phase. In particular, leave the accepted RandAugment
argument text `fill=[125, 123, 114]` untouched. Tuple versus list is accepted
by the installed PIL transform path and does not motivate production cleanup.

The integer values are fixed by the accepted normalization constants:

```text
round(255 * (0.4914, 0.4822, 0.4465)) = (125, 123, 114).
```

After `ToTensor()` and the accepted unit-standard-deviation normalization,
these pad pixels become approximately

```text
(125/255 - 0.4914, 123/255 - 0.4822, 114/255 - 0.4465)
= (-0.001204, +0.000153, +0.000559),
```

rather than the black-fill vector `(-0.4914, -0.4822, -0.4465)`. The proposal
therefore means *quantized near-zero normalized fill*, not an assertion of
mathematically exact zero.

## Mechanism and Limiter Diagnosis

The accepted learner nearly interpolates its training tail while finishing at
94.48% best accuracy and 0.2456 test loss, so system understanding identifies
generalization and decision-boundary quality, not I/O or memory, as limiting.
Backpropagation consumes about 74% of a complete step, and the accepted run
delivers only 130.304 data passes, favoring input or post-pooling changes that
add no spatial GPU work.

The crop samples each top/left coordinate uniformly from `0..8`. Only the
center `(4,4)` window avoids padding, so synthetic border pixels are contacted
with probability `80/81 = 98.7654%` (EXP032 measured 98.8020%). Under the same
uniform coordinate law, expected retained source width per axis is
`32 - E|offset-4| = 268/9`, making the expected synthetic-pixel share
approximately `1 - (268/9)^2 / 32^2 = 13.41%` of every crop. The accepted
pipeline therefore exposes the model to a frequent, substantial region whose
normalized value is about half a unit below the dataset center in every
channel.

Mean fill removes that high-magnitude artificial color while preserving the
translation/occlusion geometry. It may reduce train/test boundary-statistic
shift, prevent the network from using a black frame as a crop-offset cue, and
make early transformed examples more consistent with the accepted
RandAugment out-of-bounds fill. This targets input invariance and
generalization without adding parameters, kernels, backward work, or GPU RNG.
It composes cleanly with the accepted EXP027 interaction: the extra low-
resolution block remains fully trained, while EXP026's worker-isolated
RandAugment decisions remain exactly the same.

The countermechanism is equally concrete. Constant mean is still an
unnatural, textureless border. Black padding can act as useful structured
occlusion, enhance object/background contrast, or teach robustness to dark
regions; neutralizing it can weaken regularization or erase useful boundary
evidence. Early RandAugment operates after cropping, so geometric or
photometric operations can move, interpolate, or recolor the changed pixels;
the treatment is not confined to a static border during the first 65%.

## State, RNG, and Throughput Semantics

The installed torchvision `RandomCrop.forward` first calls `F.pad`, then
draws exactly two `torch.randint` values for `(i,j)`. For PIL RGB input and
constant mode, both scalar-black and tuple-mean fill call PIL
`ImageOps.expand`; padding itself draws no random numbers and leaves the padded
shape at `40x40`. Consequently the candidate must produce the exact accepted
crop coordinates, flip bit, terminal worker torch RNG state, subsequent
sample decisions, sampler order, and targets from independently restored
states.

`EarlyRandAugment` receives different pixels but must receive the exact same
worker-private pre-state, choose the same operation/sign/magnitude parameters,
reach the same private post-state, and restore the same accepted worker RNG in
`finally`. When its shared flag is inactive it must remain a byte-exact RNG
no-op. Because RandAugment can geometrically spread or photometrically alter
pad values, post-RandAugment pixel differences must not be required to remain
inside the original padding mask; decision and state identity are the valid
invariant.

The main-process model and optimizer construction do not depend on image
values. The candidate must retain all accepted parameter/buffer names, orders,
shapes, and initial bytes; 52 trainable tensors; exactly 1,003,482 parameters;
both optimizer groups and settings; pooled-head seed 36036; and identical
post-construction CPU/CUDA RNG. Batch-shared mixup must use the same beta and
permutation RNG decisions from cloned states. Model activations, gradients,
updates, and logits are intentionally allowed to differ because the input
pixels are the treatment.

The GPU graph, tensor shapes, batch size, and training loop are source-
identical, so counted-step exposure should remain in the accepted regime.
Worker delivery still affects total wall time because loader waiting occurs
before the scored per-step timer. Unlike reflection, tuple constant fill does
not invoke `np.asarray`, `np.pad`, or `Image.fromarray`; it merely changes the
fill argument to the same PIL primitive. This strongly predicts negligible
CPU overhead, but the EXP032 failure requires measuring active and inactive
persistent-worker stability before scoring rather than inferring it.

## Fail-Closed Semantic Preflight

Use an ignored evaluator-free harness. Print every measured payload before
asserting its gate, and require all of the following:

1. Diff accepted `a7c42dc:train.py` against production and prove the only
   intentional source change is `fill=(125, 123, 114)` on the existing
   `RandomCrop`. Require `prepare.py` and the evaluator unchanged, syntax clean,
   local CIFAR present, and no test-set construction or evaluator call by the
   verifier.
2. Inspect the production transform: exact order must remain crop -> flip ->
   `EarlyRandAugment` -> tensor -> normalize; crop size `(32,32)`, padding `4`,
   fill `(125,123,114)`, `padding_mode="constant"`, and
   `pad_if_needed=False`; all RandAugment fields and normalization values must
   equal accepted.
3. On deterministic asymmetric RGB PIL fixtures whose source pixels are
   neither black nor the fill color, independently restore one torch RNG state
   before accepted crop+flip, candidate crop+flip, and manual replay. Require
   the same two `torch.randint` crop draws, same flip draw, same terminal state,
   and exact agreement with independent NumPy constant-padding/crop/flip
   oracles.
4. Exhaust all 81 crop offsets with both flip decisions. Track a flipped
   padding-derived mask and require original-image pixels to remain byte-exact,
   all accepted/candidate differences to lie inside the mask, candidate mask
   bytes to equal `(125,123,114)`, accepted mask bytes to equal `(0,0,0)`, and
   a nonzero intended difference for every touching offset. Separately verify
   the post-normalization vectors above within FP32 tolerance.
5. Sample at least 100,000 independently seeded crop-coordinate pairs, print
   incidence first, and require padding contact within a predeclared narrow
   bound around `80/81`; this checks the treatment is actually exercised
   without reading labels or test data.
6. From independently restored main and private RNG states, run at least 64
   accepted/candidate `EarlyRandAugment` calls and independently decode its
   installed operation/sign draw sequence. Require equal decisions,
   magnitudes, private post-states, and restored main states. Then disable the
   flag and require exact no-advance behavior. Do not impose a false
   post-RandAugment pixel-confinement check.
7. Trace fresh accepted-like and candidate real CIFAR training loaders in
   production construction order with eight persistent forkserver workers.
   Exhaust one active epoch, flip the shared flag only after exhaustion, and
   sample the next inactive epoch. Require every sample index, worker id,
   target, crop/flip decision, active bit, decoded RandAugment decision,
   private-state hash, sampler order, and terminal main RNG to match. Require
   exact manual transform-oracle agreement per arm and no prefetched active
   sample after cutoff.
8. Instantiate accepted and candidate models/optimizers from cloned seed-42
   states with `Eval` blocked. Require exact state dict, optimizer grouping,
   constants, parameter count, post-construction CPU/CUDA RNG, learning-rate
   samples, mixup draws, one backward/update's RNG accounting, finite guard,
   transition ordering, and every-fifth-plus-final evaluation contract.

A semantic failure aborts before timing or scoring. Repair only an
independently demonstrated production or verifier defect. Do not change fill
values, padding width/mode, transform order, phase duration, or accepted
augmentations as a repair.

## Loader and GPU Feasibility Gate

On one idle H20 and otherwise idle host, measure accepted-like and candidate
real loaders separately so only one live worker pool exists at a time. Cover
both active and inactive RandAugment phases in a counterbalanced order, use
one unretained warm epoch and at least six complete retained 195-batch epochs
per arm/phase, tear workers down before restoring seed-42 construction state,
and use the same fixed consumer delay. Emit raw epoch windows, medians, CVs,
batch counts, shapes, weighted values, and wall projections before assertions.

Require exactly 49,920 finite training examples per retained epoch, every
window CV `<=5%`, candidate active and inactive medians each no more than
`1.05x` their accepted counterparts, no candidate epoch more than `1.10x`
the matching accepted median, and

```text
weighted = 0.65 * active_median + 0.35 * inactive_median
projected_total_wall =
    accepted_total_wall
    + max(0, candidate_weighted - accepted_weighted)
      * projected_complete_epochs
```

below 500 seconds using the accepted 130.304-pass workload. The constant-fill
change cannot alter the source-identical counted GPU body, but also run a
short paired complete-step smoke check in early-mixup and hard regimes to
detect resource errors, require finite forward/backward/update, identical
shapes and allocation below 2,048 MiB, and at least 127 projected passes. A
stable timing miss closes systems viability without an accuracy claim and
must not be rerun or rescued with another fill in the same experiment.

## Sole Score, Decision Rule, and Closure

After all gates pass, reconfirm baseline 94.48% at `a7c42dc`, one idle NVIDIA
H20, local CIFAR-10, frozen `prepare.py`/evaluator, exact `train.py`-only scope,
and no stale `run.log`. Launch exactly once:

```bash
timeout 600s uv run train.py > run.log 2>&1
```

Require exit zero, exactly one finite summary, 300.0-300.1 counted seconds,
total wall below 600 seconds, 1,003,482 parameters, no worker/OOM/non-finite
signature, one ordered mixup transition, one later exhausted-iterator
RandAugment transition, and unique every-fifth plus final evaluations. Compute
realized exposure as `num_steps * 256 / 50000`; a completed low-exposure run
still consumes the sole score and cannot be rerun.

Success requires `best_test_acc >=94.58%` and realized exposure `>=127` passes.
Final accuracy versus 94.45% and final loss versus 0.2456 are descriptive
corroboration only. A valid normal-exposure miss closes this exact
always-on `(125,123,114)` constant crop fill. Do not rescue it with floor or
ceiling quantization, scalar gray, dataset-sampled color, noise, edge/reflect/
symmetric padding, a fill blend, a padding-width change, tensor-side padding,
phase gating, a RandAugment-dependent fill, changed transform order, another
seed, or a second score. A success supports only the fixed fill treatment; it
does not authorize fill-value tuning. The result does not close a
fundamentally different boundary augmentation supported by a new diagnosis.

## Risks

- **Accuracy risk, medium-high:** black padding may be beneficial structured
  occlusion or contrast regularization; neutral fill can reduce diversity.
- **Interaction risk, medium:** early RandAugment can move or recolor changed
  pad pixels, so this is a coupled input-distribution effect even though its
  decisions and RNG are preserved.
- **Systems risk, low but gated:** constant tuple fill should retain the PIL
  path, yet EXP032 shows CPU transform feasibility must be measured.
- **Interpretation risk, low:** raw mean quantization is near normalized zero,
  not exact zero; the proposal fixes the bytes and reports the residual.
- **Search risk, controlled:** one seed and one score prevent result-conditioned
  fill tuning; strict closure prevents adjacent-value rescues.

## Falsifiable Hypothesis and Sources

If the frequent high-magnitude normalized black crop border is a material
source of train/test boundary-statistic mismatch, then replacing only that
border with quantized CIFAR-mean RGB while preserving every stochastic
decision will retain at least 127 passes and raise fixed-seed
`best_test_acc` from 94.48% to at least 94.58%. A valid normal-exposure miss
falsifies that exact claim.

Offline sources: accepted `train.py` at `a7c42dc`; goal
`01-definition.md`, `02-system-understanding.md`,
`03-experiment-learnings.md`, and `04-results.tsv`; installed torchvision
`RandomCrop`, functional PIL padding, and `EarlyRandAugment` implementations;
EXP026/027 early-invariance reports; EXP032 reflection-padding plan, execution,
and analysis; EXP036 accepted pooled-head result; local RandAugment and
time-varying-regularization knowledge notes. No network, test data, evaluator,
GitHub, or remote source was used.
