# Proposal idea-01: Mild policy-based augmentation (RandAugment / TrivialAugment), REPLACEMENT design

**Refined after cross-model review** (`01-idea-review.md` picked this idea; concerns folded in below).

## Core change
Insert a torchvision policy-augmentation transform into `train_tf` (train.py:205) between the PIL geometric augs and `ToTensor`:
```
RandomCrop(32, padding=4)         # PIL
RandomHorizontalFlip()            # PIL
<POLICY AUG HERE>                 # PIL/uint8: RandAugment | TrivialAugmentWide | AutoAugment(CIFAR10)
ToTensor()
Normalize(EVAL_MEAN, EVAL_STD)
Cutout(12)                        # tensor
RandomErasing(...)               # tensor  <-- candidate to REPLACE with policy aug
```
torchvision 0.24.1 ships `RandAugment(num_ops, magnitude, num_magnitude_bins=31)`, `TrivialAugmentWide()`, `AutoAugment(AutoAugmentPolicy.CIFAR10)` — **no new dependency**. All consume PIL images (CIFAR10 returns PIL), so placement before `ToTensor` is correct.

## Mechanism / why it can move the ceiling
Geometric (rotate, shear, translate-X/Y) + photometric (contrast, brightness, color, sharpness, posterize, solarize, equalize, auto-contrast) transforms = a DIFFERENT augmentation mechanism than the occlusion (Cutout/RandomErasing, EXP-008 won) and mixing (CutMix, EXP-011 tied) classes already tried. A generalization ceiling on fixed data is canonically raised by increasing effective data diversity; policy aug is THE documented lever taking CIFAR-10 ResNets from ~96→97%+ (TrivialAugment ICCV'21; RandAugment CVPRW'20).

## Design — SAME-SESSION multi-cell, MILD + REPLACEMENT (folds review concerns #2,#3)
Env-toggled cells, each a separate `train.py` process under `timeout 600`, all same session:
- **c0 — baseline control**: current recipe (Cutout12 + RandomErasing). Re-measured this session (the stored 96.38 is too weak at the ~0.1pp noise floor).
- **cA — RandAugment REPLACES RandomErasing, mild**: `RandAugment(num_ops=1, magnitude=6)` before ToTensor; KEEP Cutout12; DROP RandomErasing. Swaps one occlusion lever for the richer transform lever (not a 3rd stacked aug).
- **cB — one conservative variant**: `TrivialAugmentWide()` replacing RandomErasing (parameter-free, single-op; slightly different strength profile than cA), KEEP Cutout12.

Rationale: review concern #2 — do NOT just add strong policy aug on top of the full occlusion stack (over-regularizes a 150-epoch budget). Replacement keeps total regularization load near the tuned operating point. Start mild (N=1, M=6 ≪ RandAugment's CIFAR default N=2,M=14) because the budget is ~150ep, far short of the 200–2000ep canonical recipes.

## Dropped: curriculum tail-off (review concern #3)
`DataLoader(persistent_workers=True)` (train.py:226) caches the dataset+transform in each worker at first iteration; mutating `train_set.transform` from the main process mid-training would NOT reach worker copies → a tail-off curriculum would silently no-op (or require an expensive DataLoader rebuild that risks throughput). REMOVED in favor of **fixed mild strength** — cleaner single-variable test, no worker-visibility hazard.

## Throughput / wall-time guard (review concern #4)
Policy aug is CPU-worker work BEFORE ToTensor. Budget is COMPUTE-time (per-step `dt`); dataloader wait is OFF that timer, so a CPU bottleneck inflates WALL `total_seconds` toward the 600s cap rather than cutting `num_epochs`.
- PRE-SMOKE: a short run measuring img/s + first-epoch wall with cA's transform to confirm workers keep up.
- VERDICT METRICS: record `num_epochs` (must stay ~142–155) AND `total_seconds` (must stay well under 600s wall) for every cell.

## Verification (folds review concern #1,#5 — require a real effect)
- Best cell must beat the SAME-SESSION c0 by **>0.1pp** (noise floor), not the stored baseline.
- Under-fit watch: `ep25` test_acc within ~0.5pp of c0's ep25, and best NOT == final-still-climbing (would indicate the harder task didn't converge in budget).
- If all policy cells tie-or-lose c0 with healthy num_epochs and normal ep25 → input-augmentation is confirmed saturated across ALL THREE mechanisms (occlusion/mixing/transform); the ceiling is not augmentation-movable.

## Hypothesis
A mild RandAugment(1,6) (or TrivialAugment) REPLACING RandomErasing adds geometric+photometric diversity orthogonal to occlusion and lifts best_test_acc clearly above the same-session baseline (>0.1pp) without under-fitting (ep25 ≈ baseline; num_epochs in band; wall < 600s).

## Effort: low. Sources: TrivialAugment ICCV'21, RandAugment CVPRW'20, Raschka benchmark; EXP-008/011 learnings; `knowledge/references/mixing-augmentation.md`, `fast-cifar10-recipes.md`.
