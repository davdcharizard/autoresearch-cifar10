# Policy-based augmentation for CIFAR-10 (AutoAugment / RandAugment / TrivialAugment)

Standing reference for transform-based ("policy") data augmentation — the third aug MECHANISM, distinct from occlusion (Cutout/RandomErasing) and mixing (CutMix/mixup) already tested on this goal.

## The variants (all in torchvision 0.24.1 — NO new dependency)
- **AutoAugment** (Cubuk et al., CVPR 2019, arXiv:1805.09501): RL-searched policy of (op, prob, magnitude) sub-policies. `torchvision.transforms.AutoAugment(AutoAugmentPolicy.CIFAR10)` ships the published CIFAR-10 policy.
- **RandAugment** (Cubuk et al., CVPRW 2020, arXiv:1909.13719): drops the search; 2 hyperparams — `num_ops` N (apply N random ops/image) and `magnitude` M (0–`num_magnitude_bins`, default bins=31). torchvision default `RandAugment(num_ops=2, magnitude=9)`. CIFAR/WRN paper default ≈ N=2–3, M=14.
- **TrivialAugmentWide** (Müller & Hutter, ICCV 2021, arXiv:2103.10158): PARAMETER-FREE — one uniformly-sampled op at one uniformly-sampled magnitude per image. Matches AutoAugment/RandAugment with zero tuning. "Wide" = aggressive magnitude range.

Ops pool (geometric + photometric): rotate, shear-X/Y, translate-X/Y, contrast, brightness, color, sharpness, posterize, solarize, equalize, auto-contrast, invert. All operate on **PIL images or uint8 tensors** → must be placed BEFORE `ToTensor`/`Normalize` in the transform pipeline (after RandomCrop/Flip is fine; CIFAR10 dataset yields PIL).

## Mechanism vs the aug classes already tried here
Occlusion (Cutout/RE) deletes info; mixing (CutMix) pastes cross-class regions w/ soft labels. Policy aug instead *transforms* the whole image (geometry + photometrics) → increases effective data diversity along axes the other two don't touch. This is THE canonical lever taking CIFAR-10 ResNets ~96→97%+.

## THE load-bearing caveat for a fixed-TIME / ~150-epoch budget
Headline gains (AutoAugment ≈ +12% over NO aug; RandAugment +~2pp more — Raschka ResNet-18 benchmark) are (a) over a no-aug baseline, and (b) measured at **200–2000 epochs**. Strong policy aug makes the task HARDER and needs many epochs to converge (validation-acc slope still positive at 1000ep). The increment OVER a flips+pad-crop+**Cutout** baseline (what we already have) is far smaller — RandAugment paper: "competitive (within 0.1%) or SOTA across four architectures." Fast recipes (airbench ~10–40ep) deliberately AVOID strong aug for this reason. We run ~150ep — long enough to absorb SOME policy aug, but mild magnitude + replacement (not stacking on the full occlusion stack) is essential to avoid under-fit.

## Implementation notes for THIS harness (constraints)
- **Placement**: insert into `train_tf` between `RandomHorizontalFlip()` and `ToTensor()` (train.py:205). PIL in → PIL out.
- **persistent_workers TRAP**: `DataLoader(persistent_workers=True)` caches dataset+transform per worker at first iteration. Mutating `train_set.transform` mid-training from the main process does NOT propagate to workers → any curriculum/tail-off schedule via attribute mutation silently no-ops. Use FIXED strength, or rebuild the DataLoader (throughput cost), or a worker-shared schedule object.
- **Throughput**: CPU-worker work BEFORE ToTensor. Budget is COMPUTE-time (per-step dt); dataloader WAIT is off that timer → a CPU-aug bottleneck inflates WALL `total_seconds` toward the 600s cap WITHOUT cutting `num_epochs`. Pre-smoke img/s; record both `num_epochs` (~142–155) and `total_seconds` (<600s).
- **Strength**: start mild (RandAugment N=1, M≈6, ≪ CIFAR default; or TrivialAugment). Prefer REPLACING RandomErasing over stacking a 3rd occlusion-like aug (over-regularizes 150ep). Watch ep25 for under-fit.

## Status on this goal
Tested **EXP-015**: **no-improvement**. Mild RandAugment(N=1,M=6) tied the same-session baseline as BOTH a replacement for RandomErasing (96.34) and an addition on top (96.36 = c0 96.36); none ≥ 96.48, all @ matched 149 epochs. Confirmed NOT under-fit (ep25 92.04/92.19 vs 92.27; fully annealed) → the mild aug is fully absorbed and adds zero generalization benefit. This is the THIRD aug mechanism to tie (after occlusion EXP-008, mixing EXP-011) → input-augmentation lane saturated; the ~96.3–96.5 plateau is a generalization ceiling not movable by aug diversity at this scale. OPERATIONAL: RandAugment is CPU-bound on the 8-worker harness (loader 20.6k img/s < 26k GPU rate) → inflates WALL (+63s) but NOT epochs (loader wait is off the compute-budget timer). Residual untried: stronger magnitude (N=2,M≈10) — ep25 showed headroom — but low-medium confidence (under-fit risk > ceiling-break odds). See `experiments/015/04-analysis.md`.

## Sources
- AutoAugment https://arxiv.org/abs/1805.09501 ; RandAugment https://arxiv.org/abs/1909.13719 ; TrivialAugment https://arxiv.org/abs/2103.10158
- Practitioner benchmark (ResNet-18/CIFAR-10, epoch-scaling caveat): https://sebastianraschka.com/blog/2023/data-augmentation-pytorch.html
- torchvision transforms docs (RandAugment/TrivialAugmentWide/AutoAugment signatures).
