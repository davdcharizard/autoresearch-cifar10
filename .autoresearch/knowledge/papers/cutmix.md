# CutMix (Yun et al., ICCV 2019)

**Paper**: "CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features"
(Yun, Han, Oh, Chun, Choe, Yoo) — arXiv:1905.04899.

## Core idea
Cut a random rectangular region out of one training image and paste in the patch from another image; mix the
two labels in proportion to the **kept area**. Combines the strengths of Cutout (regional dropout) and Mixup
(label interpolation) without Cutout's wasted (zeroed) pixels or Mixup's unnatural global pixel blend — the pasted
region contains real, spatially-coherent local features.

## Mechanics (standard recipe)
- Sample `lam ~ Beta(alpha, alpha)`. For `alpha = 1.0`, Beta(1,1) == Uniform(0,1) — so `lam = rand()`.
- Box size: `cut_rat = sqrt(1 - lam)`, `cut_w = W*cut_rat`, `cut_h = H*cut_rat`, center uniformly random, clipped.
- Paste the box from a shuffled batch index `perm` into every image (one shared box + one `perm` per batch).
- Recompute `lam = 1 - box_area / (W*H)` (area correction after clipping).
- Loss: `lam * CE(out, y) + (1-lam) * CE(out, y[perm])`.
- Often applied with a per-batch probability `p` (0.5–1.0).

## Why it can help
A strong regional regularizer that improves localization and reduces overfitting; reported consistent CIFAR /
ImageNet top-1 gains in the paper.

## Caveats relevant to THIS project (CIFAR-10, k=4 WRN-20, 300s budget)
- **Long-schedule technique**: the paper trains 200–300 epochs on CIFAR. Label-mixing augmentations warm up slowly;
  at our ~84–91-epoch budget CutMix may UNDERFIT and fail to realize its benefit (possible null/slight regression).
- **Label-mixing family**: same family as Mixup, which nulled here as a weak α=0.2 variant on the already
  regularization-saturated net (EXP-011). CutMix is regional/stronger, so not a guaranteed repeat — but a yellow flag.
- **Test loss artifact**: training on soft (mixed) targets typically raises test cross-entropy even when accuracy
  holds/improves — judge CutMix on ACCURACY, not loss (same artifact seen with Mixup, EXP-011).
- **Implementation**: trivially GPU-vectorizable (one `randperm` + one slice-paste), throughput-neutral, no new deps.
  Used in EXP-018.
