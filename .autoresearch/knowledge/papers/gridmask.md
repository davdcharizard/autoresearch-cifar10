# GridMask Data Augmentation (Chen, Chen, Yang, Mu, Wang & Zeng, 2020, arXiv:2001.04086)

## Core idea
Information-dropping augmentation that deletes a REGULAR GRID of squares from the image, rather than one large hole (Cutout) or scattered random pixels (Random Erasing). The structured, DISTRIBUTED deletion balances two failure modes: deleting too little (no regularization) vs deleting a contiguous region so large it removes whole objects. Reported to beat Cutout on CIFAR-10/100, ImageNet classification, and detection.

## Parameters
- `d`: grid unit length (period), sampled per image from `[d_min, d_max]`.
- `ratio` (`r`): controls the kept/removed edge length within a unit. In the official impl, removed squares have side ≈ `r·d`; removed-AREA fraction ≈ `r²` for the grid-of-squares form (intersection of periodic row/col bands). Typical literature settings remove a LARGE fraction (r≈0.6 → ~36-64% depending on band convention) — much stronger than Cutout-16's ~25%.
- random `(offset_y, offset_x)` per image; optional rotation of the grid (often omitted).

## Implementation notes for THIS project (k=4 ResNet-20, GPU-vectorized aug, torch.compile reduce-overhead)
- GPU-vectorizable exactly like the project's `cutout_batch`: build a binary mask from coordinate-grid arithmetic (`((coord - offset) % d) < removed_side`), `masked_fill` to 0. No per-sample `.item()` syncs (cf. EXP-002 dataloader-throttle lesson). Static shapes (batch 128 fixed) → CUDA-graph-safe, dt-neutral (same op class as Cutout).
- `torch.remainder` (`%`) on int tensors is non-negative for positive divisor → `((coord-offset) % d) ∈ [0,d)` even for offsets exceeding the coordinate.
- **To isolate PATTERN from STRENGTH** on this regularization-saturated recipe: match GridMask's removed-area fraction to Cutout-16 (~25%) rather than using the literature's aggressive defaults — i.e. removed-square side = round(0.5·d) → removed-area ≈ 25%. Using the strong literature defaults would confound pattern with over-regularization (the regime that sank CutMix EXP-018 / dropout EXP-022 / GhostBN EXP-047).
- Eval is unaffected (train-only aug).

## Relevance
Occlusion is a PROVEN, non-redundant lever on this net (Cutout +1.1pp EXP-002/003, orthogonal to TrivialAugment EXP-013), but only Cutout's hole SIZE was ever tuned — the occlusion PATTERN (one hole vs distributed grid) is the single untested augmentation sub-lever. Used in EXP-048 as a matched-strength Cutout→GridMask swap. Risk: augmentation-quality tweaks have nulled on this saturated recipe (border-mode EXP-037).
