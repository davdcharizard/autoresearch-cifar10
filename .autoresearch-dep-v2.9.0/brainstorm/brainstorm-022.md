# Brainstorm EXP-022
**Created**: 2026-05-27
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

- **airbench96 source code** (gh api repos/KellerJordan/cifar10-airbench/contents/airbench/lib_airbench96.py, utils.py)
  96.05% in 46.3s on A100. Uses Cutout(12px zero-fill) instead of RandomErasing. 4px reflect-pad translate instead of RandomCrop with zero padding. Lookahead EMA every 5 steps with cubic alpha schedule (0.95^5 * (step/total)^3). Nesterov SGD momentum 0.85. Separate BN bias param group with 64x higher LR. NOT a ResNet — custom ConvGroup with GELU, whitening conv, Dirac init. Does NOT use ghost batch norm (earlier web search was misleading).

- **Cutout: A Simple Data Augmentation Method** (DeVries & Taylor 2017, arXiv:1708.04552)
  Fixed-size square mask filled with zeros, applied after normalization. 16x16 optimal for CIFAR-10 in the original paper (on WideResNet-28-10). airbench96 uses 12x12 on a much smaller/faster architecture.

- **CIFAR-10 speedrun leaderboard** (github.com/KellerJordan/cifar10-airbench)
  Top entries use Cutout over RandomErasing. Reflect padding for cropping is standard. These are highly optimized training recipes where every detail matters.

## Experimental History Review

- 22 experiments completed, 10 improvements. Current best: 96.46% (EXP-020).
- **Regularization stack is saturated**: CutMix (EXP-010), Mixup (EXP-017), DropPath (EXP-018) all hurt when stacked on TrivialAugmentWide + RandomErasing + WD=5e-4 + LS=0.2. Adding more regularization causes under-fitting.
- **Throughput is binding constraint**: At ~99 epochs in 300s, per-step overhead of even ~1ms costs 6 epochs and measurable accuracy. Pre-activation blocks lost 6% throughput (EXP-021), SE blocks +9ms/step (EXP-011/012).
- **EMA partially explored but not exhausted**: Parameter-only EMA broken for BN (EXP-013). Full state_dict EMA β=0.999 too conservative for 92 epochs, only +0.05pp (EXP-014). Lower β or scheduled β (like airbench96 cubic schedule) not yet tried.
- **Optimization tuning mostly done**: Cosine warmup+decay is optimal (EXP-020, +0.55pp). Batch 256 optimal (EXP-009). LR=0.2 with warmup validated. Nesterov hurt in EXP-004 (throughput cost).
- **Untried approaches**: (1) Cutout replacing RandomErasing (suggested by EXP-021 report, validated by airbench96), (2) Deeper architecture NUM_BLOCKS=4, (3) Reflect padding for RandomCrop, (4) Higher BN bias learning rate.

## Candidate Ideas

### 1. Cutout Replacing RandomErasing
**Summary**: Replace `transforms.RandomErasing(p=0.25, scale=(0.02, 0.2))` with a Cutout-style augmentation: a fixed 12×12 square of zeros applied with probability ~0.5. This swaps one occlusion method for another rather than adding regularization. The key differences: Cutout uses a fixed-size square (predictable, learnable boundary), RandomErasing uses random-aspect-ratio rectangles with random fill values. Cutout zeros out a known-size region, forcing the network to rely on non-occluded parts more consistently.

**Reasoning**: airbench96 achieves 96.05% with Cutout(12px) as its only occlusion augmentation — no RandomErasing. The original Cutout paper (DeVries & Taylor 2017) showed +0.7-1.0pp gains on CIFAR-10 with WRN-28-10. Our model has exhausted the regularization budget (CutMix, Mixup, DropPath all hurt), but this is a swap, not an addition — total regularization pressure stays similar. The fixed-size zero-fill mechanism provides a different learning signal than RandomErasing's variable-size random-fill approach.

**Sources**: airbench96 source code (Cutout 12px), DeVries & Taylor 2017, EXP-021 report next steps, EXP-010/017/018 (regularization saturation evidence)

**Estimated Effort**: low — single line change in augmentation pipeline

**Risk Assessment**: Could perform identically to RandomErasing or slightly worse if TrivialAugmentWide already provides sufficient occlusion diversity. Worst case is a marginal regression (~0.1-0.2pp) since the regularization budget is similar. Zero throughput cost.

### 2. Deeper Architecture (NUM_BLOCKS=4, ResNet-26)
**Summary**: Increase `NUM_BLOCKS` from 3 to 4, changing the model from ResNet-20 (6×3+2=20 layers) to ResNet-26 (6×4+2=26 layers). This adds one more residual block per layer group (9→12 total blocks), increasing capacity by ~33% more conv layers while keeping WIDTH_MULT=4.

**Reasoning**: The current model may be capacity-limited at 96.46% — all regularization additions hurt, suggesting the model is already well-regularized and needs more capacity to push further. Wider models (WIDTH_MULT 1→2→4) have been the strongest single improvement lever in this project's history. Deeper adds capacity along the orthogonal depth axis. The throughput cost is the primary risk: ~33% more FLOPs means ~25-30% fewer epochs (from 99 to ~70-75 epochs). Whether the capacity gain overcomes the epoch loss is the core question.

**Sources**: EXP-021 report next steps (#1 suggestion), goal-learnings Patterns (throughput is binding constraint), EXP-007 (width-4x success pattern)

**Estimated Effort**: low — single constant change (NUM_BLOCKS=3→4)

**Risk Assessment**: The ~25-30% throughput reduction is significant. EXP-021 showed that even 6% throughput loss (-6 epochs) cost 0.23pp. A 25-30% loss (~25 epochs fewer) would need substantial per-epoch accuracy gain to compensate. High risk of no-improvement if the capacity gain doesn't outweigh the epoch reduction in 300s. The model goes from ~4.3M to ~5.7M params.

### 3. Reflect Padding for RandomCrop + Cutout Combined
**Summary**: Two zero-throughput-cost changes combined: (1) Replace `RandomCrop(32, padding=4)` (default zero padding) with `RandomCrop(32, padding=4, padding_mode='reflect')` — reflect padding instead of zero padding at crop boundaries. (2) Replace `RandomErasing` with Cutout(12px). Both changes are validated by airbench96 and target data augmentation quality without changing regularization pressure.

**Reasoning**: airbench96 uses both reflect-pad translate and Cutout as its augmentation recipe. Reflect padding preserves edge statistics better than zero padding (no artificial black borders), which should improve boundary feature learning. Combined with Cutout, this gives the model two validated augmentation improvements simultaneously. Since both are swaps (not additions), the regularization budget stays constant. Combining two small improvements that are individually marginal may compound to cross the +0.1pp threshold.

**Sources**: airbench96 source code (reflect pad + Cutout 12px), EXP-021 report next steps (Cutout), goal-learnings (regularization stack saturated — swaps not additions)

**Estimated Effort**: low — two small changes in the augmentation pipeline

**Risk Assessment**: Reflect padding alone may have negligible effect on 32×32 images with only 4px padding. Combined with Cutout, the risk is that the changes interact unpredictably with TrivialAugmentWide. Worst case is a marginal regression if both changes don't compose well. The advantage is that if one component helps and the other is neutral, we still get a gain.

## Idea Evaluation

**Evidence strength**: Idea 3 (reflect + Cutout) has the strongest evidence — both components are used in airbench96's validated 96.05% recipe. Idea 1 (Cutout alone) has good evidence from the same source but tests only half the augmentation recipe. Idea 2 (deeper) has theoretical support but no direct evidence that depth helps more than width at this model scale, and the throughput cost is a serious concern given EXP-021's lesson.

**Mechanism clarity**: Idea 3 has clear mechanisms for both components: reflect padding preserves edge statistics (no artificial zero borders), Cutout provides consistent fixed-size occlusion that forces feature redundancy. Idea 2's mechanism (more capacity via depth) is clear but the trade-off with throughput is hard to predict. Idea 1 is a subset of Idea 3.

**Expected impact**: Idea 3 maximizes the chance of crossing +0.1pp by combining two independent small improvements. Idea 1 risks being too small alone. Idea 2 has the highest upside if it works but the highest downside risk from throughput loss.

**Risk profile**: Idea 3 and Idea 1 have safe failure modes (marginal regression at worst, zero throughput cost). Idea 2 risks a significant regression if throughput loss dominates.

**Feasibility**: All three are low effort. Idea 3 is two simple changes to the augmentation pipeline.

Idea 3 dominates: it subsumes Idea 1 (includes Cutout) while adding reflect padding, has the strongest evidence base, zero throughput cost, and the safest failure mode.

## Chosen Idea
**Selected**: Reflect Padding for RandomCrop + Cutout Combined

**Why this idea**:
Both components are validated by airbench96's 96.05% recipe. They target data augmentation quality — the one axis where we can still make changes without hitting regularization saturation (since these are swaps, not additions). Zero throughput cost preserves the 99-epoch budget. Combining two independently supported small improvements maximizes the chance of crossing the +0.1pp threshold that neither might achieve alone.

**Hypothesis**:
Replacing zero-padding with reflect-padding in RandomCrop and replacing RandomErasing with Cutout(12px) will improve best_test_acc by +0.1-0.3pp (reaching 96.56-96.76%) by providing higher-quality augmented training samples — reflect padding eliminates artificial zero borders and Cutout provides consistent fixed-size occlusion that complements TrivialAugmentWide better than RandomErasing's variable-size random-fill approach. Zero throughput cost means 99 epochs are preserved.
