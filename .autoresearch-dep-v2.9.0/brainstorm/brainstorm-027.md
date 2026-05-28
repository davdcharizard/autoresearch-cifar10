# Brainstorm EXP-027
**Created**: 2026-05-28
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

No new external sources — this experiment builds on validated internal findings from EXP-022, EXP-025, and EXP-026.

## Experimental History Review

- **Current best**: 96.46% (EXP-020) — cosine warmup+decay LR schedule
- **~96.5% ceiling confirmed**: Three optimizer experiments (EXP-024: -1.99pp, EXP-025: +0.03pp, EXP-026: +0.06pp) all failed to break 96.56%. Optimizer-level tricks alone are insufficient.
- **Near-miss results that could stack**: EXP-026 Nesterov +0.06pp (optimizer quality), EXP-022 augmentation swap +0.07pp (augmentation quality). Both zero throughput cost, different mechanisms.
- **Key constraint**: ~96 epochs in 300s budget. The baseline reportedly gets ~99, but both EXP-025 and EXP-026 got 96 — this may be system variability (~3% noise in epoch count).
- **Strategic insight**: Goal-learnings confirm capacity is likely binding, but combining two orthogonal near-miss improvements could clear the threshold if effects are additive.

## Candidate Ideas

### 1. Nesterov + Shortened Warmup (5→3 epochs)
**Summary**: Combine two zero-cost changes: (1) `nesterov=True` in SGD for better gradient estimates, and (2) reduce `WARMUP_EPOCHS` from 5 to 3 to free 2 more epochs of productive high-LR training. The shortened warmup reaches full LR by epoch 3 instead of epoch 5, providing 2 additional epochs of peak-LR exploration before cosine decay begins. Nesterov's look-ahead gradients improve the quality of each gradient step, and the extra high-LR epochs give the model more time to benefit from those better gradients.

**Reasoning**: Nesterov alone gave +0.06pp (EXP-026). The shortened warmup frees 2 epochs that would otherwise be spent on LR ramp-up. With cosine decay starting after warmup, the model transitions smoothly regardless of warmup length. The 5-epoch warmup was introduced for stability with batch 256 (EXP-009), but the cosine schedule (EXP-020) eliminated the instability concerns that motivated long warmup. These two changes target different aspects: Nesterov improves gradient quality, warmup shortening increases productive training time. The effects should be additive or synergistic. Combined estimate: +0.06pp (Nesterov) + 0.04-0.06pp (2 extra epochs) ≈ +0.10-0.12pp.

**Sources**: EXP-026 (Nesterov +0.06pp), EXP-009 (warmup introduced), EXP-020 (cosine schedule), airbench96 (short warmup with aggressive LR), goal-learnings § Patterns (throughput as binding constraint)

**Estimated Effort**: low — two parameter changes

**Risk Assessment**: Very safe. Nesterov is proven harmless (+0.06pp). Shortened warmup risk: model might show instability in epochs 3-5 at full LR. But cosine decay immediately begins reducing LR after warmup, providing natural damping. Worst case: warmup shortening adds no benefit and Nesterov gives the same +0.06pp as before.

### 2. Nesterov + Reflect Padding in RandomCrop
**Summary**: Combine Nesterov momentum with reflect-padded RandomCrop (replacing zero-padding). EXP-022 showed reflect padding + Cutout gave +0.07pp. Here we just change the padding mode without swapping the occlusion method (keep RandomErasing), isolating the reflect padding benefit. Combined with Nesterov (+0.06pp), this targets both optimizer and data quality axes.

**Reasoning**: Zero-padding in RandomCrop introduces artificial black borders that the model must learn to ignore. Reflect-padding provides more natural image continuity at borders, potentially improving feature learning quality. This is a different mechanism from Nesterov (optimizer) and the two effects should be orthogonal. The reflect padding change is: `transforms.RandomCrop(32, padding=4, padding_mode='reflect')`.

**Sources**: EXP-022 (reflect padding + Cutout +0.07pp; reflect padding was one of two changes), EXP-026 (Nesterov +0.06pp)

**Estimated Effort**: low — two parameter changes

**Risk Assessment**: Very safe. Reflect padding is a minor data augmentation change. Combined with Nesterov, worst case is no improvement or marginal regression. The EXP-022 result included both reflect padding and Cutout swap — the isolated effect of reflect padding alone is unknown but likely positive.

### 3. Deeper Architecture (NUM_BLOCKS=4, ResNet-26)
**Summary**: Increase `NUM_BLOCKS` from 3 to 4, creating a ResNet-26 (26 layers) with WIDTH_MULT=4. This adds 3 BasicBlocks (one per stage), increasing parameters by ~33% from ~4.3M to ~5.7M. The additional depth provides more representational capacity and gradient flow paths.

**Reasoning**: Three optimizer experiments confirm a ~96.5% ceiling that is likely capacity-bound. Adding depth is the most direct way to increase capacity. Unlike width scaling (which increases computation quadratically), depth scaling adds blocks linearly. The additional parameters come at a throughput cost — estimated ~25% fewer epochs (~74 vs 96). The risk is the same throughput trap that killed SE blocks (EXP-011/012), but depth adds fundamental capacity while SE blocks only added channel attention.

**Sources**: EXP-007 (width scaling worked: +0.57pp), EXP-011/012 (SE blocks failed from throughput cost), goal-learnings § Failed Approaches (capacity interventions vs throughput cost trade-off)

**Estimated Effort**: low — single constant change (`NUM_BLOCKS = 4`)

**Risk Assessment**: Moderate. ~25% throughput regression (96→~72 epochs) is a significant cost. SE blocks with ~50% throughput regression failed. A 25% regression is smaller but still meaningful. The model at 72 epochs might still be undertrained despite having more capacity per epoch.

## Idea Evaluation

**Evidence strength**: Nesterov + shortened warmup has the strongest evidence — Nesterov is proven at +0.06pp, and warmup shortening has indirect support from airbench96. Nesterov + reflect padding has moderate evidence — reflect padding's isolated effect is unknown. Deeper architecture has indirect evidence from width scaling (EXP-007) but depth scaling dynamics are different.

**Mechanism clarity**: Nesterov + shortened warmup has the clearest compound mechanism — better gradients + more productive epochs = higher quality exploration. Both effects are well-understood and operate on different axes (gradient quality vs training time). Nesterov + reflect padding also targets different axes but reflect padding's mechanism is weaker.

**Expected impact**: Nesterov + shortened warmup is estimated at +0.10-0.12pp if additive. This is right at the threshold. Deeper architecture could give a larger jump if the throughput cost doesn't dominate, but the risk is higher.

**Risk profile**: Nesterov + shortened warmup is the safest — both components are proven harmless and there's no throughput risk. Deeper architecture has the most risk from the throughput trade-off.

## Chosen Idea
**Selected**: Nesterov + Shortened Warmup (5→3 epochs)

**Why this idea**:
Stacks two individually-proven zero-cost changes that target different axes (gradient quality + training time). Nesterov is validated at +0.06pp with zero throughput cost. Shortened warmup frees 2 epochs of productive high-LR exploration. The combined effect should be additive since the mechanisms are orthogonal. This is the safest path to clearing the 0.1pp threshold without risking throughput regression.

**Hypothesis**:
Combining Nesterov momentum with shortened warmup (3 epochs instead of 5) will improve best_test_acc by 0.10-0.15pp (to 96.56-96.61%) by stacking better gradient estimates with more productive high-LR training time, both at zero throughput cost. The model should complete ~96 epochs as in EXP-026, with the extra 2 warmup-freed epochs providing higher effective LR earlier in training.
