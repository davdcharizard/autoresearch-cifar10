# Brainstorm EXP-030
**Created**: 2026-05-28
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

No new search — this targets standard CIFAR-10 preprocessing that our project has deviated from.

## Experimental History Review

- **Current best**: 96.46% (EXP-020)
- **Six consecutive failures** (EXP-024-029) targeting optimizer tricks, schedule tuning, capacity, and architecture. All results cluster at 96.43-96.52% — within ±0.04pp noise of baseline.
- **Pattern**: Every major improvement came from introducing a NEW CATEGORY. No further gains possible by refining existing categories. Remaining untouched categories: input normalization, loss function, weight init.
- **Key observation**: std=(1,1,1) is non-standard. The code comment acknowledges this: "Yes original paper only mention per-pixel mean and this is per band." Modern recipes use per-channel std (0.2470, 0.2435, 0.2616). This widens the input distribution from [-0.5, 0.5] to [-2, 2], better matching Kaiming init's unit-variance assumption.

## Candidate Ideas

### 1. Proper Per-Channel Std Normalization
**Summary**: Change the input normalization std from (1,1,1) to the CIFAR-10 per-channel std (0.2470, 0.2435, 0.2616). This widens the normalized input range from ~[-0.5, 0.5] to ~[-2, 2] per channel, matching the standard preprocessing used in virtually all modern CIFAR-10 training recipes.

**Reasoning**: Kaiming init assumes unit-variance inputs to properly calibrate initial weight magnitudes. With std=(1,1,1), the input variance is ~0.08 per channel (not unit variance). With proper std normalization, input variance is ~1.0, correctly matching Kaiming init. This means the first conv layer starts with better-calibrated gradient magnitudes, potentially improving early training dynamics. BN after the first conv normalizes outputs, but the gradient w.r.t. first-layer weights is proportional to input magnitude. The 4x wider input range means 4x larger gradients for the first conv — this could accelerate feature learning in early epochs.

This is zero throughput cost (normalization runs on CPU during data loading). It's a category-level change — input preprocessing hasn't been modified in any prior experiment.

**Sources**: Standard CIFAR-10 preprocessing convention, Kaiming init paper (He et al. 2015), train.py comment at line 142

**Estimated Effort**: low — single constant change

**Risk Assessment**: Low-medium. The 4x input scaling changes gradient magnitudes for the first conv layer. If too aggressive, could cause early instability. BN should mitigate this for downstream layers. The worst case is a slight regression from changed optimization dynamics.

### 2. Nesterov + Reflect Padding Combined
**Summary**: Stack Nesterov momentum with reflect-padded RandomCrop. Two orthogonal zero-cost changes.

**Reasoning**: Nesterov (+0.06pp, EXP-026) and reflect padding (part of +0.07pp in EXP-022) target different axes. Unlike EXP-027 (failed combination on same axis), this combines optimizer + data quality.

**Sources**: EXP-026, EXP-022

**Estimated Effort**: low

**Risk Assessment**: Safe but likely still in noise band.

### 3. Reduced Label Smoothing (0.1 instead of 0.2)
**Summary**: Reduce label_smoothing from 0.2 to 0.1. At 96.46%, the model might be slightly over-regularized and reducing LS could allow tighter final convergence.

**Reasoning**: LS=0.2 was validated at baseline 95.39% (EXP-015, +0.18pp). At 96.46% with extensive regularization stacking, the optimal LS might be lower. EXP-029 showed that removing ANY regularization hurts — but this reduces rather than removes LS.

**Sources**: EXP-015 (LS=0.2 added), EXP-029 (removing regularization hurts)

**Estimated Effort**: low

**Risk Assessment**: Medium. LS=0.2 is validated. Reducing it risks losing the 0.18pp gain from EXP-015.

## Idea Evaluation

**Evidence strength**: Proper std normalization has strong external evidence — it's the standard preprocessing for CIFAR-10 across all major benchmarks and training recipes. Our deviation from this standard (std=1) is the most obvious non-standard choice remaining in the setup. Nesterov + reflect padding has moderate evidence. Reduced LS has the weakest — it risks undoing a validated gain.

**Mechanism clarity**: Std normalization has a clear mechanism — it aligns input statistics with Kaiming init's unit-variance assumption, improving initial gradient calibration. The effect propagates through BN but is most impactful for the first conv layer's gradients.

**Expected impact**: Std normalization is a category-level change (input preprocessing) — every prior category-level change produced measurable improvement. Even if the effect is modest, it's the most promising untouched category.

## Chosen Idea
**Selected**: Proper Per-Channel Std Normalization

**Why this idea**:
It's the most obvious deviation from standard CIFAR-10 practice remaining in our setup. Input normalization is a completely untouched category — every other major improvement came from introducing a new category. The mechanism is clear (Kaiming init calibration). Zero throughput cost.

**Hypothesis**:
Changing std from (1,1,1) to (0.2470, 0.2435, 0.2616) will improve best_test_acc by 0.1-0.2pp (to 96.56-96.66%) by properly calibrating the input distribution for Kaiming init, improving first-layer gradient quality and early training dynamics. Zero throughput cost.
