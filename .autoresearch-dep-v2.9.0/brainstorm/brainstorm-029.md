# Brainstorm EXP-029
**Created**: 2026-05-28
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

- **ResNet shortcut connection options** (He et al. 2015, original ResNet paper)
  The paper defines three shortcut options: (A) identity mapping with zero-padding for extra channels, (B) 1x1 conv projection when dimensions mismatch, identity otherwise, (C) 1x1 conv projection at ALL shortcuts. Option B outperforms A by ~0.5pp on CIFAR-10 for ResNet-110. Option C provides marginal further improvement. Our model uses option A (zero-padding). Most modern implementations use option B.

## Experimental History Review

- **Current best**: 96.46% (EXP-020)
- **Five consecutive failures** (EXP-024-028): optimizer tricks (+0.03 to +0.06pp), warmup tuning (-0.01pp), depth increase (-0.15pp). All approaches hit the ~96.5% ceiling or throughput trap.
- **Key insight from EXP-028**: Capacity increases costing >10% throughput fail. But capacity increases with <1% throughput cost haven't been explored.
- **Architecture modifications tested**: SE blocks (failed, throughput), pre-activation (failed, throughput). Neither addressed the shortcut connection.
- **Untouched axis**: Shortcut connection quality. Current zero-padding provides no learned transformation for channel-mismatched shortcuts. This is a known architectural weakness.

## Candidate Ideas

### 1. Learned 1x1 Conv Shortcut Projections
**Summary**: Replace the zero-padding shortcut at stage transitions with a learned 1x1 conv + BN. Currently, when channels increase between stages (64→128, 128→256), the shortcut uses stride + zero-padding (`F.pad`). This means the extra channels in the shortcut receive no signal — they're just zeros added to the residual branch output. A 1x1 conv projection learns to transform ALL channels from the input to match the output dimension, providing full gradient flow and a learned feature transformation.

**Reasoning**: Zero-padded shortcuts are a known weakness of the He-2015 CIFAR ResNet. The original paper showed option B (1x1 projection) outperforms option A (zero-padding) by ~0.5pp on deeper models. Our model has 2 transitions: stage1→2 (64→128, 16×16→8×8) and stage2→3 (128→256, 8×8→4×4). The 1x1 convs add only ~41K params (<1% of 4.3M) and negligible compute — 1x1 convs on 16×16 and 8×8 feature maps are tiny operations. The throughput impact should be <0.5ms per step (<3%), far below the ~10% threshold that kills experiments.

This targets a fundamentally different axis from all recent experiments: neither optimizer quality nor model capacity, but **architectural efficiency** — making better use of the existing capacity by providing full gradient flow through shortcuts.

**Sources**: He et al. 2015 (original ResNet, shortcut options A/B/C), goal-learnings (capacity increases fail from throughput, but this adds minimal throughput), EXP-028 (depth failed at 22% throughput cost — this should be <3%)

**Estimated Effort**: medium — modify BasicBlock to use 1x1 conv shortcut when dimensions change

**Risk Assessment**: Low. The 1x1 conv projections are standard in most modern ResNet implementations. The throughput cost is minimal (<0.5ms for two tiny 1x1 convs). Additional params (~41K) are negligible. Worst case: throughput cost slightly higher than estimated, eating 1-2 epochs. The architectural change is well-understood and shouldn't cause instability.

### 2. Nesterov + Reflect Padding
**Summary**: Combine Nesterov momentum (`nesterov=True`) with reflect-padded RandomCrop (`padding_mode='reflect'`). Two orthogonal zero-cost changes.

**Reasoning**: Nesterov gave +0.06pp (EXP-026). Reflect padding was part of EXP-022 (+0.07pp combined with Cutout swap). Unlike EXP-027 (Nesterov + warmup, same axis), this combines optimizer + data quality — truly orthogonal axes.

**Sources**: EXP-026 (Nesterov +0.06pp), EXP-022 (reflect padding + Cutout +0.07pp)

**Estimated Effort**: low — two parameter changes

**Risk Assessment**: Very safe but likely still in the noise band (~96.5%). Neither individual effect exceeded 0.1pp.

### 3. Per-Channel Std Normalization
**Summary**: Change the input normalization std from (1,1,1) to the proper CIFAR-10 per-channel std (0.2470, 0.2435, 0.2616). Currently, inputs are mean-subtracted but not std-normalized, leaving values in [-0.5, 0.5] range. Proper std normalization spreads them to [-2, 2].

**Reasoning**: Most modern CIFAR-10 recipes use proper std normalization. The wider input range could affect gradient magnitudes, Kaiming init calibration, and AMP FP16 precision. However, BN after the first conv normalizes features anyway, so the effect is likely small.

**Sources**: Standard CIFAR-10 preprocessing practice, original ResNet paper (per-pixel mean only)

**Estimated Effort**: low — single constant change

**Risk Assessment**: Very safe but effect likely negligible since BN normalizes downstream features.

## Idea Evaluation

**Evidence strength**: Learned shortcut projections have the strongest evidence — the original ResNet paper directly compared options A vs B and found ~0.5pp improvement on CIFAR-10. This is a well-established architectural improvement that our model simply hasn't adopted. Nesterov + reflect padding has weaker evidence (both effects <0.1pp individually). Std normalization has the weakest evidence.

**Mechanism clarity**: Learned shortcuts have the clearest mechanism — zero-padded channels receive no gradient signal from the shortcut path, so the model relies entirely on the residual branch for those channels. A learned projection provides full gradient flow and a meaningful feature transformation for ALL channels. This is a structural gradient flow improvement, not an optimizer or regularization trick.

**Expected impact**: Learned shortcuts target a genuine architectural weakness that no prior experiment has addressed. The original paper's ~0.5pp improvement at lower baselines might translate to 0.1-0.3pp at our level. The <1% throughput cost means epoch count stays near ~96. This is the most promising approach for breaking through.

**Risk profile**: Learned shortcuts are safe — well-established technique, minimal compute, no instability risk.

## Chosen Idea
**Selected**: Learned 1x1 Conv Shortcut Projections

**Why this idea**:
Targets a genuine, untouched architectural weakness (zero-padded shortcuts) with the strongest evidence base (original ResNet paper, option B vs A). The mechanism is clear — full gradient flow through all shortcut channels instead of zero-padding half of them. Minimal throughput cost (<0.5ms per step, <3%). This is the first experiment to target shortcut quality, a fundamentally different axis from optimizer, schedule, augmentation, or capacity.

**Hypothesis**:
Replacing zero-padded shortcuts with learned 1x1 conv projections at stage transitions will improve best_test_acc by 0.1-0.3pp (to 96.56-96.76%) by providing full gradient flow through all shortcut channels. The ~41K additional parameters add <1% model size and the 1x1 convs on small feature maps add <3% throughput cost, preserving the ~96 epoch budget.
