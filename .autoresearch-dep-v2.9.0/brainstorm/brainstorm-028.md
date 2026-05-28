# Brainstorm EXP-028
**Created**: 2026-05-28
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

No new external sources — this experiment is driven by the strong internal evidence from four consecutive optimizer experiments hitting a ~96.5% ceiling.

## Experimental History Review

- **Current best**: 96.46% (EXP-020) — cosine warmup+decay LR schedule
- **~96.5% ceiling confirmed by 4 experiments**: EXP-024 (BN bias LR: -1.99pp), EXP-025 (GC: +0.03pp), EXP-026 (Nesterov: +0.06pp), EXP-027 (Nesterov+short warmup: -0.01pp). All optimizer/schedule tricks exhausted.
- **Capacity scaling precedent**: EXP-007 (WIDTH_MULT 2→4) gained +0.38pp despite 22% epoch reduction (106→83). This proves capacity gains CAN outweigh throughput losses.
- **Throughput trap precedent**: SE blocks (EXP-011/012) failed with ~50% throughput loss. Pre-activation (EXP-021) failed with 6% loss. The critical question is whether the throughput cost is small enough.
- **Exhausted categories**: Optimizer tricks (4 experiments), regularization stacking (4 experiments), EMA (3 experiments), architectural modifications (SE, pre-activation), augmentation swaps, BN tuning
- **Untried**: Depth increase (NUM_BLOCKS=4), wider width (WIDTH_MULT=5), fundamentally different architecture

## Candidate Ideas

### 1. Deeper Architecture: NUM_BLOCKS=4 (ResNet-26)
**Summary**: Increase `NUM_BLOCKS` from 3 to 4, creating a ResNet-26 (6×4+2=26 layers) with the same WIDTH_MULT=4. This adds one BasicBlock per stage (3 stages × 1 block = 3 additional blocks = 6 additional conv layers), increasing parameters from ~4.3M to ~5.7M (+33%). The additional depth provides more representational capacity through deeper feature hierarchies and more gradient flow paths.

**Reasoning**: Four consecutive optimizer experiments (EXP-024-027) confirm that optimization quality is NOT the bottleneck — the model is capacity-bound at ~96.5%. Depth is the most direct and efficient capacity increase for ResNets: it adds blocks linearly (unlike width which scales quadratically). The width scaling precedent (EXP-007: +0.38pp with 22% epoch reduction) suggests capacity gains CAN outweigh throughput losses. With NUM_BLOCKS=4, the estimated per-step time increase is ~25% (from ~16ms to ~20ms), reducing epochs from ~96 to ~75. This is a 22% epoch reduction — identical to the width scaling case that succeeded.

**Sources**: EXP-007 (width scaling: +0.38pp, 22% epoch loss), EXP-011/012 (SE: failed, ~50% epoch loss), EXP-021 (pre-activation: failed, 6% epoch loss), goal-learnings § Failed Approaches (throughput vs capacity trade-off)

**Estimated Effort**: low — single constant change (`NUM_BLOCKS = 4`)

**Risk Assessment**: Moderate. The ~25% throughput regression is the main risk — identical to the width scaling experiment that succeeded but at a higher baseline where each pp is harder. If the deeper model needs more epochs to converge (possible with 50% more parameters), the 300s budget may be insufficient. The model should also adjust ESTIMATED_EPOCHS to ~75 to match the actual epoch count for correct cosine schedule behavior. Worst case: accuracy regression from insufficient training of a larger model.

### 2. Nesterov + Reflect Padding
**Summary**: Combine Nesterov momentum (`nesterov=True`) with reflect-padded RandomCrop (`padding_mode='reflect'`). Two orthogonal zero-cost changes targeting optimizer quality and data quality respectively.

**Reasoning**: Nesterov gave +0.06pp (EXP-026). Reflect padding was part of EXP-022's augmentation swap (+0.07pp, but bundled with Cutout). Isolating reflect padding with Nesterov targets two different axes. Unlike EXP-027 which combined Nesterov with warmup (same axis — both schedule/optimizer), this combines optimizer with data quality.

**Sources**: EXP-026 (Nesterov +0.06pp), EXP-022 (reflect padding + Cutout +0.07pp)

**Estimated Effort**: low — two parameter changes

**Risk Assessment**: Very safe but likely still below threshold. Both individual effects are <0.1pp, and even if fully additive, the combined effect (~0.13pp) has high variance. May end up in the same ~96.5% noise band.

### 3. Aggressive Width: WIDTH_MULT=5
**Summary**: Increase WIDTH_MULT from 4 to 5, giving channel widths {80, 160, 320}. This increases parameters from ~4.3M to ~6.7M (+56%). Per-step time would increase significantly (~40-50%) since conv computation scales quadratically with channel count.

**Reasoning**: More capacity via width. Width scaling previously worked (EXP-007: 2→4). However, width scales quadratically in compute while depth scales linearly, making width less efficient.

**Sources**: EXP-007 (width 2→4 worked), goal-learnings § Patterns (throughput as binding constraint)

**Estimated Effort**: low — single constant change

**Risk Assessment**: High. ~40-50% throughput regression would reduce epochs from ~96 to ~55-60. This is close to the SE-block failure regime (~50% loss). The quadratic cost of width makes this less efficient than depth.

## Idea Evaluation

**Evidence strength**: NUM_BLOCKS=4 has the strongest evidence path — the width scaling precedent (EXP-007) proved capacity scaling works at exactly the same throughput cost ratio (22% epoch reduction). Nesterov + reflect padding has weaker evidence — both effects are <0.1pp individually. WIDTH_MULT=5 has the weakest — quadratic cost makes it likely to hit the throughput trap.

**Mechanism clarity**: All three have clear mechanisms. NUM_BLOCKS=4 adds depth (more feature hierarchies, more gradient paths). The mechanism is well-understood from the ResNet literature and validated in our project via WIDTH_MULT scaling.

**Expected impact**: NUM_BLOCKS=4 targets the identified bottleneck (capacity) directly. At +33% params with a 22% epoch cost, the capacity-per-epoch ratio improves by ~14%. This is the same trade-off profile as the successful EXP-007. Nesterov + reflect padding is likely <0.1pp. WIDTH_MULT=5 has high impact but unacceptable throughput cost.

**Risk profile**: NUM_BLOCKS=4 has moderate risk but the highest potential payoff. Nesterov + reflect padding is safe but likely below threshold. WIDTH_MULT=5 is too risky.

## Chosen Idea
**Selected**: Deeper Architecture: NUM_BLOCKS=4 (ResNet-26)

**Why this idea**:
After four consecutive optimizer experiments confirming a capacity-bound ~96.5% ceiling, depth increase is the most direct way to add capacity. The width scaling precedent (EXP-007: +0.38pp with identical 22% epoch cost) validates that capacity gains outweigh moderate throughput losses. Depth is more efficient than width (linear vs quadratic compute scaling). This is the first experiment to genuinely target a different bottleneck since EXP-020.

**Hypothesis**:
Increasing NUM_BLOCKS from 3 to 4 (ResNet-26, ~5.7M params) will improve best_test_acc by 0.2-0.5pp (to 96.66-96.96%) by providing 33% more representational capacity. The ~22% epoch reduction (96→~75 epochs) will be compensated by higher quality per epoch, as demonstrated by the width scaling precedent (EXP-007). The cosine schedule should be adjusted to ESTIMATED_EPOCHS=75 to match the actual epoch count.
