# Brainstorm EXP-026
**Created**: 2026-05-28
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

- **bfloat16 vs float16 for conv2d performance** (https://forums.developer.nvidia.com/t/bfloat16-has-worse-performance-than-float16-for-conv2d-in-pytorch/219852)
  bfloat16 shows worse performance than float16 for conv2d on some hardware. Since our model is conv-dominated and throughput-constrained, switching from float16 to bfloat16 risks losing epochs — ruled out.

- **PyTorch mixed precision best practices** (https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/)
  GradScaler is recommended for float16 but optional for bfloat16 (wider dynamic range avoids loss scaling). Float16 with GradScaler is the validated configuration for our setup.

## Experimental History Review

- **Current best**: 96.46% (EXP-020) — cosine warmup+decay LR schedule
- **Improvement trajectory**: 10 successful improvements across 27 experiments, from 91.72% (BASE) to 96.46%
- **Recent marginal results**: EXP-022 +0.07pp (augmentation swap), EXP-025 +0.03pp (gradient centralization), EXP-016 +0.02pp (BN momentum) — all below 0.1pp threshold, suggesting diminishing returns at this accuracy level
- **Exhausted approach classes**: EMA (3 variants), SE blocks (2 variants), regularization stacking (CutMix/Mixup/DropPath/augmentation-swap), pre-activation blocks, torch.compile, BN momentum tuning, BN bias LR, gradient centralization
- **Key constraint**: ~99 epochs in 300s budget; Python-level per-parameter operations cost ~0.5ms/step = ~3 epochs (EXP-025). Only truly zero-cost changes are viable.
- **Untried gaps**: Nesterov momentum (revisited — EXP-004 was in completely different context), alternating flip, warmup length tuning, weight decay tuning, deeper architecture
- **Critical pattern**: EXP-004 (Nesterov + LS=0.1) failed at 93.28% but the context was: no AMP, batch 128, step-decay LR, width 2x, label_smoothing=0.1. Current setup has AMP, batch 256, cosine LR, width 4x, label_smoothing=0.2. Nesterov was never tested in isolation in the current regime.

## Candidate Ideas

### 1. Nesterov Momentum
**Summary**: Enable Nesterov momentum by adding `nesterov=True` to the existing `optim.SGD()` call. This is a single-parameter change with zero computational overhead. Nesterov momentum modifies the gradient update to use a "look-ahead" position — instead of computing the gradient at the current position, it approximates the gradient at the position after the momentum step, providing a better estimate of the gradient at the next step.

**Reasoning**: Nesterov momentum is well-established to improve convergence speed and final accuracy in SGD-based training. With cosine decay to near-zero LR, the final 20-30 epochs are the most critical for accuracy — this is exactly where Nesterov's better gradient estimates compound most effectively. EXP-004 tested Nesterov but in a radically different setup: no AMP, batch 128 (vs 256), step-decay LR (vs cosine), width 2x (vs 4x), and label_smoothing=0.1 (vs 0.2). The 4-epoch throughput cost observed in EXP-004 was likely due to the pre-AMP setup where per-step time was 11ms — with AMP at 16ms/step, the Nesterov overhead (identical FLOPS, different momentum formula) should be negligible. This is the only major optimizer-level change that hasn't been properly tested in the current regime.

**Sources**: EXP-004 (failed in different context, count: 1 — low importance in goal-learnings), goal-learnings § Patterns (cosine decay as default, throughput as binding constraint)

**Estimated Effort**: low — single parameter change

**Risk Assessment**: Extremely safe — worst case is no improvement. Nesterov with cosine schedule is a standard combination used in many modern training recipes. The concern from EXP-004 (throughput cost) should not apply with AMP+batch 256 where per-step time is dominated by GPU compute, not the momentum formula. Even if there's a marginal throughput cost, the improved gradient quality should compensate.

### 2. Shortened Warmup (5 → 3 epochs)
**Summary**: Reduce `WARMUP_EPOCHS` from 5 to 3. This shortens the linear LR warmup phase, allowing the model to reach full LR (0.2) by epoch 3 instead of epoch 5. This frees 2 additional epochs at higher effective LR during the exploration phase, potentially improving the quality of the basin of attraction found before cosine decay begins.

**Reasoning**: The 5-epoch warmup was introduced in EXP-009 with batch 256 + LR 0.2 as a standard stability measure. However, with the current cosine schedule (EXP-020), the warmup transitions smoothly into the cosine decay — there's no abrupt LR transition that requires a long warmup. In the airbench96 recipe, warmup is only ~1-2 epochs for lr=9.0 and batch 1024. For our more conservative lr=0.2, 3 epochs may provide sufficient stability while freeing 2 epochs for productive exploration. The LR at the end of 3 warmup epochs would be 0.2 (full) vs 0.12 at epoch 3 in the current 5-epoch warmup. Zero throughput cost — just a schedule shape change.

**Sources**: EXP-009 (introduced 5-epoch warmup), EXP-020 (cosine schedule makes warmup transitions smooth), airbench96 (short warmup with aggressive LR)

**Estimated Effort**: low — single constant change

**Risk Assessment**: Low risk. If 3 epochs of warmup is too short, the model might show instability in epochs 3-10 as it suddenly reaches full LR. However, cosine decay immediately begins reducing LR after warmup, providing a natural damping effect. Worst case: slight regression from early instability that doesn't fully recover. Zero throughput cost.

### 3. Alternating Flip Augmentation
**Summary**: Replace the stochastic `RandomHorizontalFlip()` in the transform pipeline with a deterministic alternating flip that flips ALL images in even epochs and applies no flip in odd epochs. Remove `transforms.RandomHorizontalFlip()` from the train transform and add `inputs = inputs.flip(-1)` conditionally in the training loop when `epoch % 2 == 0`, applied after moving data to GPU.

**Reasoning**: Random flip gives each image a 50% chance of being flipped each epoch — over 99 epochs, some images may be predominantly seen in one orientation by chance. Alternating flip guarantees equal exposure across consecutive epochs. This is used in airbench96 (96.05% on CIFAR-10). The flip is applied to GPU tensors (near-zero cost). However, EXP-022 showed augmentation swaps are in the noise floor (+0.07pp) at this accuracy level.

**Sources**: airbench96 (alternating flip), EXP-022 (augmentation swaps at noise floor), brainstorm-024 (previously considered)

**Estimated Effort**: low — ~5 lines of code

**Risk Assessment**: Likely below 0.1pp threshold given augmentation saturation evidence from EXP-022. The deterministic pattern could interact with TrivialAugmentWide's randomization in unexpected ways. Zero throughput cost.

## Idea Evaluation

**Evidence strength**: Nesterov has the broadest evidence — it's a textbook optimization technique with decades of validation. The EXP-004 "failure" was clearly context-specific (different model, optimizer, schedule, batch size). Shortened warmup has indirect evidence (airbench96 uses very short warmup) but no direct validation in our setup. Alternating flip has weak evidence — airbench96 uses it but its isolated contribution is unknown, and EXP-022 showed augmentation changes yield <0.1pp.

**Mechanism clarity**: Nesterov has the clearest mechanism — look-ahead gradients provide better gradient estimates, especially valuable in the final cosine decay phase where precise gradient direction matters most for reaching a good minimum. Shortened warmup's mechanism is simpler (more productive epochs) but the tradeoff with stability is less clear. Alternating flip's mechanism (variance reduction) is modest.

**Expected impact**: Nesterov targets optimization quality — the one axis that hasn't been properly tested in the current regime. At 96.46% with regularization saturated, better optimization is the most promising remaining direction. Shortened warmup provides a quantitative benefit (2 more epochs) but whether those epochs are more valuable than the warmup stability is unclear. Alternating flip is likely in the noise floor.

**Risk profile**: All three fail gracefully. Nesterov is the safest — it's a well-understood technique that can't crash or cause instability. Shortened warmup has slightly higher risk (early instability).

## Chosen Idea
**Selected**: Nesterov Momentum

**Why this idea**:
Strongest evidence (textbook technique, widely validated), clearest mechanism (look-ahead gradients improve convergence quality), truly zero computational cost (identical FLOPS to standard momentum), and uniquely targets optimization quality in the current regime. The prior "failure" (EXP-004) was in a completely different context and classified as low importance. This is the only major optimizer-level change that hasn't been properly isolated and tested with AMP + batch 256 + cosine decay + width 4x.

**Hypothesis**:
Adding `nesterov=True` to SGD will improve best_test_acc by 0.1-0.2pp (to 96.56-96.66%) by providing better gradient estimates through look-ahead momentum, particularly improving convergence quality in the final cosine decay phase (epochs 70-99) where precise gradient direction matters most. Zero throughput cost — the model will complete ~99 epochs as usual.
