# Brainstorm EXP-022
**Created**: 2026-05-29
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

- Gradient clipping accelerates training with theoretical justification (https://arxiv.org/pdf/1905.11881) — prevents large gradient spikes, stabilizes optimization
- No additional search needed; this is a well-understood technique

## Experimental History Review

- **23 experiments** (BASE through EXP-021), baseline 96.39%, six consecutive failures
- **Architecture changes**: all failed (SE, pre-activation, stochastic depth, deeper, wider, VGG-style)
- **TTA extensions**: failed — spatial shifts dilute hflip signal (EXP-020)
- **Regularization tuning**: CutMix p=0.3 worse than p=0.5 (EXP-021)
- **Training speed**: channels_last works but CosineAnnealingLR restart kills the benefit (EXP-018, 019)
- **Key pattern**: The current recipe is well-optimized. Changes to any single component tend to hurt. What remains is gradient-level interventions that don't change the recipe but improve how it trains.
- **Never tried**: gradient clipping, CutMix alpha tuning, weight decay tuning, warmup length tuning

## Candidate Ideas

### 1. Gradient Clipping (max_norm=5.0)
**Summary**: Add `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)` after `scaler.scale(loss).backward()` and before `scaler.step(optimizer)`. Must unscale gradients first for clipping to work correctly with AMP: call `scaler.unscale_(optimizer)` before clipping. Use max_norm=5.0 (moderate, only clips extreme outliers).

**Reasoning**: CutMix creates mixed images with mixed labels. Occasionally, a CutMix batch produces a confusing mix that generates a large gradient, causing the model to take an outsized step away from its current minimum. This is most damaging in late training (low LR) where the model is refining near a minimum — a single large gradient can undo many small updates. Gradient clipping prevents these outlier updates. The mechanism is orthogonal to all existing regularization and hyperparameters. max_norm=5.0 is conservative — it only affects the largest gradient spikes, not normal training gradients.

**Sources**: Gradient clipping theory (arxiv 1905.11881), standard practice in transformer/LLM training

**Estimated Effort**: low — 2 lines (unscale + clip)

**Risk Assessment**: Very low. With max_norm=5.0, normal gradients are not affected. Only extreme outliers are clipped. Worst case: no effect (if gradients are already well-behaved). Cannot cause regression since it only limits extreme updates.

### 2. CutMix Alpha Reduction (1.0 → 0.5)
**Summary**: Change CUTMIX_ALPHA from 1.0 to 0.5. With Beta(0.5, 0.5), lambda is U-shaped — CutMix patches tend to be very small or very large, so one image dominates the mix. This reduces the difficulty of mixed samples while keeping CutMix's regularization active.

**Reasoning**: Beta(1,1)=Uniform creates equal probability for all mixing ratios. Beta(0.5,0.5) favors extreme ratios where one image dominates. This means more CutMix batches are "easy" (dominated by one class) while still providing the occasional challenging mix. Could help convergence.

**Sources**: CutMix paper uses alpha=1.0 by default but alpha=0.5 is common in practice

**Estimated Effort**: low — single constant change

**Risk Assessment**: Low. U-shaped distribution might under-regularize if the model benefits from challenging mixes.

## Idea Evaluation

Gradient clipping is the clearer intervention — it targets a specific failure mode (gradient spikes from CutMix) with a well-understood mechanism. CutMix alpha tuning changes the difficulty distribution of augmented samples, which is less targeted. Gradient clipping is also truly orthogonal — it doesn't change any hyperparameter, just adds a safety rail.

## Chosen Idea
**Selected**: Gradient Clipping (max_norm=5.0)

**Why this idea**: Orthogonal to all existing components. Targets a specific failure mode (gradient spikes) with no risk of hurting normal training. The mechanism is clear and well-validated across deep learning.

**Hypothesis**: Adding gradient clipping (max_norm=5.0) will stabilize late-stage training by preventing CutMix-induced gradient spikes, improving best_test_acc from 96.39% to ≥96.49%.
