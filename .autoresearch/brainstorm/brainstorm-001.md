# Brainstorm EXP-001
**Created**: 2026-05-28
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

- **Wide Residual Networks (Zagoruyko & Komodakis 2016)** (https://arxiv.org/pdf/1605.07146)
  Width is more efficient than depth for ResNets on CIFAR-10. WRN-40-4 (8.9M params) outperforms ResNet-1001 (10.2M params) while being 8x faster. WRN-28-10 achieves 96.11% on CIFAR-10. Key: for a fixed compute budget, wider shallow networks beat thin deep ones.

- **torch.compile speedup benchmarks** (https://medium.com/@cshrivastava2000/how-i-achieved-2-9-pytorch-training-speedup-with-one-line-of-code-a8b8f88d388f)
  torch.compile achieves ~10-25% training speedup on ResNet with CIFAR-10. Combined with AMP (float16), total speedup can reach 2-3x.

- **AMP (Automatic Mixed Precision)**: Can approximately double training throughput by using float16 for forward/backward passes. Well-supported in PyTorch 2.9.

## Experimental History Review

**Current state** (1 experiment on this goal):
- BASE: 91.81% (ResNet-20, {16,32,64}, 270K params, step LR)
- EXP-000: 92.10% (+0.29%), cosine LR + CutOut + label smoothing, same architecture

**Key learnings**:
- Protocol: Cosine T_max must match actual epoch count (High Importance)
- Pattern: Modern recipe yields only +0.29% on ResNet-20 — capacity is the bottleneck (Medium Importance)

**Untried gaps**: model capacity increase (wider/deeper), AMP, torch.compile, CutMix/Mixup, projection shortcuts, Nesterov momentum

## Candidate Ideas

### 1. Wider ResNet (k=2) + AMP + torch.compile

**Summary**: Double channel widths from {16, 32, 64} to {32, 64, 128}, creating a ~1.07M parameter model (4x more). Add AMP (torch.amp.autocast + GradScaler) and torch.compile for speed compensation. AMP should ~2x throughput, torch.compile adds ~15%, for ~2.3x combined speedup. With ~4x more FLOPs offset by ~2.3x speedup, expect ~50-55 epochs in 300s. Switch to Nesterov SGD and use projection shortcuts (1x1 conv) instead of zero-padding for dimension-mismatched skip connections. Keep EXP-000's recipe (cosine LR, CutOut, label smoothing).

**Reasoning**: EXP-000 showed capacity is the bottleneck. The WideResNet paper proves width is more compute-efficient than depth. k=2 is conservative — enough epochs for convergence with 4x more capacity. AMP and torch.compile are proven speed optimizations.

**Sources**: WideResNet paper, EXP-000 analysis, torch.compile benchmarks

**Estimated Effort**: medium

**Risk Assessment**: ~50-55 epochs may be marginal for convergence. AMP could introduce numerical instability. torch.compile compilation overhead eats into 300s budget. If epochs are too few, accuracy could underwhelm despite more capacity. Graceful failure (no-improvement, not crash).

### 2. Wider + Deeper: ResNet-32 (k=2) + AMP + torch.compile

**Summary**: Increase both width (k=2: {32, 64, 128}) and depth (NUM_BLOCKS=5 for ResNet-32) with AMP + torch.compile. ~1.8M params, 5 blocks per group vs 3. Use Nesterov SGD and projection shortcuts.

**Reasoning**: More depth provides more complex feature hierarchies at each spatial scale. ResNet-32 with k=2 is still modest. However, ~6.7x total FLOPs with only ~2.3x speedup → ~2.9x slower → ~31 epochs in 300s.

**Sources**: WideResNet paper (width > depth, but WRN-28-10 is still 28 layers), He et al. 2015

**Estimated Effort**: medium

**Risk Assessment**: Only ~31 epochs is risky for convergence. The WideResNet paper found width more efficient than depth per-FLOP, suggesting the depth increase here doesn't justify the epoch cost.

### 3. Wider ResNet (k=2) + CutMix replacing CutOut

**Summary**: Same k=2 width + AMP + torch.compile as Idea 1, but replace CutOut with CutMix augmentation. CutMix pastes a patch from another image and mixes labels proportionally — stronger regularization without wasting the masked area. Optionally combine with Mixup (alpha=0.2, 50% switch probability).

**Reasoning**: Wider model needs stronger regularization. CutOut wastes information (zeros), CutMix preserves training signal. OpenMixup benchmarks show CutMix > CutOut consistently on CIFAR-10 across model sizes.

**Sources**: OpenMixup benchmarks, CutMix paper (Yun et al. 2019)

**Estimated Effort**: medium

**Risk Assessment**: CutMix+Mixup is more complex to implement (label mixing). Combined with label smoothing, total regularization might be too aggressive for ~50 epochs. Harder to attribute gains vs just width increase.

## Idea Evaluation

**Evidence strength**: All three share the width increase (strong evidence from WideResNet paper + EXP-000). Idea 1 is the cleanest test. Idea 2 adds depth which the WideResNet paper shows is less efficient per-FLOP. Idea 3 adds CutMix which has benchmark evidence but mixes two changes.

**Mechanism clarity**: Idea 1 — more channels = more features per layer, directly increasing capacity. Clear and isolated. Idea 2 — depth adds more computation stages, but at the cost of fewer epochs. Idea 3 — better augmentation + more capacity, but attribution is muddy.

**Expected impact**: Idea 1: 93.5-95% (4x capacity with proven recipe). Idea 2: similar ceiling but higher convergence risk. Idea 3: potentially +0.2-0.5% over Idea 1 from better augmentation, but more complexity.

**Risk profile**: Idea 1 fails gracefully. Idea 2 has highest convergence risk (~31 epochs). Idea 3 has implementation complexity.

**Strategy**: Idea 1 is the purest test of the capacity hypothesis. CutMix can be layered on top in a future experiment if width proves effective.

## Chosen Idea

**Selected**: Wider ResNet (k=2) + AMP + torch.compile

**Why this idea**:
Directly targets the identified bottleneck (model capacity) with the most compute-efficient approach (width > depth). AMP + torch.compile compensate for increased compute. Cleanly isolates the width variable while keeping EXP-000's proven recipe.

**Hypothesis**:
Doubling channel widths from {16,32,64} to {32,64,128} (~1.07M params) with AMP + torch.compile will improve best_test_acc from 92.10% to approximately 93.5-95%, driven by quadrupled representational capacity while maintaining ~50-55 training epochs via speed optimizations.
