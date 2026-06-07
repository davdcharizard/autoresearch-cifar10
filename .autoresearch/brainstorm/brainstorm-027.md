# Brainstorm EXP-027
**Created**: 2026-05-29
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review
- CutMix paper (Yun et al. 2019) uses alpha=1.0 by default (Uniform mixing). Many practitioners use alpha=0.2-0.5 for lighter mixing.

## Experimental History Review
- **28 experiments**, baseline 96.39%, 11 consecutive failures
- CutMix prob=0.3 failed (EXP-021) — the FREQUENCY was wrong to reduce
- CutMix alpha=1.0 → 0.5 changes MIXING INTENSITY, not frequency — qualitatively different
- With Beta(0.5, 0.5), lambda clusters near 0 and 1 (U-shaped) — most CutMix patches are very small or very large, so one image dominates. This is lighter mixing while keeping the frequency at 50%.

## Candidate Ideas

### 1. CutMix Alpha 0.5 (U-shaped mixing)
**Summary**: Change CUTMIX_ALPHA from 1.0 to 0.5. The Beta(0.5, 0.5) distribution is U-shaped — lambda values cluster near 0 and 1 instead of being uniformly distributed. This means most CutMix batches have one image clearly dominating the mix, with occasional strong mixes. The effect is lighter mixing on average while preserving CutMix's regularization at the same 50% application frequency.

**Reasoning**: EXP-021 showed reducing CutMix frequency (p=0.5→0.3) hurt. But the problem might not be the frequency — it's the mixing intensity. With alpha=1.0 (Uniform), ~50% of CutMix batches have lambda in [0.3, 0.7] meaning substantial mixing. With alpha=0.5, only ~20% of CutMix batches have lambda in [0.3, 0.7]. This gives the model more "easy" CutMix samples (dominated by one class) while maintaining the 50% CutMix frequency that EXP-021 showed is optimal.

**Sources**: CutMix paper, EXP-021 (frequency reduction hurts), common practice alpha=0.2-0.5

**Estimated Effort**: low — single constant change

**Risk Assessment**: Low. The model still sees CutMix at 50% frequency. Worst case: the lighter mixing doesn't provide enough regularization, losing ~0.2%.

## Chosen Idea
**Selected**: CutMix Alpha 0.5

**Why this idea**: The only remaining untried CutMix parameter. Qualitatively different from the prob reduction that failed — changes mixing intensity, not frequency.

**Hypothesis**: CutMix alpha 0.5 (U-shaped mixing distribution) will provide better convergence than alpha=1.0 by reducing the average mixing difficulty while maintaining regularization frequency, improving best_test_acc to ≥96.49%.
