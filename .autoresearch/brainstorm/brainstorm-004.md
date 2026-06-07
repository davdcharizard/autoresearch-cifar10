# Brainstorm EXP-004
**Created**: 2026-05-28
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

No new sources — continuing the width trajectory established across EXP-000 through EXP-003.

## Experimental History Review

**Trajectory** (4 experiments):
- BASE: 91.81% (270K, 89 ep)
- EXP-000: 92.10% (270K, 89 ep) — recipe
- EXP-001: 94.03% (1.08M, 78 ep) — k=2 + AMP + compile
- EXP-002: 94.09% (2.4M, 62 ep) — k=3, broken T_max → no-improvement
- EXP-003: 94.80% (2.4M, 65 ep) — k=3, fixed T_max + CutMix

**Width scaling trend**: k=1→k=2: +1.93%, k=2→k=3: +0.77%. Diminishing returns but still positive. VRAM at k=3 only 425MB (< 0.5% of 98GB available).

**Epoch scaling**: k=1: 89ep, k=2: 78ep, k=3: 65ep. Ratio k3/k2=0.83, k2/k1=0.88. At k=4: est ~65*0.83 = ~54 epochs. T_max should be ~49 (54 - 5 warmup).

## Candidate Ideas

### 1. k=4 Width + T_max=49

**Summary**: WIDTH_MULT=4 ({64,128,256}, ~4.3M params). T_max=49 based on epoch scaling trend. Keep CutMix, AMP, torch.compile, Nesterov, projection shortcuts, label smoothing.

**Reasoning**: Width has been the most reliable lever. k=4 is 1.78x more capacity than k=3. Expected ~54 epochs gives enough training time. VRAM is not a constraint.

**Sources**: EXP-001/003 width scaling trend

**Estimated Effort**: low (single config change)

**Risk Assessment**: Low. Same architecture pattern as k=2 and k=3, both of which worked. Only risk is epoch estimate being off, but T_max mismatch of a few epochs is tolerable.

### 2. k=3 + Deeper (ResNet-32, NUM_BLOCKS=5)

**Summary**: Keep k=3 but add depth: NUM_BLOCKS=5 for 32 layers. ~4.0M params. More feature extraction stages.

**Reasoning**: WideResNet paper shows depth still helps, just less efficiently than width. Untried dimension.

**Sources**: WideResNet paper, He et al. 2015

**Estimated Effort**: low

**Risk Assessment**: Medium. More layers without width increase → more sequential compute → fewer epochs. May get ~45 epochs.

## Idea Evaluation

k=4 is the natural continuation of the proven width trajectory. k=3+depth is an untested dimension change. Width has consistently delivered; depth is less efficient per the WideResNet paper. k=4 is also simpler (single config change).

## Chosen Idea

**Selected**: k=4 Width + T_max=49

**Why this idea**: Direct continuation of the most reliable improvement direction. Single variable change from the proven k=3 baseline.

**Hypothesis**: k=4 ({64,128,256}, ~4.3M params) will improve best_test_acc from 94.80% to approximately 95.0-95.5%, continuing the width scaling trend with diminishing but positive returns.
