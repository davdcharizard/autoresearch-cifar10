# Brainstorm EXP-036
**Created**: 2026-06-04
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Key Context
- Current system gets 49 epochs without channels_last (EXP-035)
- With channels_last, expect ~54-55 epochs (~10% speedup)
- T_max=43 gives 95.89% with 49 epochs (EXP-035 baseline for this system)
- Need T_max=48-49 to match channels_last epoch count on this system
- Must clamp LR after cosine to prevent CosineAnnealingLR periodic restart

## Chosen Idea
**Selected**: channels_last + T_max=49 + LR clamp after cosine

**Why T_max=49**: With channels_last on this slower system, expect ~54-55 epochs. 5 warmup + 49 cosine = 54. This is well-aligned. LR clamp prevents restart if we get 55+.

**Hypothesis**: channels_last gives ~5 more cosine epochs (49 vs 43), providing more training at productive LR, pushing accuracy from 95.89% to ≥96.49%.
