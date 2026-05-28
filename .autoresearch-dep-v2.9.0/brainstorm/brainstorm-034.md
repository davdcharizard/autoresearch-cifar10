# Brainstorm EXP-034
**Created**: 2026-05-28
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Experimental History Review
- **Baseline**: 96.56% (EXP-031, Nesterov+reflect)
- **EXP-032**: Alternating flip alone → 96.64% (+0.08pp). Best near-miss.
- **EXP-033**: Alternating flip + WD 4e-4 → 96.52% (-0.04pp). Reducing regularization hurts.
- **Key insight**: Don't reduce regularization. If anything, slightly increase it to balance the changed augmentation pattern.

## Chosen Idea
**Selected**: Alternating Flip + RandomErasing p=0.30 (from 0.25)

**Why**: Alternating flip gave +0.08pp. EXP-033 showed reducing regularization negates the benefit. Instead of reducing, slightly INCREASE regularization through more frequent random erasing (25% → 30%). The additional occlusion regularization may stabilize the model for the deterministic flip pattern and push the remaining 0.02pp.

**Hypothesis**: Alternating flip + slightly higher RandomErasing (p=0.30) will reach 96.66%+ by balancing the changed flip pattern with marginally stronger input-space regularization.
