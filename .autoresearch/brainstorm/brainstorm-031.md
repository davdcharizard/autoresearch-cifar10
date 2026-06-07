# Brainstorm EXP-031
**Created**: 2026-05-29
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Experimental History Review
- EXP-019: channels_last + T_max=49 → LR RESTART (96.28% best, 95.38% final)
- EXP-029: channels_last + LR clamp to 0 → NO LEARNING in extra epochs (96.25% best=final)
- MISSING MIDDLE: channels_last + LR clamp to SMALL CONSTANT (e.g., 1e-4) → gentle refinement in extra epochs

## Chosen Idea
**Selected**: Channels_last + LR cooldown at 1e-4

**Hypothesis**: After cosine completes at epoch 54, continuing at LR=1e-4 for ~10 extra epochs provides gentle weight refinement that the EMA model benefits from, pushing accuracy above 96.49%.
