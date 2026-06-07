# Brainstorm EXP-016
**Created**: 2026-05-29
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Chosen Idea
**Selected**: Test-Time Augmentation (horizontal flip) in EMA model forward

Add TTA to the EMA model's forward method: when not training, compute logits for both the original input and its horizontal flip, average them. This is a free accuracy boost that compounds with all existing training improvements. Eval time doubles but is not counted in the 300s training budget.

**Hypothesis**: TTA with horizontal flip will improve from 95.73% to ~96.0% by averaging out left-right asymmetry noise in predictions.
