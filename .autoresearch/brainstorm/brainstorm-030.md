# Brainstorm EXP-030
**Created**: 2026-05-29
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Experimental History Review
- 31 experiments, 96.39% baseline, 14 consecutive failures
- RandomCrop padding=4 is the DEFAULT CIFAR-10 value — never tuned
- Padding=4 means max ±4px shift on 32x32 images. Padding=6 allows ±6px, 50% more diversity

## Chosen Idea
**Selected**: RandomCrop padding 6

**Hypothesis**: Larger crop padding (6 vs 4) provides more translation augmentation diversity, improving generalization to ≥96.49%.
