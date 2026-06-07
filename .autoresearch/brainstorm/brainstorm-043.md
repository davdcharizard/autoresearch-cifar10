# Brainstorm EXP-043
**Created**: 2026-06-04
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Chosen Idea
**Selected**: Multi-step LR decay (milestones at epochs 30, 40, gamma=0.1) + warmup 5 epochs

Replaces CosineAnnealingLR. Step decay keeps LR=0.1 for 25 epochs after warmup (vs cosine which starts decaying immediately). More high-LR training enables broader loss landscape exploration before fine-tuning. The classic ResNet CIFAR-10 recipe uses step decay.
