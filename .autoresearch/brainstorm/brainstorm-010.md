# Brainstorm EXP-010
**Created**: 2026-05-28
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Experimental History Review
Baseline: 95.73% (EXP-007). 5 consecutive failures from architectural/augmentation changes. Need pure hyperparameter tuning.

## Chosen Idea
**Selected**: Batch 256 + LR 0.2 + T_max=37

Simple hyperparameter change: double batch size (128→256) with linear LR scaling (0.1→0.2). Larger batch = better GPU utilization on the H20. T_max=37 estimated from fewer epochs at larger batch.

**Hypothesis**: Better GPU utilization and faster effective convergence will improve from 95.73% to ~95.9-96.1%.
