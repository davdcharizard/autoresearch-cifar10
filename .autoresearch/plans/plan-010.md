# Plan EXP-010: Batch 256 + LR 0.2
- **Created**: 2026-05-28
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-010.md

## Milestones
- [ ] BATCH_SIZE=256, LR=0.2, COSINE_T_MAX=37
- [ ] Run and verify best_test_acc >= 95.83%

## Code Changes
Three-line config change in train.py hyperparameters.

## Verification Protocol
1. Run completion, time budget, accuracy >= 95.83%, eval count
