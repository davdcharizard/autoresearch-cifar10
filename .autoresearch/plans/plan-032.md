# Plan EXP-032: RandomCrop padding_mode='reflect'
- **Created**: 2026-06-04
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-032.md

## Code Changes
- **train.py**: `transforms.RandomCrop(32, padding=4)` → `transforms.RandomCrop(32, padding=4, padding_mode='reflect')`

## Verification Protocol
### Verification Procedure
1. `grep "^best_test_acc:" run.log` — must be >= 96.49%
2. `grep "^training_seconds:" run.log` — must be <= 300
