# Plan EXP-034: persistent_workers=True
- **Created**: 2026-06-04
- **Goal**: goals/maximize-cifar10-test-accuracy.md

## Code Changes
- **train.py**: Add `persistent_workers=True` to the DataLoader constructor. This keeps workers alive between epochs.

## Verification
1. `grep "^best_test_acc:" run.log` — must be >= 96.49%
2. Check num_epochs — should be >= 54 (proving data/ is cached and workers are persistent)
