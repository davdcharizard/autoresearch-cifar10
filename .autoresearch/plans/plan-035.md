# Plan EXP-035: T_max=43 (schedule alignment for current system)
- **Created**: 2026-06-04
- **Goal**: goals/maximize-cifar10-test-accuracy.md

## Code Changes
- **train.py**: `COSINE_T_MAX = 49` → `COSINE_T_MAX = 43`

This restores cosine schedule alignment: 5 warmup + 43 cosine = 48 epochs total, matching the current system's epoch count.

## Verification
1. `grep "^best_test_acc:" run.log` — must be >= 96.49%
2. Check best≈final (confirms T_max alignment)
