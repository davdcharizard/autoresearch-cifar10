# Plan EXP-040: T_max=43 + np.random.seed(0)
- **Created**: 2026-06-04

## Code Changes
- **train.py**: Two changes:
  1. `COSINE_T_MAX = 49` → `COSINE_T_MAX = 43`
  2. Add `np.random.seed(0)` after `torch.cuda.manual_seed(42)`
