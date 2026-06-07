# Plan EXP-041: torch.compile max-autotune + T_max=43
- **Created**: 2026-06-04

## Code Changes
- **train.py**:
  1. `COSINE_T_MAX = 49` → `COSINE_T_MAX = 43`
  2. `model = torch.compile(model)` → `model = torch.compile(model, mode='max-autotune')`
