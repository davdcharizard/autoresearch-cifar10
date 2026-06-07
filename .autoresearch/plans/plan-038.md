# Plan EXP-038: Full speedup stack
- **Created**: 2026-06-04

## Code Changes
- **train.py**:
  1. `COSINE_T_MAX = 49` → `COSINE_T_MAX = 43` (align to current system)
  2. channels_last on model + training inputs
  3. Remove per-step torch.cuda.synchronize(), add epoch-level sync
  4. Move loss.item() to only execute every 50 steps (for logging)
  5. Track training time at epoch boundaries
  6. LR clamp after cosine completion
