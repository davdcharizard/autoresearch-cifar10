# Plan EXP-039: Progressive EMA decay + T_max=43
- **Created**: 2026-06-04

## Code Changes
- **train.py**:
  1. `COSINE_T_MAX = 49` → `COSINE_T_MAX = 43`
  2. In the EMA update section, replace fixed `EMA_DECAY` with progressive decay:
     ```python
     progress = min(epoch / (WARMUP_EPOCHS + COSINE_T_MAX), 1.0)
     ema_decay_val = 0.99 + (0.9999 - 0.99) * progress
     ```
     Then use `ema_decay_val` instead of `EMA_DECAY` in the EMA update.
