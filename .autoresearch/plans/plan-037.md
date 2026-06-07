# Plan EXP-037: Epoch-level sync + channels_last + T_max=49 + LR clamp
- **Created**: 2026-06-04

## Code Changes
- **train.py**: Multiple changes:
  1. channels_last on model (before EMA deepcopy) + training inputs
  2. Remove `torch.cuda.synchronize()` from the inner training loop
  3. Replace per-step time tracking with epoch-level: track `epoch_start = time.time()` before each epoch's inner loop, and `torch.cuda.synchronize()` + `epoch_time = time.time() - epoch_start` after the inner loop completes. Accumulate to `total_training_time`.
  4. Budget check at epoch boundary: `if total_training_time >= TIME_BUDGET_S: break`
  5. Keep per-step `dt` for logging but use `time.time()` without synchronize (approximate)
  6. LR clamp after cosine: `if epoch > WARMUP_EPOCHS + COSINE_T_MAX: pg['lr'] = 0.0`
