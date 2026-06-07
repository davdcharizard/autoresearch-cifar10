# Plan EXP-036: channels_last + T_max=49 + LR clamp
- **Created**: 2026-06-04

## Code Changes
- **train.py**: Four changes:
  1. `model = model.to(memory_format=torch.channels_last)` before EMA deepcopy
  2. `inputs = inputs.to(device, memory_format=torch.channels_last, non_blocking=True)` in training loop
  3. Keep `COSINE_T_MAX = 49` (matches expected ~54 epochs with channels_last on this system)
  4. After `scheduler.step()`, add LR clamp: `if epoch > WARMUP_EPOCHS + COSINE_T_MAX: for pg in optimizer.param_groups: pg['lr'] = 0.0`

## Verification
1. `grep "^best_test_acc:" run.log` — must be >= 96.49%
