# Plan EXP-031: Channels_last + LR cooldown at 1e-4
- **Created**: 2026-05-29
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-031.md

## Code Changes
- **train.py**: Three changes:
  1. `model = model.to(memory_format=torch.channels_last)` before EMA deepcopy
  2. `inputs = inputs.to(device, memory_format=torch.channels_last, non_blocking=True)` in training loop
  3. After `scheduler.step()`, clamp LR to 1e-4 instead of allowing restart:
     ```python
     if epoch > WARMUP_EPOCHS + COSINE_T_MAX:
         for pg in optimizer.param_groups:
             pg['lr'] = 1e-4
     ```

## Verification Protocol
### Verification Procedure
1. `grep "^best_test_acc:" run.log` — must be >= 96.49%
