# Plan EXP-029: Channels_last + LR clamp after cosine
- **Created**: 2026-05-29
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-029.md

## Milestones
### Milestone 1: Code changes
- [ ] Add channels_last on model (before EMA deepcopy) and training inputs
- [ ] Add LR clamp after scheduler.step(): if epoch > WARMUP_EPOCHS + COSINE_T_MAX, set LR to 0
- [ ] Keep T_max=49, LR=0.1, all other params unchanged

### Milestone 2: Training completes
- [ ] ~59-64 epochs, 300s budget, no LR restart (verify best≈final)

### Milestone 3: Verification
- [ ] best_test_acc >= 96.49%

## Code Changes
- **train.py**: Three changes:
  1. `model = model.to(memory_format=torch.channels_last)` after model creation, before EMA deepcopy
  2. `inputs = inputs.to(device, memory_format=torch.channels_last, non_blocking=True)` in training loop
  3. After `scheduler.step()`, add LR clamp:
     ```python
     if epoch > WARMUP_EPOCHS + COSINE_T_MAX:
         for pg in optimizer.param_groups:
             pg['lr'] = 0.0
     ```

## Configuration Changes
- None. T_max=49, LR=0.1, all baseline values.

## Execution Environment
- Method: `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`
- Log output: `run.log`

## Abort Criteria
- Loss divergence, crash

## Verification Protocol
### Verification Procedure
1. `grep "^best_test_acc:" run.log` — must be >= 96.49%
2. `grep "^training_seconds:" run.log` — must be <= 300
3. Verify no LR restart: best/final gap should be < 0.3%
