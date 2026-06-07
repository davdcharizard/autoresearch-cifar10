# Plan EXP-026: Width k=5 with calibrated T_max
- **Created**: 2026-05-29
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-026.md

## Milestones

### Milestone 1: Code changes
- [ ] Change WIDTH_MULT from 4 to 5
- [ ] Add dynamic T_max calibration: measure epoch 2 time, estimate total epochs, set T_max

### Milestone 2: Training completes
- [ ] Run full experiment within 300s budget

### Milestone 3: Verification
- [ ] best_test_acc >= 96.49%

## Code Changes
- **train.py**: Two changes:
  1. `WIDTH_MULT = 4` → `WIDTH_MULT = 5`
  2. Add dynamic T_max calibration after epoch 2 completes. Measure epoch 2 wall time (excluding compile overhead in epoch 1). Calculate: `expected_epochs = int(TIME_BUDGET_S / epoch2_time)` then `T_max = max(expected_epochs - WARMUP_EPOCHS, 1)`. Replace the static cosine scheduler with a new one using the calibrated T_max.

   Implementation: after epoch 2's training loop, before scheduler.step(), check if epoch == 2. If so:
   - Calculate per-epoch time from epoch 2
   - Estimate remaining epochs
   - Create new cosine scheduler with calibrated T_max
   - Recreate SequentialLR
   - Print the calibrated T_max for logging

## Configuration Changes
- WIDTH_MULT: 4 → 5
- COSINE_T_MAX: static 49 → dynamically calibrated after epoch 2

## Execution Environment
- Method: `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`
- Estimated runtime: ~5-6 minutes
- Log output: `run.log`

## Abort Criteria
- Loss divergence, crash, OOM (k=5 needs more memory)
- Fewer than 30 epochs (model too slow to converge)

## Verification Protocol

### Verification Procedure
1. `grep "^best_test_acc:" run.log` — must be >= 96.49%
2. `grep "^training_seconds:" run.log` — must be <= 300

### Informational Metrics (Optional)
- All standard metrics via `grep "^{metric}:" run.log`
