# Plan EXP-030: Proper Per-Channel Std Normalization
- **Created**: 2026-05-28
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-030.md

## Milestones

### Milestone 1: Change std normalization
- [ ] Change `std = (1, 1, 1)` to `std = (0.2470, 0.2435, 0.2616)` in train.py

### Milestone 2: Run experiment
- [ ] Run `uv run python train.py > run.log 2>&1`
- [ ] Confirm ~96-98 epochs, 16ms/step

### Milestone 3: Verify results
- [ ] Check best_test_acc > 96.56%

## Code Changes
- **train.py** (line 141): Change `(1, 1, 1)` to `(0.2470, 0.2435, 0.2616)`

## Configuration Changes
- Normalization std: (1, 1, 1) → (0.2470, 0.2435, 0.2616)

## Execution Environment
- Method: `uv run python train.py > run.log 2>&1`
- Resources: single H20 GPU
- Estimated runtime: ~310-320s
- Log output: run.log

## Abort Criteria
- Loss NaN/inf in first 100 steps
- Per-step time > 20ms
- Training crash

## Verification Protocol

### Verification Procedure
**Condition 1: best_test_acc > 96.56%**
- Command: `grep "^best_test_acc:" run.log | awk '{print $2}' | tr -d '%'`
- Pass: > 96.56

**Condition 2: Clean completion**
- Command: `grep "^best_test_acc:" run.log`

**Condition 3: Max 1 eval per epoch**
- Command: eval count <= epoch count
