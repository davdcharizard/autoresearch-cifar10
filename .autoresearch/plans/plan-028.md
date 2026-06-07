# Plan EXP-028: Peak LR 0.15
- **Created**: 2026-05-29
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-028.md

## Milestones
### Milestone 1: Code change
- [ ] Change LR from 0.1 to 0.15

### Milestone 2: Training completes
- [ ] ~54 epochs, 300s budget

### Milestone 3: Verification
- [ ] best_test_acc >= 96.49%

## Code Changes
- **train.py**: `LR = 0.1` → `LR = 0.15`

## Configuration Changes
- LR: 0.1 → 0.15

## Execution Environment
- Method: `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`
- Log output: `run.log`

## Abort Criteria
- Loss divergence (higher LR increases divergence risk)

## Verification Protocol
### Verification Procedure
1. `grep "^best_test_acc:" run.log` — must be >= 96.49%
2. `grep "^training_seconds:" run.log` — must be <= 300
