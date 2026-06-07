# Plan EXP-030: RandomCrop padding 6
- **Created**: 2026-05-29
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-030.md

## Milestones
### Milestone 1: Code change
- [ ] Change RandomCrop padding from 4 to 6

### Milestone 2: Training completes
### Milestone 3: Verification
- [ ] best_test_acc >= 96.49%

## Code Changes
- **train.py**: `transforms.RandomCrop(32, padding=4)` → `transforms.RandomCrop(32, padding=6)`

## Execution Environment
- Method: `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`

## Verification Protocol
### Verification Procedure
1. `grep "^best_test_acc:" run.log` — must be >= 96.49%
