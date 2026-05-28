# Plan EXP-031: Nesterov + Reflect Padding
- **Created**: 2026-05-28
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-031.md

## Milestones
### Milestone 1: Apply two changes
- [ ] Add `nesterov=True` to SGD
- [ ] Add `padding_mode='reflect'` to RandomCrop

### Milestone 2: Run and verify
- [ ] Run experiment, confirm ~96 epochs at 16ms/step
- [ ] Check best_test_acc > 96.56%

## Code Changes
- **train.py** (line 172-174): Add `nesterov=True` to SGD
- **train.py** (line 145): Change `transforms.RandomCrop(32, padding=4)` to `transforms.RandomCrop(32, padding=4, padding_mode='reflect')`

## Configuration Changes
- nesterov: False → True
- RandomCrop padding_mode: 'constant' (default, zero-fill) → 'reflect'

## Execution Environment
- Method: `uv run python train.py > run.log 2>&1`
- Resources: single H20 GPU
- Estimated runtime: ~310-320s
- Log output: run.log

## Abort Criteria
- Loss NaN/inf, crash, epoch count < 90

## Verification Protocol
### Verification Procedure
**Condition 1**: best_test_acc > 96.56%
**Condition 2**: Clean completion
**Condition 3**: Max 1 eval per epoch
