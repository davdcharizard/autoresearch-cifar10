# Plan EXP-033: Alternating Flip + WD 4e-4
- **Created**: 2026-05-28
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-033.md

## Milestones
### Milestone 1: Apply changes
- [ ] Remove RandomHorizontalFlip, add alternating flip in training loop
- [ ] Change WEIGHT_DECAY from 5e-4 to 4e-4

### Milestone 2: Run and verify
- [ ] Confirm ~99 epochs at 16ms/step
- [ ] Check best_test_acc > 96.66%

## Code Changes
- **train.py** (line 146): Remove `transforms.RandomHorizontalFlip(),`
- **train.py** (training loop): Add `if epoch % 2 == 0: inputs = inputs.flip(-1)`
- **train.py** (line 24): Change `WEIGHT_DECAY = 5e-4` to `WEIGHT_DECAY = 4e-4`

## Verification Protocol
**Condition 1**: best_test_acc > 96.66%
