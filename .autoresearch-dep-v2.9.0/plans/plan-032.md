# Plan EXP-032: Alternating Flip Augmentation
- **Created**: 2026-05-28
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-032.md

## Milestones
### Milestone 1: Replace RandomHorizontalFlip with alternating flip
- [ ] Remove `transforms.RandomHorizontalFlip()` from train_tf
- [ ] Add conditional flip `inputs = inputs.flip(-1)` in training loop when `epoch % 2 == 0`, after GPU transfer

### Milestone 2: Run and verify
- [ ] Confirm ~99 epochs at 16ms/step
- [ ] Check best_test_acc > 96.66% (new baseline 96.56% + 0.1pp)

## Code Changes
- **train.py** (line 146): Remove `transforms.RandomHorizontalFlip(),` from transform pipeline
- **train.py** (training loop, after GPU transfer): Add `if epoch % 2 == 0: inputs = inputs.flip(-1)` between the GPU transfer and `optimizer.zero_grad()`

## Execution Environment
- Method: `uv run python train.py > run.log 2>&1`
- Resources: single H20 GPU
- Log output: run.log

## Abort Criteria
- Epoch count < 90, loss NaN, crash

## Verification Protocol
**Condition 1**: best_test_acc > 96.66% (baseline 96.56% + 0.1pp)
**Condition 2**: Clean completion
**Condition 3**: Max 1 eval per epoch
