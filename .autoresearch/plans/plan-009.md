# Plan EXP-009: Pre-activation Blocks at k=4
- **Created**: 2026-05-28
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-009.md

## Milestones
### Milestone 1: Code changes
- [ ] Convert BasicBlock to pre-activation (BN→ReLU→Conv→BN→ReLU→Conv→+shortcut)
- [ ] Add bn_final before pooling in ResNet.forward
- [ ] Remove unused bn1 from ResNet constructor
- [ ] Projection shortcut: apply after shared BN+ReLU
- [ ] Verify ruff passes

### Milestone 2: Run and verify
- [ ] best_test_acc >= 95.83% (baseline 95.73% + 0.1%)

## Code Changes
Same pre-activation architecture from EXP-005 but at k=4 instead of k=6.

## Execution Environment
- Method: `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`
- Single H20 GPU

## Verification Protocol
### Verification Procedure
1. Run completion, time budget, accuracy >= 95.83%, eval count
