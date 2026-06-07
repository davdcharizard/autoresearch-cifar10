# Plan EXP-008: k=4 + Stochastic Depth + EMA
- **Created**: 2026-05-28
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-008.md

## Milestones
### Milestone 1: Code changes
- [ ] Add stochastic depth to BasicBlock: survival probability linearly decreasing from 1.0 to (1-drop_rate)
- [ ] Pass block index and total blocks to each BasicBlock for survival probability calculation
- [ ] Only apply stochastic depth during training, not eval
- [ ] Verify ruff passes

### Milestone 2: Run and verify
- [ ] best_test_acc >= 95.83% (baseline 95.73% + 0.1%)

## Code Changes

- **train.py — BasicBlock stochastic depth**: Add `drop_rate` parameter. During training, with probability `drop_rate`, skip the residual path and pass input through identity. The drop_rate linearly increases from 0 for the first block to `max_drop_rate` for the last block.

  Implementation: In `BasicBlock.__init__`, add `self.drop_rate = drop_rate`. In `forward`, during training, sample from Bernoulli and if dropped, return `self.shortcut(x)` instead of the full residual computation.

- **train.py — ResNet._make_layer**: Pass block index for computing per-block drop rate. Total blocks = 3 * NUM_BLOCKS = 9. Block i gets drop_rate = max_drop_rate * (i / (total_blocks - 1)).

- **train.py — Config**: Add STOCHASTIC_DEPTH_RATE = 0.2.

## Configuration Changes
- STOCHASTIC_DEPTH_RATE: new, 0.2

## Execution Environment
- Method: `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`
- Single H20 GPU, ~8 min

## Abort Criteria
- Run exceeds 10 min, traceback, loss NaN/inf

## Verification Protocol
### Verification Procedure
1. Run completion, time budget, accuracy >= 95.83%, eval count
