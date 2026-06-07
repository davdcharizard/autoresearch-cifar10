# Plan EXP-023: Weight decay 1e-3
- **Created**: 2026-05-29
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-023.md

## Milestones

### Milestone 1: Code change
- [ ] Change WEIGHT_DECAY from 5e-4 to 1e-3

### Milestone 2: Training completes
- [ ] ~54 epochs, 300s budget

### Milestone 3: Verification
- [ ] best_test_acc >= 96.49%

## Code Changes
- **train.py**: `WEIGHT_DECAY = 5e-4` → `WEIGHT_DECAY = 1e-3`

## Configuration Changes
- WEIGHT_DECAY: 5e-4 → 1e-3

## Execution Environment
- Method: `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`
- Resources: single GPU, AMP
- Estimated runtime: ~5-6 minutes
- Log output: `run.log`

## Abort Criteria
- Loss divergence, crash, OOM

## Verification Protocol

### Verification Procedure
1. `grep "^best_test_acc:" run.log` — must be >= 96.49%
2. `grep "^training_seconds:" run.log` — must be <= 300

### Informational Metrics (Optional)
- All standard metrics via `grep "^{metric}:" run.log`
