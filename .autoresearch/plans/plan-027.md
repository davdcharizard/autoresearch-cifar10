# Plan EXP-027: CutMix alpha 0.5
- **Created**: 2026-05-29
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-027.md

## Milestones
### Milestone 1: Code change
- [ ] Change CUTMIX_ALPHA from 1.0 to 0.5

### Milestone 2: Training completes
- [ ] ~54 epochs, 300s budget

### Milestone 3: Verification
- [ ] best_test_acc >= 96.49%

## Code Changes
- **train.py**: `CUTMIX_ALPHA = 1.0` → `CUTMIX_ALPHA = 0.5`

## Configuration Changes
- CUTMIX_ALPHA: 1.0 → 0.5

## Execution Environment
- Method: `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`
- Estimated runtime: ~5-6 minutes
- Log output: `run.log`

## Abort Criteria
- Loss divergence, crash

## Verification Protocol
### Verification Procedure
1. `grep "^best_test_acc:" run.log` — must be >= 96.49%
2. `grep "^training_seconds:" run.log` — must be <= 300
### Informational Metrics (Optional)
- All standard metrics via `grep "^{metric}:" run.log`
