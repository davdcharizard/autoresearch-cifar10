# Plan EXP-021: CutMix probability reduction (0.5 → 0.3)
- **Created**: 2026-05-29
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-021.md

## Milestones

### Milestone 1: Code change
- [ ] Change CUTMIX_PROB from 0.5 to 0.3 in train.py
- [ ] Verify no other changes (all other hyperparameters identical)

### Milestone 2: Training completes
- [ ] Run experiment, confirm ~54 epochs and 300s budget

### Milestone 3: Verification
- [ ] best_test_acc >= 96.49%

## Code Changes
- **train.py**: Change `CUTMIX_PROB = 0.5` to `CUTMIX_PROB = 0.3`. Single line change. No other modifications.

## Configuration Changes
- CUTMIX_PROB: 0.5 → 0.3 (rationale: over-regularization hypothesis from EXP-006/011; 70% clean batches vs 50% accelerates convergence in ~54 epoch budget)

## Execution Environment
- Method: `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`
- Resources: single GPU, AMP
- Estimated runtime: ~5-6 minutes
- Log output: `run.log`

## Abort Criteria
- Loss divergence, crash, OOM
- Epoch count significantly different from ~54

## Verification Protocol

### Verification Procedure
1. Run experiment
2. `grep "^best_test_acc:" run.log` — must be >= 96.49%
3. `grep "^training_seconds:" run.log` — must be <= 300
4. `grep -c "eval ep" run.log` must equal num_epochs

### Informational Metrics (Optional)
- All standard metrics via `grep "^{metric}:" run.log`
