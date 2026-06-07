# Plan EXP-024: Label smoothing 0.05 + fixed numpy seed
- **Created**: 2026-05-29
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-024.md

## Milestones

### Milestone 1: Code changes
- [ ] Change LABEL_SMOOTHING from 0.1 to 0.05
- [ ] Add `np.random.seed(42)` after torch seed lines in main()

### Milestone 2: Training completes
- [ ] ~54 epochs, 300s budget

### Milestone 3: Verification
- [ ] best_test_acc >= 96.49%

## Code Changes
- **train.py**: Two changes:
  1. `LABEL_SMOOTHING = 0.1` → `LABEL_SMOOTHING = 0.05`
  2. Add `np.random.seed(42)` after `torch.cuda.manual_seed(42)` in main()

## Configuration Changes
- LABEL_SMOOTHING: 0.1 → 0.05
- numpy random seed: unfixed → 42

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
