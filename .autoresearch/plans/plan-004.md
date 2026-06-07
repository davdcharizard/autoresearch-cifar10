# Plan EXP-004: k=4 Width + T_max=49
- **Created**: 2026-05-28
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-004.md

## Milestones

### Milestone 1: Code changes
- [ ] WIDTH_MULT: 3 → 4
- [ ] COSINE_T_MAX: 57 → 49
- [ ] Verify ruff passes

### Milestone 2: Run and verify
- [ ] Run, confirm ~54 epochs
- [ ] best_test_acc >= 94.90% (baseline 94.80% + 0.1%)

## Code Changes
- **train.py**: Change WIDTH_MULT from 3 to 4, COSINE_T_MAX from 57 to 49. Two-line change.

## Configuration Changes
- WIDTH_MULT: 3 → 4 ({64,128,256}, ~4.3M params)
- COSINE_T_MAX: 57 → 49 (based on epoch scaling trend)

## Execution Environment
- Method: `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`
- Resources: single H20 GPU
- Estimated runtime: ~8 min

## Abort Criteria
- Run exceeds 10 min, traceback, loss NaN/inf

## Verification Protocol
### Verification Procedure
1. Run completion check
2. Time budget <= 300
3. best_test_acc >= 94.90%
4. Eval count == epochs
### Informational Metrics (Optional)
All summary metrics
