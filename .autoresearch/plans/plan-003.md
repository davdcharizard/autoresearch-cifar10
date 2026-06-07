# Plan EXP-003: k=3 + T_max=57 + CutMix
- **Created**: 2026-05-28
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-003.md

## Milestones

### Milestone 1: Code changes
- [ ] WIDTH_MULT=3, COSINE_T_MAX=57
- [ ] Replace CutOut with CutMix (in training loop)
- [ ] Remove CutOut class and CUTOUT_SIZE constant
- [ ] Verify ruff passes

### Milestone 2: Run experiment
- [ ] Run, confirm completion, ~62 epochs expected

### Milestone 3: Verify
- [ ] best_test_acc >= 94.13% (baseline 94.03% + 0.1%)

## Code Changes

Same as EXP-002 but with static COSINE_T_MAX=57 instead of dynamic calibration:
- WIDTH_MULT: 2 → 3
- COSINE_T_MAX: 55 → 57
- CutOut → CutMix(alpha=1.0, p=0.5) in training loop
- Remove CutOut class and CUTOUT_SIZE

## Configuration Changes
- WIDTH_MULT: 2 → 3
- COSINE_T_MAX: 55 → 57
- CutOut → CutMix

## Execution Environment
- Method: `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`
- Resources: single H20 GPU
- Estimated runtime: ~8 min
- Log output: run.log

## Abort Criteria
- Run exceeds 10 min, traceback/CUDA error, loss NaN/inf

## Verification Protocol

### Verification Procedure
1. `grep "^best_test_acc:" run.log` — FAIL if empty
2. `grep "^training_seconds:" run.log` — FAIL if > 300
3. best_test_acc >= 94.13% — FAIL otherwise
4. eval count == num_epochs

### Informational Metrics (Optional)
All summary metrics via grep
