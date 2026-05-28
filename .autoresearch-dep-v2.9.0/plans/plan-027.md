# Plan EXP-027: Nesterov + Shortened Warmup (3 epochs)
- **Created**: 2026-05-28
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-027.md

## Milestones

### Milestone 1: Enable Nesterov and shorten warmup
- [ ] Add `nesterov=True` to `optim.SGD()` in train.py
- [ ] Change `WARMUP_EPOCHS = 5` to `WARMUP_EPOCHS = 3` in train.py
- [ ] Verify no other code changes needed

### Milestone 2: Run experiment and capture output
- [ ] Run `uv run python train.py > run.log 2>&1` and confirm training starts
- [ ] Confirm training completes within 300s budget with ~96 epochs

### Milestone 3: Verify results
- [ ] Extract best_test_acc from run.log
- [ ] Check best_test_acc > 96.56% (baseline 96.46% + 0.1pp threshold)

## Code Changes
- **train.py** (line 172-174): Add `nesterov=True` to SGD constructor
- **train.py** (line 179): Change `WARMUP_EPOCHS = 5` to `WARMUP_EPOCHS = 3`

## Configuration Changes
- nesterov: False → True — enables look-ahead gradient computation
- WARMUP_EPOCHS: 5 → 3 — shortens LR warmup, freeing 2 epochs of productive high-LR training

## Execution Environment
- Method: local command `uv run python train.py > run.log 2>&1`
- Resources: single H20 GPU (existing setup)
- Estimated runtime: ~310-320s total
- Log output: stdout+stderr captured to `run.log` in project root
- Tool skill: none (local execution)

## Abort Criteria
- No output in run.log after 60s
- Loss goes to NaN/inf — indicates instability from shorter warmup
- Epoch count drops below 90 — would indicate unexpected throughput cost
- Training crashes with any error

## Verification Protocol

### Verification Procedure

**Condition 1: best_test_acc > 96.56%** (baseline 96.46% + 0.1pp threshold)
- Command: `grep "^best_test_acc:" run.log | awk '{print $2}' | tr -d '%'`
- Pass: extracted value > 96.56
- Fail: value <= 96.56 or missing
- Timeout: 10s

**Condition 2: Clean completion**
- Command: `grep "^best_test_acc:" run.log`
- Pass: line exists with a numeric value
- Fail: missing or malformed
- Timeout: 10s

**Condition 3: Max 1 eval per epoch**
- Command: `grep -c "eval ep" run.log` vs `grep "^num_epochs:" run.log | awk '{print $2}'`
- Pass: eval count <= epoch count
- Fail: eval count > epoch count
- Timeout: 10s

### Informational Metrics (Optional)
- final_test_acc: `grep "^final_test_acc:" run.log | awk '{print $2}'`
- num_epochs: `grep "^num_epochs:" run.log | awk '{print $2}'`
- training_seconds: `grep "^training_seconds:" run.log | awk '{print $2}'`
- peak_vram_mb: `grep "^peak_vram_mb:" run.log | awk '{print $2}'`
