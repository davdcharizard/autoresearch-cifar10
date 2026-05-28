# Plan EXP-026: Nesterov Momentum
- **Created**: 2026-05-28
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-026.md

## Milestones

### Milestone 1: Enable Nesterov momentum
- [ ] Add `nesterov=True` to the `optim.SGD()` call in train.py
- [ ] Verify no other code changes needed (Nesterov is transparent to the rest of the training loop)

### Milestone 2: Run experiment and capture output
- [ ] Run `uv run python train.py > run.log 2>&1` and confirm training starts
- [ ] Confirm training completes within 300s budget with ~99 epochs (zero throughput cost expected)

### Milestone 3: Verify results
- [ ] Extract best_test_acc from run.log
- [ ] Check best_test_acc > 96.56% (baseline 96.46% + 0.1pp threshold)

## Code Changes
- **train.py** (line 172-174): Add `nesterov=True` to the SGD constructor. The current code:
  ```python
  optimizer = optim.SGD(
      model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
  )
  ```
  Becomes:
  ```python
  optimizer = optim.SGD(
      model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY,
      nesterov=True,
  )
  ```
  This is the only change. Nesterov momentum is fully transparent to the LambdaLR scheduler, GradScaler, and all other training loop components.

## Configuration Changes
- nesterov: False (PyTorch default) → True — enables look-ahead gradient computation in SGD momentum update

## Execution Environment
- Method: local command `uv run python train.py > run.log 2>&1`
- Resources: single H20 GPU (existing setup)
- Estimated runtime: ~310-320s total (300s training budget + ~10-20s startup/eval)
- Log output: stdout+stderr captured to `run.log` in project root
- Tool skill: none (local execution)

## Abort Criteria
- No output in run.log after 60s — indicates crash or hang
- Loss goes to NaN/inf — indicates numerical instability (unlikely with Nesterov)
- Epoch count drops below 97 — would indicate unexpected throughput cost from Nesterov
- Training crashes with any error

## Verification Protocol

### Verification Procedure

**Condition 1: best_test_acc > 96.56%** (baseline 96.46% + 0.1pp threshold)
- Command: `grep "^best_test_acc:" run.log | awk '{print $2}' | tr -d '%'`
- Pass: extracted value > 96.56
- Fail: value <= 96.56 or missing
- Timeout: 10s

**Condition 2: Clean completion**
- Command: `grep "^best_test_acc:" run.log` (non-empty means summary block printed)
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
