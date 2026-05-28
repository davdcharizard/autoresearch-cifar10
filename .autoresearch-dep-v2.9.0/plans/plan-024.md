# Plan EXP-024: BN Bias 64x LR Multiplier
- **Created**: 2026-05-28
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-024.md

## Milestones

### Milestone 1: Implement BN bias parameter group separation
- [ ] Identify all BatchNorm bias parameters in the model by iterating named parameters
- [ ] Create two optimizer parameter groups: `norm_biases` (BN bias params, lr=LR*64) and `other_params` (everything else, lr=LR)
- [ ] Set weight_decay=0 for the BN bias group (biases should not be decayed)
- [ ] Verify the optimizer has two parameter groups and the total parameter count matches

### Milestone 2: Run experiment and capture output
- [ ] Run `uv run python train.py > run.log 2>&1` and confirm training starts
- [ ] Confirm training completes within 300s budget with ~99 epochs (zero throughput cost)

### Milestone 3: Verify results
- [ ] Extract best_test_acc from run.log
- [ ] Check best_test_acc > 96.56% (baseline 96.46% + 0.1pp threshold)

## Code Changes
- **train.py** (lines 172-174): Replace `optimizer = optim.SGD(model.parameters(), ...)` with parameter group construction. Iterate `model.named_parameters()` to separate BN bias parameters (those in BatchNorm modules with name ending in `.bias`) from all other parameters. Create two param groups:
  1. `norm_biases`: BN bias params with `lr=LR * 64.0` and `weight_decay=0`
  2. `other_params`: all remaining params with `lr=LR` and `weight_decay=WEIGHT_DECAY`

  The identification logic: a parameter is a BN bias if its name contains `bn` and ends with `.bias`. In our ResNet, BN layers are named `bn1`, `bn2` in BasicBlocks and `bn1` in the stem — so matching `"bn"` in the name and `".bias"` suffix captures exactly the 19 BN bias parameters (18 in blocks + 1 in stem).

  The existing `LambdaLR` scheduler applies the cosine warmup+decay multiplicatively to all param groups uniformly, so the 64x ratio is maintained throughout training.

## Configuration Changes
- BN_BIAS_LR_MULT: (new) 64.0 — from airbench96 recipe (bias_scaler=64.0)
- BN bias weight_decay: WEIGHT_DECAY (5e-4) -> 0 — biases generally should not be decayed, and the 64x LR already provides strong regularization pressure through parameter magnitude

## Execution Environment
- Method: local command `uv run python train.py > run.log 2>&1`
- Resources: single H20 GPU (existing setup)
- Estimated runtime: ~310-320s total (300s training budget + ~10-20s startup/eval)
- Log output: stdout+stderr captured to `run.log` in project root
- Tool skill: none (local execution)

## Abort Criteria
- No output in run.log after 60s — indicates crash or hang
- Loss goes to NaN/inf in first 100 steps — indicates LR too aggressive for BN biases
- Training crashes with CUDA error or OOM — would indicate unexpected memory issue (unlikely since no new computation)
- Epoch count drops significantly below 99 — would indicate unexpected throughput cost (should be zero)

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
