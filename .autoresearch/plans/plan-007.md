# Plan EXP-007: k=4 + EMA + Weight Decay 5e-4
- **Created**: 2026-05-28
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-007.md

## Milestones
### Milestone 1: Code changes
- [ ] Change WEIGHT_DECAY from 1e-4 to 5e-4
- [ ] Add EMA: after each optimizer step, update ema_model weights with decay=0.999
- [ ] Evaluate with ema_model instead of model
- [ ] Verify ruff passes

### Milestone 2: Run and verify
- [ ] best_test_acc >= 95.35%

## Code Changes

- **train.py — WEIGHT_DECAY**: Change from 1e-4 to 5e-4.

- **train.py — EMA**: After creating the model and torch.compile:
  1. Create `ema_model` as a deep copy of the model (before compile)
  2. After each `scaler.update()`, update EMA weights:
     `for p_ema, p in zip(ema_model.parameters(), model.parameters()): p_ema.data.mul_(0.999).add_(p.data, alpha=0.001)`
  3. Use `ema_model` for evaluation instead of `model`

  Important: EMA model should NOT be compiled (it's only used for eval). The main `model` stays compiled for training speed.

## Configuration Changes
- WEIGHT_DECAY: 1e-4 → 5e-4
- EMA decay: 0.999 (new)

## Execution Environment
- Method: `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`
- Single H20 GPU, ~8 min

## Abort Criteria
- Run exceeds 10 min, traceback, loss NaN/inf

## Verification Protocol
### Verification Procedure
1. Run completion, time budget, accuracy >= 95.35%, eval count
### Informational Metrics (Optional)
All summary metrics
