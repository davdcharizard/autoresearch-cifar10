# Plan EXP-005: k=6 + Pre-activation Blocks
- **Created**: 2026-05-28
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-005.md

## Milestones

### Milestone 1: Code changes
- [ ] WIDTH_MULT: 4 → 6
- [ ] COSINE_T_MAX: 49 → 35 (est ~40 epochs - 5 warmup)
- [ ] Convert BasicBlock to pre-activation: BN→ReLU→Conv instead of Conv→BN→ReLU
- [ ] Verify ruff passes

### Milestone 2: Run and verify
- [ ] Run, confirm ~40 epochs
- [ ] best_test_acc >= 95.35% (baseline 95.25% + 0.1%)

## Code Changes

- **train.py — WIDTH_MULT**: 4 → 6. Channels become {96, 192, 384}. ~9.7M params.

- **train.py — COSINE_T_MAX**: 49 → 35 (estimated ~40 epochs minus 5 warmup).

- **train.py — Pre-activation BasicBlock**: Restructure the forward path:
  ```
  Current: x → Conv1→BN1→ReLU → Conv2→BN2 → +shortcut → ReLU
  New:     x → BN1→ReLU→Conv1 → BN2→ReLU→Conv2 → +shortcut
  ```
  The shortcut also changes: projection shortcut applies after the shared BN+ReLU.
  The initial conv1+bn1 in ResNet stays post-activation (standard practice for pre-act ResNets).

## Configuration Changes
- WIDTH_MULT: 4 → 6
- COSINE_T_MAX: 49 → 35
- Block type: post-activation → pre-activation

## Execution Environment
- Method: `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`
- Single H20 GPU, ~8-9 min runtime

## Abort Criteria
- Run exceeds 10 min, traceback, loss NaN/inf

## Verification Protocol
### Verification Procedure
1. Run completion
2. Time budget <= 300
3. best_test_acc >= 95.35%
4. Eval count == epochs

### Informational Metrics (Optional)
All summary metrics
