# Plan EXP-006: k=4 + TrivialAugment + CutMix
- **Created**: 2026-05-28
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-006.md

## Milestones
### Milestone 1: Code changes
- [ ] Revert to k=4 (WIDTH_MULT=4, COSINE_T_MAX=49) from the autoresearch/dev baseline
- [ ] Add `transforms.TrivialAugmentWide()` before ToTensor in train transforms
- [ ] Keep CutMix in training loop
- [ ] Verify ruff passes

### Milestone 2: Run and verify
- [ ] best_test_acc >= 95.35% (baseline + 0.1%)

## Code Changes
- **train.py**: Add `transforms.TrivialAugmentWide()` to the training transform pipeline after RandomHorizontalFlip and before ToTensor. TrivialAugment operates on PIL images so it must go before ToTensor.

Note: the autoresearch/dev branch already has the k=4 + CutMix code (from EXP-004). EXP-005 changes were discarded. Only need to add TrivialAugment.

## Configuration Changes
- Augmentation: add TrivialAugmentWide (before ToTensor)
- All else unchanged from EXP-004 baseline (k=4, T_max=49, CutMix, AMP, compile)

## Execution Environment
- Method: `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`
- Single H20 GPU, ~8 min

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
