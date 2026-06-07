# Report EXP-005: k=6 + Pre-activation Blocks
- **Created**: 2026-05-28
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-005.md
- **Plan**: plans/plan-005.md

## Goal
Maximize CIFAR-10 test accuracy. Baseline: 95.25% (EXP-004, k=4).

## Idea & Hypothesis
Widen to k=6 ({96,192,384}, 9.7M params) with pre-activation blocks. Hypothesis: 95.5-96%.

## Approach
WIDTH_MULT=6, COSINE_T_MAX=35, pre-activation blocks (BN→ReLU→Conv), CutMix, AMP, torch.compile.

## Execution
Single run, 32 epochs, no crashes. T_max=35 was close to actual (good alignment — best/final gap 0.05%).

## Results
- **Primary metric**: 94.52% (baseline: 95.25%, delta: -0.73%)
- **Observations**: The model is too large (9.7M params) for the 300s budget. Only 32 epochs completed vs 58 at k=4. Despite good T_max alignment and pre-activation architecture, the model didn't converge. This demonstrates a clear capacity-vs-epochs trade-off: k=4 at 58 epochs outperforms k=6 at 32 epochs.
- **Key Learning**: k=6 (9.7M params) is past the sweet spot for 300s budget. The capacity-vs-epochs trade-off favors k=4 (4.3M, 58 ep). Further improvements must come from dimensions other than raw width.

## Verification
- **Conditions**: FAILED (94.52% < 95.35%)
- **Verdict**: no-improvement

## Unexplored Avenues
- k=5 as a compromise (~6.8M params, ~45 epochs)
- Keep k=4 and improve training efficiency (stochastic depth, better augmentation, higher LR)
- Pre-activation at k=4 (isolate the architecture change from width)
- Different optimizer (AdamW, LAMB) for faster convergence

## Next Steps
1. **k=4 + pre-activation** (high confidence): Test if pre-activation helps at k=4 where epoch count is sufficient.
2. **k=4 + stochastic depth + Mixup** (medium confidence): Better regularization and faster training.

## Exit Action Results
