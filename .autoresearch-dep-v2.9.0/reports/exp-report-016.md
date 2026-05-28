# Report EXP-016: Higher BN Momentum (0.5)
- **Created**: 2026-05-27
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-016.md
- **Plan**: plans/plan-016.md
- **Log**: logs/exp-log-016.md

## Goal

Maximize CIFAR-10 test accuracy (best_test_acc, %). Higher is better. Baseline: 95.57% (EXP-015, commit 626e9d1). Improvement threshold: baseline + 0.1pp = 95.67%.

## Idea & Hypothesis

Increase BatchNorm momentum from PyTorch default 0.1 to 0.5 on all BN layers. Motivated by cifar10-airbench speedrun recipe (uses 0.6) and the "94% in 3.29s" paper, which identify BN momentum tuning as a key ingredient for short-budget training. Hypothesis: faster BN running stat convergence would reduce train-eval distribution mismatch, improving test accuracy by 0.1–0.3pp.

## Approach

Added a 3-line loop in train.py (after model creation, before num_params calculation) that iterates `model.modules()`, checks `isinstance(m, nn.BatchNorm2d)`, and sets `m.momentum = 0.5`. This applies to all 13 BN layers in the width-4x ResNet-20. No other code or hyperparameters changed. No deviations from the plan.

## Execution

Single local run completed normally. 98 epochs, 19007 steps in 300.0s. Throughput ~16,300 img/s throughout — zero overhead from the BN momentum change, as expected. Test accuracy oscillated more during warmup (81–86% at epochs 20–35) compared to typical baseline, which may reflect the noisier running stats from higher momentum. Best accuracy 95.59% achieved at the final epoch.

## Results

- **Primary metric**: 95.59% (baseline: 95.57%, delta: +0.02pp, +0.02%)
- **Observations**: The +0.02pp gain is within noise — stochastic training variance on this setup is at least ±0.1pp between runs. The increased warmup oscillation is consistent with noisier running stats, but it self-corrected by mid-training. The final accuracy curve was nearly identical to baseline.
- **Analysis**: The hypothesis that BN running stat lag is a meaningful source of train-eval mismatch at 98 epochs appears incorrect. With ~194 batches per epoch × 98 epochs ≈ 19K updates, even the default momentum (0.1) produces well-converged running stats by the end of training. The speedrun recipes that benefit from high BN momentum train for far fewer epochs (~10-27) where the stat convergence issue is more acute. At 98 epochs, 0.1 momentum gives an effective averaging window that already captures the relevant statistics.
- **Key Learning**: BN momentum tuning is a short-training optimization (sub-30 epochs); at ~100 epochs the default stats are already well-converged and higher momentum adds only noise.

## Verification

- **Conditions**: Condition 1 FAILED (95.59% < 95.67%); Conditions 2 and 3 PASSED
- **Review Notes**: Results confirmed trustworthy — full summary block present, epoch count matches baseline, throughput unchanged. The +0.02pp is genuine but negligible.
- **Verdict**: no-improvement
- **Verdict Basis**: Primary metric did not exceed baseline + 0.1pp threshold (condition 1 failure)

## Unexplored Avenues

- **BN momentum scheduling** (e.g., high momentum early → low momentum late): Could capture the fast-convergence benefit during early training while reducing noise during the critical final low-LR phase. However, given the negligible effect at 98 epochs, even scheduled momentum is unlikely to yield a meaningful gain.
- **Momentum 0.6 or higher**: The speedrun uses 0.6, but our 0.5 result suggests the mechanism doesn't apply at our epoch count, making a higher value unlikely to help.

## Next Steps

1. **Mixup α=0.2 replacing RandomErasing** (medium confidence): Well-supported by literature, replaces rather than stacks augmentation. Cross-sample regularization provides a qualitatively different signal than per-sample augmentation. The CutMix failure (EXP-010) was at aggressive α=1.0 while stacking — α=0.2 with replacement is a safer approach.
2. **OneCycleLR triangular schedule** (low confidence): Speedrun-validated but contradicts our high-importance MultiStepLR pattern. Would need careful parameter selection.
3. **Gradient clipping** (low confidence): Could stabilize AMP training and reduce the oscillation during high-LR phases, potentially allowing a slightly higher peak LR.

## Exit Action Results
