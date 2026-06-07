# Report EXP-021: CutMix probability reduction (0.5 → 0.3)
- **Created**: 2026-05-29
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-021.md
- **Plan**: plans/plan-021.md
- **Log**: logs/exp-log-021.md

## Goal

Maximize CIFAR-10 test accuracy within 300s. Baseline: 96.39%.

## Idea & Hypothesis

Reduce CutMix probability from 0.5 to 0.3 to give the model more clean training batches, testing the hypothesis that the model is over-regularized and convergence-limited.

## Approach

Single change: CUTMIX_PROB 0.5 → 0.3. All other hyperparameters identical.

## Execution

Single clean run. 58 epochs in 300s.

## Results

- **Primary metric**: 96.06% (baseline: 96.39%, delta: -0.33%, -0.34%)
- **Analysis**: The over-regularization hypothesis was wrong for CutMix probability. CutMix at p=0.5 provides valuable regularization — reducing it loses information about mixed-class boundaries that helps generalization. EXP-011's lesson about redundant regularization (Dropout) was about ADDING new regularizers, not about reducing existing well-tuned ones. The current regularization balance (CutMix p=0.5 + label smoothing 0.1 + WD 5e-4 + EMA 0.999) appears to be well-optimized as a whole.
- **Key Learning**: CutMix p=0.3 loses 0.33% vs p=0.5; the current regularization balance is well-tuned, not over-regularized.

## Verification

- **Conditions**: best_test_acc >= 96.49% FAILED (actual: 96.06%)
- **Verdict**: no-improvement

## Unexplored Avenues

- CutMix alpha tuning (1.0 → 0.5 for U-shaped mixing distribution) — changes mixing intensity, not frequency
- Label smoothing tuning (0.1 → 0.05) — but the over-regularization hypothesis is weakened
- Gradient clipping — orthogonal to regularization, targets gradient stability

## Next Steps

1. **Gradient clipping** (medium confidence) — add max-norm gradient clipping. Orthogonal to regularization tuning. Stabilizes training against CutMix gradient spikes.
2. **Label smoothing 0.05** (low confidence) — the over-regularization hypothesis is weakened but label smoothing is a different mechanism than CutMix.
3. **Different random seed** (low confidence) — the baseline 96.39% may have run-to-run variance of ~0.2%. A different seed might get lucky.

## Exit Action Results
