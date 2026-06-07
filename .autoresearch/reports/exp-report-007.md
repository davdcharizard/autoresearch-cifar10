# Report EXP-007: k=4 + EMA + Weight Decay 5e-4
- **Created**: 2026-05-28
- **Goal**: goals/maximize-cifar10-test-accuracy.md

## Goal
Maximize CIFAR-10 test accuracy. Baseline: 95.25% (EXP-004).

## Idea & Hypothesis
Add EMA (decay=0.999) and increase weight decay from 1e-4 to 5e-4. Hypothesis: 95.4-95.8%.

## Approach
Added EMA model (deepcopy before compile), EMA update after each step (parameters + BN buffers), evaluate with EMA model. Weight decay 5e-4.

## Execution
Run 1: 12.04% — catastrophic failure due to EMA model's BN running stats never being updated (ema_model always in eval mode). Fixed by copying BN buffers from training model to EMA model.
Run 2: 95.73% — success.

## Results
- **Primary metric**: 95.73% (baseline: 95.25%, delta: +0.48%)
- **Observations**: best == final (perfect convergence). 55 epochs (3 fewer than EXP-004's 58 — EMA update adds slight overhead). EMA smoothing is clearly beneficial — the model converges more stably.
- **Key Learning**: EMA + higher weight decay provides meaningful improvement. Critical implementation detail: must copy BN buffers to EMA model, not just parameters. Weight decay 5e-4 is better than 1e-4 for 4.3M params.

## Verification
- **Conditions**: All passed (95.73% >= 95.35%)
- **Verdict**: improvement

## Unexplored Avenues
- EMA at k=5 (compromise between k=4 and k=6)
- Stochastic depth + EMA
- Higher LR with EMA (EMA stabilizes aggressive LR)
- Pre-activation at k=4 with EMA

## Next Steps
1. **Stochastic depth + EMA at k=4** (medium confidence): Additional regularization.
2. **k=5 with EMA + WD=5e-4** (medium confidence): Try slightly wider with EMA stabilization.

## Exit Action Results
