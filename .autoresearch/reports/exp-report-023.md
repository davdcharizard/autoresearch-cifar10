# Report EXP-023: Weight decay 1e-3
- **Created**: 2026-05-29
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-023.md
- **Plan**: plans/plan-023.md
- **Log**: logs/exp-log-023.md

## Goal
Maximize CIFAR-10 test accuracy within 300s. Baseline: 96.39%.

## Idea & Hypothesis
Increase WD from 5e-4 to 1e-3, extending the successful 1e-4→5e-4 improvement trajectory.

## Approach
Single change: WEIGHT_DECAY 5e-4 → 1e-3.

## Execution
Single clean run. 58 epochs.

## Results
- **Primary metric**: 96.01% (baseline: 96.39%, delta: -0.38%)
- **Analysis**: WD 1e-3 over-regularizes. The improvement from 1e-4→5e-4 does not continue linearly. WD 5e-4 is the sweet spot for this model size and training recipe. Higher WD suppresses weight magnitudes too aggressively, reducing model expressiveness.
- **Key Learning**: WD 5e-4 is the optimal weight decay; 1e-3 over-regularizes (-0.38%).

## Verification
- **Conditions**: best_test_acc >= 96.49% FAILED (96.01%)
- **Verdict**: no-improvement

## Unexplored Avenues
- WD 7e-4 or 8e-4 — finer grid between 5e-4 and 1e-3, but likely diminishing returns
- Label smoothing 0.05 — softer targets from loss function side
- CutMix alpha 0.5 — U-shaped mixing distribution

## Next Steps
1. **Label smoothing 0.05** (low-medium confidence) — the only remaining regularization lever not tried.
2. **CutMix alpha 0.5** (low confidence) — different mixing distribution, qualitatively distinct from prob change.
3. Accept 96.39% as near-ceiling for this architecture (realistic assessment).

## Exit Action Results
