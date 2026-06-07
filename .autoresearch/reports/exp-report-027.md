# Report EXP-027: CutMix alpha 0.5
- **Created**: 2026-05-29
- **Goal**: goals/maximize-cifar10-test-accuracy.md

## Goal
Maximize CIFAR-10 test accuracy within 300s. Baseline: 96.39%.

## Idea & Hypothesis
CutMix alpha 0.5 (U-shaped mixing) for lighter average mixing intensity.

## Results
- **Primary metric**: 96.13% (baseline: 96.39%, delta: -0.26%)
- **Key Learning**: CutMix alpha 0.5 slightly worse than 1.0; all CutMix params (alpha=1.0, p=0.5) confirmed optimal.

## Verification
- **Verdict**: no-improvement

## Next Steps
1. Higher LR (0.15) — the only core hyperparameter never tested at a different value.

## Exit Action Results
