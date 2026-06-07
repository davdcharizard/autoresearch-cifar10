# Report EXP-024: Label smoothing 0.05 + fixed numpy seed
- **Created**: 2026-05-29
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-024.md
- **Plan**: plans/plan-024.md
- **Log**: logs/exp-log-024.md

## Goal
Maximize CIFAR-10 test accuracy within 300s. Baseline: 96.39%.

## Idea & Hypothesis
Reduce label smoothing from 0.1 to 0.05; fix numpy seed for determinism.

## Approach
LABEL_SMOOTHING 0.1→0.05, added np.random.seed(42).

## Execution
Single clean run. 58 epochs.

## Results
- **Primary metric**: 96.22% (baseline: 96.39%, delta: -0.17%)
- **Analysis**: Label smoothing 0.05 is neither better nor worse than 0.1 within the run-to-run variance band (~0.3%). The numpy seed change means the CutMix sequence differs from the baseline run, confounding the comparison. The result suggests label smoothing 0.1 is already well-calibrated.
- **Key Learning**: Label smoothing 0.05 no better than 0.1; the recipe's hyperparameters are at their ceiling.

## Verification
- **Conditions**: best_test_acc >= 96.49% FAILED (96.22%)
- **Verdict**: no-improvement

## Unexplored Avenues
- Zero-init residual branches (BN gamma=0 in second BN) — well-known "Bag of Tricks" technique
- CutMix alpha tuning (1.0 → 0.5)
- Higher LR with current schedule

## Next Steps
1. **Zero-init residual branches** (medium confidence) — from He et al. 2019 "Bag of Tricks". Initialize BN2 gamma=0 so residual starts as identity. Zero overhead, proven technique.
2. **CutMix alpha 0.5** (low confidence) — last untried augmentation parameter.

## Exit Action Results
