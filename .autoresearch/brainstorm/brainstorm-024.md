# Brainstorm EXP-024
**Created**: 2026-05-29
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

- No new search needed.

## Experimental History Review

- **25 experiments**, baseline 96.39%, eight consecutive failures
- **Regularization map**: WD 5e-4 optimal (1e-4 worse, 1e-3 worse). CutMix p=0.5 optimal (p=0.3 worse). Label smoothing 0.1 — NEVER TUNED.
- **Key observation**: numpy random seed is unfixed — CutMix patterns vary across runs, creating ~0.3% variance. This means the 96.39% baseline might be a high-variance result. All recent experiments (96.01-96.28%) might be within normal variance of the same underlying accuracy.
- **Remaining lever**: Label smoothing 0.1 is the only regularization hyperparameter not yet tested at a different value.

## Candidate Ideas

### 1. Label Smoothing 0.05 + Fixed numpy Seed
**Summary**: Two changes: (1) reduce LABEL_SMOOTHING from 0.1 to 0.05, and (2) add `np.random.seed(42)` for deterministic CutMix. The label smoothing reduction lets the model predict true classes more confidently. The numpy seed ensures fair comparison (same CutMix patterns every run).

**Reasoning**: Label smoothing 0.1 means the true class target is 0.9, with 0.011 spread to each of 9 other classes. With CutMix already providing soft targets through mixed labels, the additional smoothing from label_smoothing=0.1 may be excessive — doubly softening the targets. Reducing to 0.05 (true class target = 0.95) gives the model more freedom to be confident on correct predictions while retaining mild regularization. The numpy seed eliminates run-to-run variance, making the comparison definitive.

**Sources**: EXP-023 (WD over-regularization confirmed existing recipe is well-tuned), label smoothing theory

**Estimated Effort**: low — 2 changes (1 constant + 1 seed line)

**Risk Assessment**: Low. Label smoothing 0.05 is still meaningful regularization. Adding np.random.seed changes the CutMix sequence, which could help or hurt independently of the label smoothing change. However, the determinism enables proper comparison in future experiments.

### 2. Label Smoothing 0.0 (No Smoothing)
**Summary**: Remove label smoothing entirely, keeping np.random.seed(42). Tests whether label smoothing provides any benefit at all given the existing CutMix + WD + EMA regularization.

**Reasoning**: If the model is already well-regularized, label smoothing might be entirely redundant. CutMix provides soft targets through mixing; EMA provides implicit regularization through weight averaging. Removing label smoothing could let the model learn harder classification boundaries.

**Sources**: EXP-011 (Dropout was redundant), EXP-021 (CutMix p=0.3 hurt)

**Estimated Effort**: low

**Risk Assessment**: Medium. No smoothing could cause overconfident predictions. But CutMix already provides soft targets.

## Idea Evaluation

Label smoothing 0.05 is the safer choice — it reduces regularization moderately rather than removing it entirely. Combined with the numpy seed fix, this is the most informative experiment we can run.

## Chosen Idea
**Selected**: Label Smoothing 0.05 + Fixed numpy Seed

**Why this idea**: The last untried regularization lever. Label smoothing 0.05 is conservative (not removed, just halved). The numpy seed fix makes the result definitive. If this doesn't improve over baseline, the model is truly at its ceiling.

**Hypothesis**: Reducing label smoothing from 0.1 to 0.05 will allow the model to make more confident predictions on correct classes, improving best_test_acc to ≥96.49%. The fixed numpy seed ensures reproducibility.
