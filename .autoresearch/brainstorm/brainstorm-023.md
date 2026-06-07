# Brainstorm EXP-023
**Created**: 2026-05-29
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

- No new search needed. The WD tuning direction is grounded in experimental history.

## Experimental History Review

- **24 experiments**, baseline 96.39%, seven consecutive failures (017-022)
- **WD history**: EXP-007 increased WD from 1e-4 to 5e-4, yielding +0.48% (the second-largest single-change gain after width scaling). This was the most successful hyperparameter change.
- **Variance observation**: Runs with identical training code show ~0.3% variance (EXP-020 got 96.13% with training-identical code to baseline 96.39%). numpy.random is unseeded — CutMix patterns differ across runs.
- **Recent failures**: Architecture (SE), TTA (spatial shifts), regularization tuning (CutMix prob), memory format (channels_last), gradient clipping — all failed. The recipe is near-optimal but WD is the one hyperparameter with demonstrated upward trajectory (1e-4 → 5e-4 helped; 5e-4 → 1e-3 untried).

## Candidate Ideas

### 1. Weight Decay 1e-3
**Summary**: Increase WEIGHT_DECAY from 5e-4 to 1e-3. This doubles the L2 regularization penalty on all parameters, encouraging smaller weights and potentially better generalization with 4.3M parameters.

**Reasoning**: WD 1e-4→5e-4 was the most successful hyperparameter change (+0.48% in EXP-007). The 4.3M param model may still have room for stronger regularization. WD 1e-3 is a common value in ResNet CIFAR-10 training (used in many WideResNet papers). The mechanism is clear: stronger L2 penalty → smaller weight magnitudes → smoother decision boundaries → better generalization. Combined with EMA (which smooths weights temporally), higher WD may further reduce parameter noise.

**Sources**: EXP-007 (+0.48% from WD 5e-4), WideResNet papers commonly use WD 5e-4 to 1e-3

**Estimated Effort**: low — single constant change

**Risk Assessment**: Low-medium. WD 1e-3 might over-regularize and slow convergence. But the model has shown robustness to moderate regularization (CutMix p=0.5 + label smoothing 0.1 + WD 5e-4 all help). Worst case: slight accuracy drop ~0.2-0.3%.

### 2. Fix numpy Random Seed
**Summary**: Add `np.random.seed(42)` at the start of main() to make CutMix patterns deterministic. This reduces run-to-run variance and ensures fair comparisons. Then re-run the exact baseline to establish the true seed-42 accuracy.

**Reasoning**: The current code only seeds torch, not numpy. CutMix uses np.random for beta sampling and random integers. This means every run produces different CutMix patterns, contributing ~0.3% variance. Adding the numpy seed makes training fully deterministic and ensures apples-to-apples comparisons.

**Sources**: Variance analysis from EXP-020 (identical training code gave 96.13% vs 96.39%)

**Estimated Effort**: low — add 1 line

**Risk Assessment**: Very low. This is a determinism fix, not a training change. But the numpy seed value might luck into a worse (or better) CutMix sequence. The accuracy will be DIFFERENT from 96.39% simply because the CutMix patterns change.

## Idea Evaluation

WD 1e-3 is the only remaining hyperparameter with demonstrated upward trajectory. The numpy seed fix is important for fair comparisons but won't improve accuracy. Let me combine both: add numpy seed for determinism AND increase WD to 1e-3. But that confounds two changes.

Better: WD 1e-3 alone. If the baseline already has ~0.3% variance, we might get lucky or unlucky, but the WD change is the real signal.

## Chosen Idea
**Selected**: Weight Decay 1e-3

**Why this idea**: WD is the only hyperparameter with a demonstrated positive trajectory (1e-4→5e-4 gave +0.48%). Further increase to 1e-3 is well-motivated for a 4.3M param model. All other avenues have been exhausted.

**Hypothesis**: Doubling weight decay from 5e-4 to 1e-3 will provide additional L2 regularization for the 4.3M param model, improving best_test_acc from 96.39% to ≥96.49%.
