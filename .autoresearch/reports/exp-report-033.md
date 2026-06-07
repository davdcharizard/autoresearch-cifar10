# Report EXP-033: Probability-averaged TTA
- **Created**: 2026-06-04

## Results
- **Primary metric**: 94.32% (baseline: 96.39%, delta: -2.07%)
- **Key Learning**: Probability-averaged TTA is much worse than logit-averaged TTA. Logit averaging preserves the model's raw confidence, which is better for argmax-based classification.
- **Also noted**: 48 epochs (vs baseline ~54) — likely caused by data/ directory being removed between runs, forcing CIFAR-10 re-download and cold data cache.

## Verification
- **Verdict**: no-improvement

## Exit Action Results
