# Brainstorm EXP-033
**Created**: 2026-06-04
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Experimental History Review
- 34 experiments, 96.39% baseline, 17 consecutive failures
- TTA method: currently averaging RAW LOGITS. Never tried averaging PROBABILITIES (softmax first)
- The model is well-calibrated (label smoothing + EMA), so probability averaging should work well
- This is purely an eval-time change — zero training impact

## Chosen Idea
**Selected**: Probability-averaged TTA

**Summary**: Change TTA from logit averaging `(logits + logits_flip) / 2` to probability averaging `(softmax(logits) + softmax(logits_flip)) / 2`. Return log-probabilities so argmax still works correctly with the evaluator.

**Hypothesis**: Probability averaging handles confidence disagreements between views better than logit averaging for calibrated models, potentially flipping a few borderline predictions correctly and improving accuracy by ~0.1-0.3%.
