# Plan EXP-033: Probability-averaged TTA
- **Created**: 2026-06-04
- **Goal**: goals/maximize-cifar10-test-accuracy.md

## Code Changes
- **train.py**: Change eval-mode forward() from logit averaging to probability averaging:
  ```python
  logits = self._features(x)
  logits_flip = self._features(x.flip(3))
  probs = (F.softmax(logits, dim=1) + F.softmax(logits_flip, dim=1)) / 2
  return torch.log(probs + 1e-8)
  ```
  Returns log-probabilities. The evaluator's `outputs.argmax(1)` works correctly since argmax(log(p)) == argmax(p). The cross_entropy loss reported will be slightly off (it applies softmax again on log-probs) but accuracy is what matters for verification.

## Verification
1. `grep "^best_test_acc:" run.log` — must be >= 96.49%
