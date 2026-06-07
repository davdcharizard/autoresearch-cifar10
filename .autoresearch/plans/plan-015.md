# Plan EXP-015: Mixup + EMA 0.9995
## Code Changes
- Replace CutMix with Mixup (alpha=0.2, p=0.5): blend full images + labels
- EMA_DECAY: 0.999 → 0.9995
## Verification
best_test_acc >= 95.83%
