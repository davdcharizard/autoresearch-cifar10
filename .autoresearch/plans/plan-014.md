# Plan EXP-014: k=4 + AdamW + EMA
## Code Changes
- Replace SGD optimizer with AdamW(lr=1e-3, weight_decay=0.05)
- Remove MOMENTUM, nesterov params
- Update LR and WEIGHT_DECAY constants
- Remove GradScaler (AdamW with AMP doesn't need loss scaling on modern PyTorch)

Actually, keep GradScaler — AMP still benefits from it. AdamW works fine with GradScaler.

## Verification
best_test_acc >= 95.83%
