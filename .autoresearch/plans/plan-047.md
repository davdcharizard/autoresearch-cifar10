# Plan EXP-047
## Code Changes
1. COSINE_T_MAX = 49 → 55
2. BF16 autocast + no GradScaler
3. channels_last on model + inputs
4. LR clamp after cosine
