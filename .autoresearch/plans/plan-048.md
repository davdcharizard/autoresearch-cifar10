# Plan EXP-048
## Code Changes (same as EXP-047 plus seed):
1. COSINE_T_MAX = 55
2. BF16 autocast + no GradScaler
3. channels_last on model + inputs
4. LR clamp after cosine
5. np.random.seed(42)
