# Plan EXP-046: BF16 + T_max=50 + LR clamp
## Code Changes (same as EXP-045 but T_max=50 and LR clamp):
1. COSINE_T_MAX = 49 → 50
2. autocast dtype=torch.bfloat16
3. Remove GradScaler, use direct loss.backward() + optimizer.step()
4. LR clamp after epoch > WARMUP_EPOCHS + COSINE_T_MAX
