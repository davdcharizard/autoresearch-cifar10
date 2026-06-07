# Plan EXP-044: channels_last + T_max=48 + LR clamp
## Code Changes
1. COSINE_T_MAX = 49 → 48
2. model.to(memory_format=torch.channels_last) before EMA deepcopy
3. inputs.to(device, memory_format=torch.channels_last, non_blocking=True)
4. LR clamp after epoch > WARMUP_EPOCHS + COSINE_T_MAX
