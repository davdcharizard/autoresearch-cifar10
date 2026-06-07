# Plan EXP-045: BFloat16 + T_max=43
## Code Changes
1. COSINE_T_MAX = 49 → 43
2. Change autocast to bfloat16: `torch.amp.autocast("cuda", dtype=torch.bfloat16)`
3. Remove GradScaler: delete `scaler = torch.amp.GradScaler()`
4. Replace `scaler.scale(loss).backward()` with `loss.backward()`
5. Replace `scaler.step(optimizer)` with `optimizer.step()`
6. Remove `scaler.update()`
7. Also update the compile warmup autocast to bfloat16
