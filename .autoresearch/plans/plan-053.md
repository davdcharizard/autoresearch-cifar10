# Plan EXP-053
## Changes
1. Replace `w = [16 * width_mult, 32 * width_mult, 64 * width_mult]` with `w = [64, 128, 320]`
2. COSINE_T_MAX = 44 (estimated for ~49 epochs with BF16+CL)
3. BF16 + channels_last + seed(42) + LR clamp
