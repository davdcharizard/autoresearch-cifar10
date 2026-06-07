# Brainstorm EXP-045
**Created**: 2026-06-04
## Chosen Idea: BFloat16 + T_max=43
Replace float16 + GradScaler with bfloat16 (no scaler). BF16 has full float32 exponent range, eliminating scaling overhead. May give faster per-step and more epochs.
