# Brainstorm EXP-050
## Chosen: BF16 + channels_last + T_max=49 + LR clamp + seed(42)
T_max=55 (EXP-048/049) gave 96.31% consistently. The ORIGINAL baseline used T_max=49 (faster decay) and got 96.39%. Let's try T_max=49 with BF16+channels_last (60 ep) + LR clamp to prevent restart after epoch 54.
