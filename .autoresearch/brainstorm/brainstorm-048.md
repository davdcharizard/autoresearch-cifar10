# Brainstorm EXP-048
## Chosen: BF16 + channels_last + T_max=55 + LR clamp + np.random.seed(42)
Same as EXP-047 but with deterministic numpy seed. 96.17% ± 0.3% variance → potential to hit 96.49%.
