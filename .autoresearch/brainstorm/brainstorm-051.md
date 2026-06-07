# Brainstorm EXP-051
## Chosen: BF16 + channels_last + T_max=55 + seed(42) + LR clamp + zero-init residual
Adding zero-init BN2 gamma=0 to the best config. Zero-init was tested at 49 ep (broken T_max) and showed 96.08%. At 60 properly-aligned epochs, it may give +0.1-0.2% improvement.
