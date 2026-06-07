# Brainstorm EXP-053
## Chosen: Asymmetric widths [64, 128, 320] + BF16+channels_last+T_max=44+seed(42)+LR clamp
Wider layer3 (320 vs 256) gives more classification features. Estimated ~49 epochs with BF16+CL. T_max=44 aligns to 49 epochs (5+44=49).
