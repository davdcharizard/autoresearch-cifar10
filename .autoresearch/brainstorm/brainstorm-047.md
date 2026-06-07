# Brainstorm EXP-047
## Chosen: BF16 + channels_last + T_max=55 + LR clamp
BF16 alone: 55 epochs. channels_last: +10% speedup → ~60 epochs. T_max=55: 5+55=60 aligned.
