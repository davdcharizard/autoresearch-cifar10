# Brainstorm EXP-052
## Chosen: BF16+channels_last+T_max=55+seed(42)+LR clamp + 5×5 first conv
Change self.conv1 kernel from 3×3 to 5×5 (padding 2). Captures wider spatial context in the stem. Only +3K params. Many CIFAR architectures benefit from wider first-layer receptive field.
