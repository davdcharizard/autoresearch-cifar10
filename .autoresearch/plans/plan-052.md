# Plan EXP-052
## Changes: EXP-048 config + 5×5 first conv
All BF16+channels_last+T_max=55+seed(42)+LR clamp changes PLUS:
Change `self.conv1 = nn.Conv2d(3, w[0], 3, stride=1, padding=1, bias=False)` to
`self.conv1 = nn.Conv2d(3, w[0], 5, stride=1, padding=2, bias=False)`
