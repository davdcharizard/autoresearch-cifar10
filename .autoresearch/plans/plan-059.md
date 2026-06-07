# Plan EXP-059
## Changes: EXP-054 config + CutMix cooldown after epoch 45
Same as EXP-054 (torch.seed(0)+np.seed(42)+BF16+CL+T_max=55+LR clamp) plus:
Change CutMix decision: `use_cutmix = np.random.rand() < CUTMIX_PROB and epoch <= 45`
