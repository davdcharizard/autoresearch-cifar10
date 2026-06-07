# Plan EXP-051
## Code Changes: EXP-048 config + zero-init BN2 gamma
1. COSINE_T_MAX = 55
2. BF16 + channels_last + seed(42) + LR clamp (same as EXP-048)
3. Add zero-init: `for m in self.modules(): if isinstance(m, BasicBlock): nn.init.zeros_(m.bn2.weight)` after weights_init
