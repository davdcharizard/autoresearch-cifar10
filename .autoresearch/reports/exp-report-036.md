# Report EXP-036: channels_last + T_max=49 + LR clamp
- **Created**: 2026-06-04

## Results
- **Primary metric**: 95.96% (baseline: 96.39%, delta: -0.43%)
- 53 epochs with channels_last (vs 49 without). best==final. LR clamp never activated (53<54).
- **Key Learning**: channels_last gives 4 extra epochs on this slower system (53 vs 49), yielding +0.07% over T_max=43 baseline (95.89%).
- **Opportunity identified**: torch.cuda.synchronize() called every step adds significant overhead. Moving to epoch-level sync could give 10-20% more speedup.

## Verification
- **Verdict**: no-improvement
## Exit Action Results
