# Report EXP-028: Peak LR 0.15
- **Created**: 2026-05-29
- **Goal**: goals/maximize-cifar10-test-accuracy.md

## Results
- **Primary metric**: 95.98% (baseline: 96.39%, delta: -0.41%)
- **Key Learning**: LR 0.15 is too high; LR 0.1 is optimal for batch=128 with this architecture.

## Verification
- **Verdict**: no-improvement

## Next Steps
1. **Channels_last + LR clamp** — use channels_last for speedup, keep T_max=49, manually clamp LR after cosine finishes to prevent periodic restart. Addresses the root cause of EXP-018/019 failures while keeping the optimal decay rate.

## Exit Action Results
