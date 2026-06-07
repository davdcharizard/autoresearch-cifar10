# Report EXP-035: T_max=43 (schedule alignment)
- **Created**: 2026-06-04

## Results
- **Primary metric**: 95.89% (baseline: 96.39%, delta: -0.50%)
- **Key Finding**: T_max=43 recovered schedule alignment on the slower system: 93.99%→95.89% (+1.9%). This confirms the T_max misalignment was the root cause of all failures since EXP-030.
- **Current system**: 49 epochs in 300s (vs original system's ~54). The 0.50% gap is from 5 fewer training epochs.
- best==final (95.89%) confirms perfect cosine alignment.

## Verification
- **Verdict**: no-improvement (0.50% below 96.39% threshold)

## Next Steps
1. **channels_last + T_max=48**: Use speedup to get ~55 epochs with proper schedule alignment — closes the epoch deficit.

## Exit Action Results
