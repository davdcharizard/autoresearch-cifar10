# EXP-011 Adversarial Plan Review

Offline local `plan-critic` fallback review, 2026-07-24.

## Prioritized Concerns

1. **High - Verification Protocol steps 5-6**: CV units are ambiguous. The plan defines CV as `statistics.pstdev/mean` but compares it to "5%." Specify the ratio threshold as `<=0.05` so an implementation cannot accidentally admit 500% variability.
2. **Medium - Verification Protocol step 3**: "all later blocks' 1-stride 128-to-128 shapes" is inaccurate across all stages. Enumerate the accepted later-block shapes as 32-to-32, 64-to-64, and 128-to-128, with the candidate's additional block explicitly 128-to-128.
3. **Low - Milestone 4 / Verification Protocol step 8**: `[300.0,300.1]` is brittle because the time-budget loop completes the final step before checking the budget. Permit a small documented final-step overshoot while retaining the hard 600-second wall cap.

## Disposition

All three concerns are valid execution-soundness improvements. The plan is refined to use CV ratio `<=0.05`, enumerate every stage topology, and accept reported counted time in `[300.0,300.5]`; larger overshoot is invalid because normal FP32 steps are far below 0.5 seconds.
