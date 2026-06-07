# Report EXP-026: Width k=5 with calibrated T_max
- **Created**: 2026-05-29
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-026.md
- **Plan**: plans/plan-026.md
- **Log**: logs/exp-log-026.md

## Goal
Maximize CIFAR-10 test accuracy within 300s. Baseline: 96.39%.

## Idea & Hypothesis
k=5 width (6.7M params) with dynamic T_max calibration from epoch 2.

## Approach
WIDTH_MULT 4→5, dynamic T_max calibration measuring total_training_time/2 after epoch 2.

## Execution
Single run. Catastrophic: T_max calibrated to 12, but 31 actual epochs.

## Results
- **Primary metric**: 89.94% (baseline: 96.39%, delta: -6.45%)
- **Analysis**: Two compounding failures: (1) T_max calibration was broken — `total_training_time/2` after 2 epochs averaged ~17.2s/epoch (inflated by torch.compile overhead in epoch 1), estimating only 17 total epochs when actual was 31. This set T_max=12, causing the LR to crash to zero in 12 cosine epochs then restart. (2) Even with correct T_max, 31 epochs is insufficient for 6.7M params — k=6 with 32 epochs also failed. k=4 with ~54 epochs remains the capacity sweet spot.
- **Key Learning**: k=5 only gets 31 epochs (too few for 6.7M params); dynamic T_max calibration from early epochs is unreliable.

## Verification
- **Conditions**: best_test_acc >= 96.49% FAILED (89.94%)
- **Verdict**: no-improvement

## Unexplored Avenues
- None for width >4 — both k=5 and k=6 are now confirmed too slow.

## Next Steps
- The model is at its ceiling. 11 consecutive failures confirm 96.39% is near-optimal for this ResNet-20 architecture with 300s training.

## Exit Action Results
