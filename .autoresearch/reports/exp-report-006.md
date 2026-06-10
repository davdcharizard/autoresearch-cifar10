# Report EXP-006: Schedule-Calibrated ResNet-32
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-006.md
- **Plan**: plans/plan-006.md
- **Log**: logs/exp-log-006.md

## Goal
Improve CIFAR-10 `best_test_acc` under the fixed `prepare.py` evaluation harness and 300 second training budget. The current baseline remains EXP-002 at 91.95%, and the tightened goal requires at least 92.05% to count as an improvement.

## Idea & Hypothesis
EXP-006 tested whether a modest capacity increase from ResNet-20 to ResNet-32 could move beyond the apparent recipe ceiling of the current baseline. The hypothesis was that calibrated LR milestones would let the larger model get enough low-LR refinement to reach at least 92.05%.

## Approach
`train.py` changed `NUM_BLOCKS` from 3 to 5, which reuses the existing CIFAR ResNet implementation and instantiates ResNet-32. It also added `LR_MILESTONES = [26000, 39000]` and wired the scheduler to that constant. Optimizer, batch size, augmentation, seed, FP32 compile, channels-last, and once-per-epoch evaluation were preserved.

## Execution
One local single-GPU run completed cleanly on an NVIDIA H20. Startup reported ResNet-32 with 464,154 parameters. The known non-fatal TF32 warning appeared again. No traceback, OOM, compile failure, or timeout occurred.

## Results
- **Primary metric**: 88.18% (baseline: 91.95%, delta: -3.77 points, -4.10%)
- **Observations**: The model only reached 23,642 optimizer steps in 300 training seconds, below the first planned LR drop at 26,000. Accuracy peaked at 88.18% while still at LR 0.1, then final accuracy fell to 84.61%.
- **Analysis**: The hypothesis failed because the schedule was not actually reached. The larger model's per-step cost made `[26000, 39000]` too late, so EXP-006 mostly measured high-LR ResNet-32 undertraining rather than low-LR refined capacity. This does not fully discredit capacity increases, but it rules out this depth/schedule combination.
- **Key Learning**: ResNet-32 is too slow for a 26k first LR drop under the fixed budget and badly undertrains at LR 0.1.

## Verification
- **Conditions**: primary metric condition failed.
- **Review Notes**: Results are trustworthy: the run completed, reported numeric metrics, and the failure is consistent with the logged step budget and unreached LR milestone.
- **Verdict**: no-improvement
- **Verdict Basis**: `best_test_acc=88.18%` is below the 91.95% baseline and the required 92.05% threshold.

## Unexplored Avenues
- A ResNet-32 retry would need a much earlier first LR drop, likely around 14k-16k steps, but this is no longer a clean capacity-only test.
- A width/depth variant should only be tried if paired with a pre-run throughput estimate or a schedule based on a measured short warmup.
- TF32 throughput enablement remains a narrow way to increase steps before revisiting larger models.

## Next Steps
- High confidence: test TF32 enablement on the existing ResNet-20 FP32 throughput baseline to improve step budget without changing capacity.
- Medium confidence: use a short measured warmup inside the script to set milestones as fractions of the observed step budget, if this can be done without violating the fixed evaluation harness.
- Low confidence: retry capacity with a smaller architecture change or much earlier schedule, because EXP-006 shows depth increases are easy to undertrain.

## Exit Action Results
