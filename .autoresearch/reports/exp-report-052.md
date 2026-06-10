# Report EXP-052: Hybrid Post-Drop Cosine LR Tail
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-052.md
- **Plan**: plans/plan-052.md
- **Log**: logs/exp-log-052.md

## Goal
Improve CIFAR-10 `best_test_acc` under the fixed harness, where higher is better. The active baseline before EXP-052 was 93.97% at commit `755be2c`, and the goal requires at least +0.10 percentage points to count as an improvement, so this experiment needed `best_test_acc >= 94.07%`.

## Idea & Hypothesis
EXP-052 tested a hybrid post-drop cosine LR tail. The selected idea preserved the validated 21k-step high-LR phase and first drop, then replaced the flat LR 0.01 tail with a smooth cosine decay toward a nonzero 0.002 floor. The hypothesis was that a gentler late LR reduction could improve post-drop refinement without recreating the failed abrupt second-drop behavior or the failed full-budget cosine schedule.

## Approach
`train.py` was the only tracked code file modified. The patch removed `MultiStepLR`, added explicit tail constants, added `lr_after_step` and `set_optimizer_lr`, and updated the optimizer LR after each completed optimizer step so `step 21000` logs `lr: 0.0100`. Model width, batch size, optimizer family, momentum, weight decay, reflection padding, label smoothing, compile, channels-last, and once-per-epoch validation were left unchanged.

## Execution
One local foreground run executed on GPU0 with output captured to `run.log`. Startup was clean, the run reached the planned first LR drop, and the cosine tail behaved as intended: LR was 0.0093 near step 25000, 0.0069 at step 30000, 0.0040 at step 35000, 0.0022 at step 40000, and 0.0020 near the end. The process exited 0 with numeric final metrics and no traceback, OOM, nan, or inf patterns.

## Results
- **Primary metric**: 93.87% (baseline: 93.97%, delta: -0.10pp, -0.11%)
- **Observations**: Post-drop convergence quickly reached 93.61% by epoch 61 and peaked at 93.87% on epoch 85, but later epochs stayed in a narrow 93.4-93.6% band.
- **Analysis**: The hypothesis was not supported. The smooth nonzero tail preserved schedule integrity and avoided a crash, but lowering LR below 0.01 late in training did not outperform the flat-tail anchor. This narrows the schedule-only search space: the problem is not simply late-update noise from the LR 0.01 plateau.
- **Key Learning**: A smooth 0.01-to-0.002 tail preserved the first drop and reached 93.87%, but did not beat the flat-tail anchor.

## Verification
- **Conditions**: all passed.
- **Review Notes**: Results are trustworthy. The tracked diff was limited to `train.py`, compile and ruff passed, the first LR drop logged `lr: 0.0100`, tail LR values stayed at or above the 0.002 floor, `num_params` remained 822,790, and final summary metrics were present.
- **Verdict**: no-improvement.
- **Verdict Basis**: The run was valid, but `best_test_acc=93.87%` is below both the 93.97% baseline and the 94.07% improvement threshold.

## Unexplored Avenues
- Couple a late schedule tail with an independently motivated non-schedule change. EXP-052 only tests the schedule shape, so a regularizer or optimizer change might still need a different tail.
- Test a higher nonzero floor such as 0.005 only if another signal suggests flat LR 0.01 is too high. EXP-052 makes isolated floor tuning low priority.

## Next Steps
- Medium confidence: test a very mild residual drop-path regularizer, since it targets co-adaptation through a different mechanism than schedule-only tuning.
- Medium confidence: test a modest larger batch such as 160 with careful step-budget monitoring to probe image-throughput versus update-count tradeoffs.
- Low confidence: revisit schedule tails only as part of a coupled experiment; isolated cosine variants are now lower priority.

## Exit Action Results
