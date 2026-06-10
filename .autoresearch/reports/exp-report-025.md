# Report EXP-025: Batch Size 96 with Step-Budget-Aware Milestones
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-025.md
- **Plan**: plans/plan-025.md
- **Log**: logs/exp-log-025.md

## Goal
EXP-025 targeted higher CIFAR-10 `best_test_acc` under the fixed harness and fixed 300s training budget. The current experiment-index baseline was 93.23% from commit `f187edf`, so the goal's +0.10 percentage-point rule required at least 93.33% to count as an improvement.

## Idea & Hypothesis
The chosen idea was to reduce the batch size from 128 to 96 on the validated 28/56/112 anchor and recalibrate LR milestones to `[26000, 44000]`. The hypothesis was that smaller-batch stochasticity and a higher optimizer-step budget would improve generalization while keeping both LR drops meaningful under the wall-clock budget.

## Approach
`train.py` changed only two top-level constants: `BATCH_SIZE` from `128` to `96`, and `LR_MILESTONES` from `[21000, 64000]` to `[26000, 44000]`. Architecture, `1e-4` weight decay, optimizer class, augmentation, FP32 compile/channels-last path, fixed training budget, and once-per-epoch validation were preserved.

## Execution
One local single-GPU run was launched on GPU 0 with stdout/stderr captured to `run.log`. Startup was clean, CUDA saw one NVIDIA H20, the expected 822,790-parameter model was used, and `Batches per epoch: 520` confirmed the batch-size change. The first LR drop fired at step 26000 with `lr=0.0100`, but the run completed at 32,996 steps and never reached the planned step-44000 second drop. No traceback, OOM, NaN, or Inf patterns appeared.

## Results
- **Primary metric**: 93.11% (baseline: 93.23%, delta: -0.12 points, -0.13%)
- **Observations**: The smaller batch reduced images-per-step and completed only 64 epochs despite more batches per epoch. Accuracy jumped after the first LR drop, peaking at 93.11% by epoch 60, then finished at 93.06% with final loss 0.2605.
- **Analysis**: The run rejected the schedule-calibration premise. Batch size 96 did not create enough reachable optimizer steps for the planned second drop; instead it lowered throughput enough that the experiment had less useful schedule coverage than intended. The best accuracy remained below both the 93.23% baseline and the 93.33% improvement threshold.
- **Key Learning**: Batch size 96 slowed the run to 32,996 steps, missed the planned second drop, and peaked below the 93.33% threshold.

## Verification
- **Conditions**: Process and hard-constraint checks passed; the batch-size effect was confirmed, but the planned second LR drop was unreachable and the metric improvement condition failed.
- **Review Notes**: Results are trustworthy. The run completed successfully, reported numeric metrics, modified only `train.py`, preserved once-per-epoch validation, hit the first planned LR drop, and finished in 364.1 total seconds.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid result, but `best_test_acc=93.11%` is below the 93.23% baseline and the required 93.33% improvement threshold; additionally, the schedule plan failed because step 44000 was unreachable.

## Unexplored Avenues
- Batch size 112 could be a less aggressive noise-scale change that preserves more throughput, but the expected effect may be too small to clear the +0.10 threshold.
- Batch size 96 with a much earlier second drop could make the second LR phase reachable, but this result suggests the throughput cost makes smaller batches a weak isolated lever.
- Momentum coefficient tuning remains a cleaner optimizer-side lever because it does not reduce the step budget or validation cadence.

## Next Steps
Medium confidence: test `MOMENTUM = 0.95` on the current 28/56/112, 21k first-drop anchor, because it changes optimizer dynamics without reducing throughput.

Low confidence: try isolated mild label smoothing only if optimizer-side changes are exhausted, since explicit regularization has repeatedly underperformed.

Low confidence: revisit batch size only with a less aggressive value such as 112 and measured schedule milestones, not with 96.
