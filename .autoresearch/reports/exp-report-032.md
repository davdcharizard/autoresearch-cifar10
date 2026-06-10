# Report EXP-032: Mild Isolated Label Smoothing
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-032.md
- **Plan**: plans/plan-032.md
- **Log**: logs/exp-log-032.md

## Goal
Improve CIFAR-10 `best_test_acc` under the fixed harness while modifying only `train.py`. The active baseline before EXP-032 was 93.58% from EXP-029, and the goal requires at least +0.10 percentage points to count as an improvement, so the concrete EXP-032 threshold was 93.68%.

## Idea & Hypothesis
The chosen idea was to preserve the current reflection-padding 28/56/112 ResNet-20 anchor and add only mild label smoothing to the training loss. The hypothesis was that `label_smoothing=0.05` would reduce overconfident late updates enough to improve the peak accuracy without reducing step throughput or changing the schedule, architecture, augmentation, optimizer, seed, or evaluation cadence.

## Approach
EXP-032 changed the training loss call from `F.cross_entropy(outputs, targets)` to `F.cross_entropy(outputs, targets, label_smoothing=0.05)`. All other anchor settings were preserved: `STAGE_WIDTHS = (28, 56, 112)`, reflected `RandomCrop`, `BATCH_SIZE = 128`, `LR = 0.1`, `MOMENTUM = 0.9`, `WEIGHT_DECAY = 1e-4`, `LR_MILESTONES = [21000, 64000]`, FP32 channels-last training, cuDNN benchmark, `torch.compile`, fixed seed, SGD, and once-per-epoch validation. There were no deviations from the plan.

## Execution
One local single-GPU run was launched on GPU 0 and completed successfully. Startup confirmed CUDA execution, `ResNet-20 | params: 822,790`, `Time budget: 300s`, and `Batches per epoch: 390`. The planned first LR drop fired at step 21000 with `lr: 0.0100`; the second milestone at step 64000 was not reached. The run exited cleanly in 398.1 total seconds.

## Results
- **Primary metric**: 93.70% (baseline: 93.58%, delta: +0.12 percentage points, +0.13% relative)
- **Observations**: Best accuracy crossed the 93.68% improvement threshold at epoch 87 and finished close to the peak, with `final_test_acc = 93.65%`.
- **Analysis**: The result supports the hypothesis that mild confidence regularization helps the reflection-padding anchor. Unlike the earlier strong combined regularization bundle, this isolated mild setting preserved throughput and improved late stability without changing capacity or the LR schedule.
- **Key Learning**: Mild label smoothing is a useful no-throughput regularizer for the reflection-padding anchor and should remain in the baseline.

## Verification
- **Conditions**: all passed
- **Review Notes**: Results are trustworthy. The run completed without crash, produced numeric metrics, changed only `train.py`, preserved the fixed 300s training budget, stayed under the 10-minute wall-clock limit, preserved parameter count and batch count, hit the first LR drop at step 21000, avoided the unreachable second LR drop, and had no error/OOM/NaN/Inf signatures.
- **Verdict**: improvement
- **Verdict Basis**: All verification conditions passed, and `best_test_acc = 93.70%` exceeded the 93.68% threshold required by the goal's +0.10 percentage-point rule.

## Unexplored Avenues
- Try an even milder smoothing value such as 0.03 if future evidence suggests 0.05 improves calibration but slightly limits peak accuracy.
- Combine mild label smoothing with a carefully chosen low-overhead late-stability mechanism only if it preserves the 21k first drop and does not add validation overhead.

## Next Steps
Continue from the new 93.70% baseline with low-overhead changes that preserve the reflection-padding, label-smoothed anchor. Medium confidence: test a mild batch-size/stochasticity adjustment only if the plan verifies LR milestone reachability. Medium confidence: revisit short-window late averaging with strict throughput and validation-cadence safeguards.

## Exit Action Results
