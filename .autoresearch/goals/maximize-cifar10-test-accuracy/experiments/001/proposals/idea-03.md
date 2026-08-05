# Capacity-Matched Pre-Activation ResNet with Time-Aligned Cosine Decay

## Thesis

The baseline is simultaneously too small for an H20 and trained with a schedule that does not fit the actual run. It uses only 330 MiB, while the 300-second run ends at 38,254 steps: the first decay at step 32,000 occurs very late and the second decay at step 48,000 is never reached. Replace it with a modestly wider pre-activation ResNet-20, a larger batch, and a learning-rate schedule parameterized by elapsed training time. Add only mild Cutout as the generalization regularizer. This is one coherent test of whether spending the fixed compute budget on more useful capacity, then actually annealing that capacity to convergence, beats repeatedly updating an under-capacity model at a mostly constant learning rate.

## Concrete Change

- Convert each residual block to ResNet-v2 ordering: `BN -> ReLU -> Conv -> BN -> ReLU -> Conv`, with a final `BN -> ReLU` before global pooling. Keep three stages and three blocks per stage so depth remains ResNet-20. Preserve the cheap option-A strided/padded shortcut, applied to the pre-activated tensor when dimensions change.
- Increase stage widths only from `(16, 32, 64)` to `(24, 48, 96)`. This is a conservative 1.5x width increase, roughly 2.25x the convolutional parameters/FLOPs rather than a large WRN jump. It should remain far below the H20 memory limit while making each launch more compute-dense.
- Increase `BATCH_SIZE` from 128 to 256. Use `zero_grad(set_to_none=True)` and enable `torch.backends.cudnn.benchmark = True`. Do not introduce compilation or AMP in this first test; either could dominate the throughput result or add numerical/startup risk.
- Keep SGD with momentum 0.9. Use LR 0.2 for batch 256 (linear scaling from the baseline's 0.1), with a 5% linear warmup followed by cosine decay to 0.002 at 98% of the 300-second training budget, then hold at 0.002 for the tail. Set LR directly each step from `total_training_time / TIME_BUDGET_S`; do not use fixed step milestones or assume a step count.
- Raise weight decay modestly to `5e-4`, the conventional CIFAR residual-network regime. Keep crop and horizontal flip unchanged.
- Add one mild 8x8 Cutout square to every training image, but only while elapsed-time progress is below 70%. Apply it after images reach the GPU, using one vectorized mask per batch. Disable it for the final 30% so late optimization sees the original hard examples and the cosine tail can refine test accuracy.
- Keep the fixed seed, one evaluation at the end of each epoch, the unmodified evaluator, and the existing wall-clock accounting. Leave `MAX_STEPS` high enough that time, not the legacy 64,000-step cap, ends the run.

## Why This Combination

The three changes are coupled rather than an arbitrary bag of tricks. Width spends the otherwise idle H20 capacity on representation quality; batch 256 amortizes kernel and input-pipeline overhead while preserving approximately the baseline's number of image exposures even if optimizer-step throughput falls; elapsed-time cosine guarantees a complete optimization trajectory regardless of the new model's measured step rate. Mild early Cutout is the sole added augmentation and counterbalances the wider model's extra capacity without permanently depressing late hard-label convergence.

This is preferable to a large WRN or batch 512 in EXP-001. Those settings could cut the number of parameter updates and completed epochs sharply enough that a failed run would not distinguish insufficient optimization from an unhelpful architecture. It is also preferable to stacking mixup and EMA: mixup changes the loss target and may need its own alpha/timing calibration, while EMA changes which weights are evaluated. Both are good follow-ups after the capacity/schedule hypothesis is measured.

The architecture direction is supported by `.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/001/papers/wide-residual-networks.md`: wider residual networks improve CIFAR accuracy and can use compute more effectively than thin networks. The early-only regularization timing is supported by `.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/001/papers/time-matters-regularization.md`. The baseline metric and schedule mismatch come directly from `.autoresearch/goals/maximize-cifar10-test-accuracy/04-results.tsv` and the observed baseline summary (91.54%, 38,254 steps, 99 epochs, 330 MiB).

## Testable Prediction

The run should complete within the normal roughly 300-second training budget, use materially more than 330 MiB without approaching H20 capacity, execute the full warmup/cosine/tail schedule, and exceed 91.64% `best_test_acc`. A reasonable target is 92.3-93.2%, with at least 70 completed epochs; the accuracy threshold, rather than that target range, determines success.

## Instrumentation and Interpretation

Print model width, parameter count, batch size, schedule phase, progress-derived LR, epochs, steps, images processed, and peak VRAM in the existing logs. Record the final number of epochs and images processed alongside accuracy.

- **Accuracy improves and at least 70 epochs complete:** supports the combined capacity-matching hypothesis; next isolate additional regularization such as EMA or mixup.
- **Fewer than 70 epochs complete and accuracy regresses:** treat the model/batch combination as too compute-heavy, not as evidence against cosine or widening in general. The next experiment should retain the schedule and Cutout but use width 20 or batch 128.
- **At least 70 epochs complete but training accuracy/loss remains poor:** LR 0.2 or early Cutout is too aggressive. Retain width 24 and time cosine, reduce initial LR to 0.15 or Cutout to 4x4 in a follow-up.
- **Training converges but test accuracy remains near baseline:** the capacity/schedule bundle did not improve generalization; test EMA as a clean, isolated follow-up rather than adding more architecture changes.

## Risk Controls

- **Shortcut correctness:** pre-activation option-A shortcuts are easy to implement incorrectly. Add a local shape assertion during development for stage transitions, then remove or leave it inert before timing. Run a short import/forward/backward smoke test before the full experiment.
- **Throughput risk:** width is capped at 24 and batch at 256. Do not escalate width dynamically during the run. A fixed configuration keeps the result reproducible and interpretable.
- **Schedule boundary:** compute LR from the measured training-time accumulator, not total wall time, so excluded validation time cannot advance the schedule. Clamp progress to `[0, 1]`.
- **Cutout overhead:** create the mask with tensor operations on-device and avoid a Python loop over examples. If the implementation requires per-image Python work, omit Cutout rather than compromising throughput.
- **Scope:** modify only `train.py`; add no dependency, seed change, extra validation, evaluator change, or data leakage.

## Estimated Effort

Medium. The pre-activation block and time-derived scheduler are localized changes, but the transition shortcut and early Cutout need a smoke test. Only one full 300-second experiment is required.
