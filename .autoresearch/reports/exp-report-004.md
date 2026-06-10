# Report EXP-004: EMA Evaluation Weights
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-004.md
- **Plan**: plans/plan-004.md
- **Log**: logs/exp-log-004.md

## Goal

Maximize CIFAR-10 `best_test_acc` in the higher-is-better direction while modifying only `train.py` and preserving the fixed `prepare.py` evaluation harness. The current baseline entering this experiment was EXP-002 at 91.95%, and the updated goal requires at least +0.10 percentage points over baseline, so EXP-004 needed `best_test_acc >= 92.05%`.

## Idea & Hypothesis

The chosen idea was to maintain exponential moving average weights for the successful EXP-002 FP32 throughput ResNet-20 and evaluate the EMA model once per epoch. The hypothesis was that averaged weights and buffers would smooth late SGD noise enough to lift the run above the tightened 92.05% threshold without changing architecture, optimizer, data augmentation, precision, or the evaluation harness.

## Approach

`train.py` added PyTorch's `AveragedModel` and `get_ema_multi_avg_fn`, with `USE_EMA=True` and `EMA_DECAY=0.999`. The implementation keeps a real `base_model` for optimizer updates and EMA source parameters, compiles only the training forward path, updates EMA after each optimizer step, and evaluates the EMA model once per epoch. BatchNorm buffers are included in the EMA copy through `use_buffers=True`.

## Execution

One local single-GPU run completed normally under `CUDA_VISIBLE_DEVICES=0`. Startup was clean, CUDA was active, EMA was enabled, and no tracebacks, OOMs, or compiler failures occurred. The run used the full 300.0 training seconds and 387.9 total seconds, completing 94 epochs and 36,574 optimizer steps.

## Results

- **Primary metric**: 91.98 (baseline: 91.95, delta: +0.03, +0.03%)
- **Observations**: EMA update overhead reduced throughput relative to EXP-002, which had reached 43,398 steps. EXP-004 reached the first LR drop only near 87.4% of training time, peaked at 91.98% on epoch 80 before the drop, and did not improve during the short LR 0.01 tail.
- **Analysis**: The hypothesis was only weakly supported. EMA evaluation slightly exceeded the current baseline, but the effect was far below the required +0.10 percentage point margin and likely not large enough to distinguish from noise. The overhead cost is material: losing about 6,800 optimizer steps delayed the schedule and may have offset any smoothing benefit.
- **Key Learning**: Per-step EMA is too expensive for this fixed-budget ResNet-20 recipe; it produced only a +0.03 point gain and missed the +0.10 threshold.

## Verification

- **Conditions**: primary metric condition failed.
- **Review Notes**: Results are trustworthy. The run completed, reported numeric metrics, used one GPU, changed only `train.py`, preserved the evaluation harness, and evaluated once per epoch. The failure is a valid no-improvement under the updated noise-margin rule.
- **Verdict**: no-improvement
- **Verdict Basis**: `best_test_acc=91.98%` is below the required `92.05%` threshold, even though it is slightly above the 91.95% baseline.

## Unexplored Avenues

- Lower-overhead EMA could update every N steps or only after the first LR drop, preserving more of EXP-002's optimizer-step budget while still smoothing late weights.
- Mild cutout-only remains untested on the EXP-002 FP32 throughput baseline and may provide a larger generalization gain than EMA if kept separate from label smoothing and slow cosine.
- Compact architecture changes such as WRN-style widening or deeper ResNet variants still offer a higher ceiling, but need schedule/runtime calibration under the fixed budget.

## Next Steps

- High confidence: avoid per-step EMA in its current form because it delays the LR schedule and lands below the +0.10 threshold.
- Medium confidence: try mild cutout-only on the FP32 throughput baseline as the next low-risk generalization experiment.
- Medium confidence: test a lower-overhead EMA variant only if the goal is to isolate whether averaging helps apart from the step-count loss.

## Exit Action Results
