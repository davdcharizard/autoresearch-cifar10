# EXP-048 Report: Lower BatchNorm Momentum to 0.05

## Summary

- **Verdict**: no-improvement
- **Primary Metric**: `best_test_acc`
- **Baseline**: 93.97% at commit `755be2c`
- **Improvement Threshold**: 94.07%
- **Result**: 93.48%
- **Delta vs Baseline**: -0.49 percentage points

## Goal

The active goal is to maximize CIFAR-10 `best_test_acc` under the fixed benchmark harness. Experiments must modify only `train.py`, use the fixed training budget and evaluation path, and count as an improvement only when `best_test_acc` clears the current baseline by at least +0.10 percentage points.

## Idea & Hypothesis

EXP-048 tested whether slower BatchNorm running-stat updates improve evaluation accuracy. The hypothesis was that setting all `nn.BatchNorm2d` layers to `momentum=0.05` would smooth noisy augmented mini-batch statistics used by `model.eval()` and lift the current 93.97% anchor to at least 94.07% without adding runtime overhead.

## Approach

The implementation added `BN_MOMENTUM = 0.05` in `train.py` and passed it to the three BatchNorm construction sites: the ResNet stem and both BatchNorm layers inside `BasicBlock`. The experiment preserved the current anchor: `STAGE_WIDTHS=(28, 56, 112)`, batch size 128, LR 0.1, momentum 0.9, weight decay 2e-4, LR milestones `[21000, 64000]`, reflection crop padding, label smoothing 0.05, FP32 compile, and channels-last.

## Execution

The run launched locally on GPU0 with output captured to `run.log`. Both GPUs were idle at launch, startup was clean, and no traceback/OOM patterns appeared. The first LR drop occurred at step 21000, so the comparison is not confounded by the missed-milestone issue seen in EXP-047.

Final run metrics:

- `best_test_acc`: 93.48%
- `final_test_acc`: 93.37%
- `final_test_loss`: 0.2270
- `training_seconds`: 300.0
- `total_seconds`: 377.9
- `startup_seconds`: 2.9
- `peak_vram_mb`: 660.4
- `num_epochs`: 61
- `num_steps`: 23449
- `num_params`: 822,790

## Results

Lowering BatchNorm momentum clearly underperformed the anchor. The run behaved normally and reached the first LR drop, then improved from 92.03% at epoch 54 to a peak of 93.48% at epoch 57 before ending at 93.37%. That trajectory shows the intervention did not crash or distort throughput, but it also did not improve generalization enough to approach the 94.07% threshold.

The result suggests the PyTorch default BatchNorm momentum of 0.1 is better calibrated for this fixed-budget recipe than a slower 0.05 running-stat update. A likely mechanism is that the lower momentum makes running statistics lag the rapidly changing feature distribution during the short post-drop refinement window. Because this was a clean run, EXP-048 provides stronger negative evidence than EXP-047 did for its augmentation idea.

## Verification

All process and integrity checks passed:

- Baseline check reported `baseline=93.97` and `baseline_commit=755be2c`.
- Tracked code diff was limited to `train.py`.
- `python3 -m py_compile train.py` exited 0.
- `uv run ruff check train.py` passed.
- `run.log` reported numeric final metrics.
- Total runtime was 377.9 seconds, below the 600-second cap.
- The first LR drop was observed at step 21000 with `lr: 0.0100`.

Verdict basis: valid no-improvement. `best_test_acc=93.48%` is below both the 93.97% baseline and the 94.07% improvement threshold.

## Key Learning

Lower BatchNorm momentum smoothed eval statistics but underperformed the anchor at 93.48%, so default BN momentum remains preferable.

## Unexplored Avenues

- **Higher BN momentum, low confidence**: A value like 0.2 could adapt running statistics faster, but it is the symmetric sibling of a failed isolated BN-stat tweak and risks adding noise.
- **BatchNorm affine or initialization tweaks, low confidence**: Partial residual BN scaling remains possible, but EXP-028 makes this family risky unless the chosen value avoids fixed-budget undertraining.
- **Optimizer dynamics, medium confidence**: Decoupled weight decay at 2e-4 remains a distinct mechanism from BN-stat smoothing and may test whether the successful shrinkage magnitude can be separated from SGD momentum-buffer coupling.

## Next Steps

1. **Test decoupled SGD weight decay at 2e-4 (medium confidence)**: Keep the validated regularization magnitude while changing decay semantics, with careful diff and throughput checks.
2. **Retry photometric augmentation only under clean GPU conditions (low-medium confidence)**: EXP-047 was confounded by missing the LR drop, but isolated augmentation should wait for a stronger policy rationale.
3. **Consider partial residual BN scale initialization (low confidence)**: A nonzero value such as 0.1 is less extreme than EXP-028 zero-gamma, but the family has negative prior evidence.
