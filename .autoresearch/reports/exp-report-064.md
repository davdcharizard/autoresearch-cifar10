# Report EXP-064: Probabilistic CutMix Regional Mixing
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-064.md
- **Plan**: plans/plan-064.md
- **Log**: logs/exp-log-064.md

## Goal

Maximize CIFAR-10 `best_test_acc (%)` under the fixed 300s training harness. Before this experiment the active baseline was 93.97% at commit `755be2c`, and the explicit noise guard required at least 94.07% to count as an improvement.

## Idea & Hypothesis

The chosen idea was probabilistic CutMix regional mixing: apply CutMix to half of training batches with `alpha=1.0`, preserve full-run label smoothing at 0.05, and keep the model, optimizer, schedule, crop padding, compile path, and validation cadence unchanged. The hypothesis was that regional patch mixing would provide useful invariance while avoiding the information loss of Cutout and the global interpolation behavior that made direct mixup variants underperform.

## Approach

`train.py` was modified only to add `CUTMIX_ALPHA = 1.0`, `CUTMIX_PROB = 0.5`, `CUTMIX_LABEL_SMOOTHING = 0.05`, a clipped `rand_bbox` helper, startup logging for the CutMix settings, and a training-loop branch that samples CutMix after moving the batch to the selected GPU. CutMix batches clone inputs, paste a permuted rectangular patch, recompute lambda from the actual patch area, and use weighted endpoint cross entropy. Non-CutMix batches keep the original `label_smoothing=0.05` loss path.

No architecture, optimizer, LR schedule, batch size, dataset transform, dependency, or evaluation harness changes were made.

## Execution

One local foreground run was launched on GPU0 with `env CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`. Startup confirmed CUDA, ResNet-20 with 822,790 parameters, `CutMix alpha: 1.0, prob: 0.5, label smoothing: 0.05`, the 300s budget, and 390 batches per epoch. The run reached the first LR drop at step 21000 in epoch 54 and completed cleanly with final summary metrics.

There were no tracebacks, CUDA OOMs, runtime errors, or non-finite-loss markers.

## Results

- **Primary metric**: 94.11% (baseline: 93.97%, delta: +0.14 percentage points, +0.15%)
- **Observations**: Pre-drop best accuracy was 87.97%; post-drop refinement reached 93.89% by epoch 71, 93.93% by epoch 89, and crossed the improvement threshold at epoch 96. Final accuracy fell to 93.02%, so the useful signal is a peak-checkpoint improvement, not a final-checkpoint lift.
- **Analysis**: The hypothesis is supported. Regional CutMix produced the first valid improvement since EXP-038 while preserving the 21k LR drop and the 822,790-parameter anchor. This distinguishes regional patch replacement from the failed direct mixup and Cutout families.
- **Key Learning**: Probabilistic CutMix at `p=0.5`, `alpha=1.0` improves the tuned label-smoothed 2e-4 anchor enough to become the new baseline.

## Verification

- **Conditions**: all passed
- **Review Notes**: Results are trustworthy. The scoped diff was limited to `train.py`; compile and ruff checks passed; final metrics were present; model depth, CutMix settings, batch geometry, and LR-drop behavior matched the plan.
- **Verdict**: improvement
- **Verdict Basis**: All verification conditions passed and `best_test_acc=94.11%` exceeded both the 93.97% baseline and the 94.07% improvement threshold.

## Unexplored Avenues

- Bracket CutMix strength around the successful recipe. `CUTMIX_PROB=0.25` may preserve more clean-anchor batches, while `CUTMIX_PROB=0.75` may add stronger regional regularization.
- Bracket patch-area distribution. `CUTMIX_ALPHA=0.5` could bias toward more extreme patch sizes and may change the late peak behavior without introducing a new mechanism.
- Test whether CutMix should keep endpoint label smoothing at 0.05. This is riskier because label-smoothing deviations are a failed family, but it may matter specifically for mixed-label batches.

## Next Steps

1. **High confidence**: Run a local CutMix probability bracket, starting with `CUTMIX_PROB=0.25` or `0.75`, against the new 94.11% baseline and 94.21% threshold.
2. **Medium confidence**: Bracket `CUTMIX_ALPHA` while keeping `CUTMIX_PROB=0.5`, because the mechanism worked but the exact patch-size distribution is untested.
3. **Low confidence**: Revisit CIFAR AutoAugment only after CutMix brackets are exhausted; policy augmentation remains less distinct from the failed RandAugment probe.

## Exit Action Results

- Experiment index update: passed, EXP-064 inserted as `improvement`; baseline updated to 94.11 at commit `1119ff8`.
