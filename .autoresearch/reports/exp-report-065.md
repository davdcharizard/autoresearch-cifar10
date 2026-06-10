# Report EXP-065: Lower CutMix Probability to 0.25
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-065.md
- **Plan**: plans/plan-065.md
- **Log**: logs/exp-log-065.md

## Goal

Maximize CIFAR-10 `best_test_acc (%)` under the fixed 300s training harness. The active baseline before this experiment was 94.11% at commit `1119ff8`, and the explicit noise guard required at least 94.21% to count as an improvement.

## Idea & Hypothesis

The chosen idea was a one-scalar CutMix probability bracket: lower `CUTMIX_PROB` from 0.5 to 0.25 while preserving the successful EXP-064 regional-mixing implementation. The hypothesis was that less frequent mixed-label regional training might keep the CutMix benefit while reducing target-noise pressure and improving the late peak.

## Approach

`train.py` was modified only to change `CUTMIX_PROB = 0.5` to `CUTMIX_PROB = 0.25`. `CUTMIX_ALPHA=1.0`, `CUTMIX_LABEL_SMOOTHING=0.05`, ResNet-20 `(28,56,112)`, `WEIGHT_DECAY=2e-4`, `LR_MILESTONES=[21000,64000]`, reflection padding, FP32 compile, channels-last, batch size, and validation cadence were unchanged.

## Execution

One local foreground run was launched on GPU1 with `env CUDA_VISIBLE_DEVICES=1 uv run train.py > run.log 2>&1`. Startup confirmed CUDA, ResNet-20 with 822,790 parameters, `CutMix alpha: 1.0, prob: 0.25, label smoothing: 0.05`, the 300s budget, and 390 batches per epoch. The first LR drop occurred at step 21000 in epoch 54. The run exited cleanly with final summary metrics and no error patterns.

## Results

- **Primary metric**: 94.09% (baseline: 94.11%, delta: -0.02 percentage points, -0.02%)
- **Observations**: Pre-drop best was 88.23%. Post-drop refinement reached 93.76% by epoch 65 and peaked at 94.09% at epoch 75, then stayed below that level through the final epoch. The run completed 40,685 steps and 105 epochs, more than EXP-064, so the miss was not due to lower coverage.
- **Analysis**: The hypothesis was not supported. Reducing CutMix frequency improved final accuracy relative to EXP-064 but failed to improve the primary best-checkpoint metric. This suggests the EXP-064 `p=0.5` setting was not simply too strong; the lower-frequency setting removed enough regional regularization to fall just under the new baseline.
- **Key Learning**: `CUTMIX_PROB=0.25` is a clean near-miss below the `p=0.5` CutMix anchor, so the next bracket should test stronger or area-distribution changes.

## Verification

- **Conditions**: all integrity checks passed; improvement threshold failed
- **Review Notes**: Results are trustworthy. The diff was limited to `train.py`, compile and ruff checks passed, final metrics were present, startup markers matched the plan, batch geometry was unchanged, the first LR drop was reached, and no error patterns were found.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid run, but `best_test_acc=94.09%` is below both the 94.11% baseline and the 94.21% threshold required by the +0.10 percentage-point noise guard.

## Unexplored Avenues

- Raise `CUTMIX_PROB` to 0.75. This tests the opposite probability bracket and may show whether the successful mechanism wants stronger regional exposure.
- Change `CUTMIX_ALPHA` to 0.5 while restoring `CUTMIX_PROB=0.5`. This keeps the successful frequency but alters patch-area distribution.
- Avoid changing label smoothing until CutMix probability and alpha are bracketed, because label-smoothing deviations are a repeated failed family.

## Next Steps

1. **High confidence**: Test `CUTMIX_PROB=0.75` with `CUTMIX_ALPHA=1.0`, because the lower bracket underperformed and the opposite direction remains the cleanest diagnostic.
2. **Medium confidence**: Test `CUTMIX_ALPHA=0.5` at `CUTMIX_PROB=0.5`, preserving the successful application frequency while changing patch areas.
3. **Low confidence**: Revisit unrelated augmentation only after the CutMix bracket is better mapped.

## Exit Action Results

- None; no exit actions are configured for this goal.
