# Report EXP-066: CutMix Probability 0.75
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-066.md
- **Plan**: plans/plan-066.md
- **Log**: logs/exp-log-066.md

## Goal

Maximize CIFAR-10 `best_test_acc (%)` under the fixed 300s training harness. The active baseline before this experiment was 94.11% at commit `1119ff8`, and the explicit noise guard required at least 94.21% to count as an improvement.

## Idea & Hypothesis

The chosen idea was a high-frequency CutMix probability bracket: raise `CUTMIX_PROB` from 0.5 to 0.75 while preserving `CUTMIX_ALPHA=1.0`, `CUTMIX_LABEL_SMOOTHING=0.05`, and the rest of the successful EXP-064 recipe. The hypothesis was that if the EXP-064 anchor was still slightly under-regularized by regional mixing, more frequent CutMix exposure could raise the late best checkpoint above 94.21%.

## Approach

`train.py` was modified only to change `CUTMIX_PROB = 0.5` to `CUTMIX_PROB = 0.75`. The existing CutMix helper, label-mixing loss path, ResNet-20 `(28,56,112)` architecture, `WEIGHT_DECAY=2e-4`, `LR_MILESTONES=[21000,64000]`, reflection padding, FP32 compile path, channels-last memory format, batch size, and validation cadence were unchanged.

No implementation deviations were needed; the experiment was a one-line hyperparameter bracket.

## Execution

One local foreground run was launched on GPU0 with `env CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`. Startup confirmed CUDA, ResNet-20 with 822,790 parameters, `CutMix alpha: 1.0, prob: 0.75, label smoothing: 0.05`, the 300s budget, and 390 batches per epoch. The run reached the first LR drop at step 21000 in epoch 54 and exited cleanly with final summary metrics.

There were no tracebacks, CUDA OOMs, runtime errors, or non-finite-loss markers.

## Results

- **Primary metric**: 94.11% (baseline: 94.11%, delta: +0.00 percentage points, +0.00%)
- **Observations**: Pre-drop best accuracy reached 88.61%. Post-drop refinement climbed to 93.74% by epoch 66, plateaued below threshold, then made a late jump to 94.11% at epoch 89. Final accuracy was 93.47%. The run completed 35,953 steps and 93 epochs, fewer than EXP-064/065 because late batch times slowed, but still reached the required first LR drop.
- **Analysis**: The hypothesis was not supported. Stronger CutMix exposure recovered to the existing 94.11% baseline but did not clear the 94.21% threshold, so the result is too small to count under the explicit noise guard. Together with EXP-065, the local probability bracket indicates `CUTMIX_PROB=0.5` is the best tested frequency for this anchor.
- **Key Learning**: `CUTMIX_PROB=0.75` ties but does not beat the `p=0.5` CutMix anchor, closing the simple probability bracket around the current best recipe.

## Verification

- **Conditions**: all integrity checks passed; improvement threshold failed
- **Review Notes**: Results are trustworthy. The diff was limited to `train.py`, compile and ruff checks passed, final metrics were present, startup markers matched the plan, the first LR drop was reached, and no error patterns were found.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid run, but `best_test_acc=94.11%` is below the 94.21% threshold required by the +0.10 percentage-point noise guard.

## Unexplored Avenues

- Bracket `CUTMIX_ALPHA` while restoring `CUTMIX_PROB=0.5`. Probability is now locally mapped, but patch-area distribution remains untested around the successful anchor.
- Test a CutMix-specific late schedule only after alpha bracketing. EXP-066's final accuracy and late plateau suggest timing may matter, but schedule and smoothing deviations are recurring failure modes.
- Consider an orthogonal low-overhead augmentation only after the CutMix strength bracket is exhausted; direct probability changes alone are no longer promising.

## Next Steps

1. **High confidence**: Test `CUTMIX_ALPHA=0.5` with `CUTMIX_PROB=0.5`, because probability brackets failed but patch-area distribution is still untested.
2. **Medium confidence**: Test `CUTMIX_ALPHA=2.0` if alpha 0.5 fails, mapping the opposite area-distribution side around the validated anchor.
3. **Low confidence**: Try a post-drop CutMix probability schedule, but only if static alpha brackets do not produce an improvement.

## Exit Action Results

- None; no exit actions are configured for this goal.
