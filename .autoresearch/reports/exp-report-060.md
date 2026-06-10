# Report EXP-060: Mixup Without Additional Label Smoothing
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-060.md
- **Plan**: plans/plan-060.md
- **Log**: logs/exp-log-060.md

## Goal
Maximize CIFAR-10 `best_test_acc` under the fixed evaluation harness and fixed training budget while modifying only `train.py`. The active baseline remains 93.97% at commit `755be2c`; because the goal requires at least +0.10 absolute percentage points to count, EXP-060 needed `best_test_acc >= 94.07%`.

## Idea & Hypothesis
The chosen idea was to retry mild `MIXUP_ALPHA=0.1` while removing the additional endpoint label smoothing used in EXP-055. EXP-055 peaked at 93.85% with `label_smoothing=0.05` applied to both mixup endpoint losses, so the hypothesis was that mixup's interpolated labels plus endpoint smoothing may have over-softened supervision. Using unsmoothed endpoint cross entropy could keep mixup's regularization while restoring enough final fit to clear 94.07%.

## Approach
`train.py` was modified only to add `MIXUP_ALPHA = 0.1` and `MIXUP_LABEL_SMOOTHING = 0.0`, print both at startup, construct one beta sampler before training, and mix each batch on-device using one scalar lambda. The loss became the weighted sum of two endpoint cross-entropy calls with `label_smoothing=MIXUP_LABEL_SMOOTHING`. The anchor architecture, optimizer, schedule, reflection crop padding, batch size, compile path, and evaluation loop were preserved.

## Execution
One local foreground run executed on GPU0 with output captured to `run.log`. Startup confirmed CUDA, `ResNet-20 | params: 822,790`, `Mixup alpha: 0.1, mixup label smoothing: 0.0`, and `Batches per epoch: 390`. The first LR drop was reached cleanly at `step 21000` with `lr: 0.0100`; there were no tracebacks, OOMs, NaNs, or runtime errors. The run completed with numeric final metrics.

## Results
- **Primary metric**: 93.81% (baseline: 93.97%, delta: -0.16pp, -0.17%)
- **Observations**: Pre-drop best reached 89.60%, then post-drop refinement climbed quickly to 93.81% by epoch 72 but did not improve afterward. Final accuracy was lower at 93.04%, with 41,074 steps, 106 epochs, 396.7 total seconds, and unchanged 822,790 parameters.
- **Analysis**: Removing endpoint label smoothing did not improve on EXP-055's 93.85%; it slightly worsened the already below-threshold mixup result. This weakens the compounded-soft-label explanation and suggests that mild mixup itself, not just mixup plus smoothing, is below the current `2e-4` label-smoothed reflection anchor under this fixed-budget recipe.
- **Key Learning**: Mild mixup remains below the current anchor even when endpoint label smoothing is removed, so label interpolation is not the next promising lever.

## Verification
- **Conditions**: all passed
- **Review Notes**: Results are trustworthy: only `train.py` was modified, compile and ruff checks passed, startup configuration matched the plan, the LR drop occurred at step 21000, parameter count stayed 822,790, and final summary metrics were present.
- **Verdict**: no-improvement
- **Verdict Basis**: The run was valid, but `best_test_acc=93.81%` is below both the 93.97% baseline and the 94.07% improvement threshold.

## Unexplored Avenues
- Partial-batch or probabilistic mixup could reduce regularization strength while retaining some sample interpolation, but the two completed mild mixup runs make this lower priority than non-mixup levers.
- CutMix-style regional mixing might behave differently from global convex interpolation, but prior cutout and mixup misses suggest any erased/mixed-image augmentation should be coupled carefully with the schedule or model.

## Next Steps
- High confidence: stop direct mild mixup variants and move to a different lever, such as a low-overhead classifier-head regularizer or targeted late-stage architectural change.
- Medium confidence: test a very small final-classifier dropout because it regularizes only the head rather than every residual feature or every target.
- Medium confidence: explore a stage-limited architecture tweak only if it has lower overhead than the all-block SE and shortcut experiments.

## Exit Action Results
