# Report EXP-058: Squeeze-and-Excitation BasicBlocks
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-058.md
- **Plan**: plans/plan-058.md
- **Log**: logs/exp-log-058.md

## Goal
Improve CIFAR-10 `best_test_acc` under the fixed `prepare.py` evaluation harness while modifying only `train.py`. The active experiment-index baseline is 93.97% from commit `755be2c`; with the +0.10 percentage-point noise guard, EXP-058 needed at least 94.07% to count as an improvement.

## Idea & Hypothesis
EXP-058 tested lightweight Squeeze-and-Excitation channel recalibration in every residual block. The hypothesis was that SE gates would improve representation quality inside the validated 28/56/112 ResNet-20 backbone without changing the optimizer, augmentation, schedule, smoothing, shortcut type, validation cadence, or fixed evaluation harness.

## Approach
`train.py` was changed to add `USE_SE=True`, `SE_REDUCTION=16`, and an `SEBlock` using adaptive average pooling plus two `1x1` convolution layers. Each `BasicBlock` applies the SE gate after `bn2(conv2)` and before shortcut addition. The implementation used 4D `Conv2d` gates to stay channels-last friendly and enforced a minimum bottleneck width of 4 for the first stage. All other anchor settings were preserved.

## Execution
One local foreground run was launched on GPU0 with output captured to `run.log`. Preflight checks passed and no retries were needed. Startup confirmed `SE blocks: enabled, reduction=16`, unchanged `Batches per epoch: 390`, and `num_params=830,143`. The first LR drop occurred at step 21000, so the comparison is valid.

## Results
- **Primary metric**: 93.71% (baseline: 93.97%, delta: -0.26 percentage points, -0.28%)
- **Observations**: SE added 7,353 parameters and reduced the step budget to 25,716 steps. The run reached the 21k LR drop and climbed post-drop from 91.58% at epoch 54 to a best of 93.71% at epoch 62, then ended at 93.65%.
- **Analysis**: The hypothesis is rejected for this implementation. SE channel recalibration did not recover enough post-drop accuracy to beat the anchor, and the added residual-block gating appears to trade useful step budget and/or optimization fit for a representation mechanism that this small fixed-budget model does not need.
- **Key Learning**: SE channel gates preserve the LR milestone but underperform the current anchor; lightweight attention is weaker than the plain 28/56/112 block here.

## Verification
- **Conditions**: all process conditions passed; metric threshold failed.
- **Review Notes**: Results are trustworthy. The run completed, produced numeric summary metrics, modified only `train.py`, preserved the fixed harness, and verified the LR drop, SE startup setting, and batch geometry.
- **Verdict**: no-improvement.
- **Verdict Basis**: `best_test_acc=93.71%` is below both the 93.97% baseline and the 94.07% improvement threshold.

## Unexplored Avenues
- Try a cheaper stage-limited SE variant only in the final stage. This may reduce overhead, but the all-block result makes the expected upside modest.
- Try non-parametric or near-free transition improvements, such as average-pool option-A downsampling, because they target architecture quality without adding gates to every block.

## Next Steps
- Medium confidence: test average-pool option-A downsample shortcut as a low-parameter architecture probe distinct from learned projection shortcuts.
- Medium confidence: test mixup without label smoothing only if revisiting coupled regularization; recent mixup reached 93.85% but remains below threshold.
- Higher confidence: continue searching for changes that preserve the current anchor recipe and avoid per-block overhead, because SE reduced useful coverage and remained below baseline.

## Exit Action Results
