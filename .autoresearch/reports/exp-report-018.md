# Report EXP-018: Learned Projection Shortcuts at Downsample Transitions
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-018.md
- **Plan**: plans/plan-018.md
- **Log**: logs/exp-log-018.md

## Goal
Improve CIFAR-10 `best_test_acc` under the fixed `prepare.py` evaluation harness and 300s training budget. The current experiment-index baseline before EXP-018 was 93.23%, so the goal's noise rule required at least 93.33% for this experiment to count as an improvement.

## Idea & Hypothesis
The chosen idea was to keep the best known 28/56/112 ResNet-20 width and 21k first LR drop, but replace zero-padding shortcuts at downsample/channel-transition blocks with learned 1x1 convolution plus BatchNorm projections. The hypothesis was that learned transition mappings would improve feature transfer while adding far less compute than another broad width increase.

## Approach
Only `train.py` was modified. `BasicBlock` now used identity shortcuts for same-shape blocks and `nn.Sequential(Conv2d(1x1, stride=stride, bias=False), BatchNorm2d)` for stride/channel-change shortcuts. All hyperparameters, augmentation, seed, optimizer, compile/channels-last path, validation cadence, and the fixed time budget were preserved.

## Execution
One local run was launched with `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`. Preflight checks passed: syntax, ruff, tracked diff scope, and validation cadence were clean. The run completed without crash before the 10-minute cap, reached the first LR drop at step 21000, and produced a numeric final summary.

## Results
- **Primary metric**: 92.97% (baseline: 93.23%, delta: -0.26 points, -0.28%)
- **Observations**: The projection shortcut added only 8,176 parameters, raising params from 822,790 to 830,966. Throughput was not the failure mechanism: the run completed 38,322 steps, more than EXP-016's 34,208 steps, and reached the 21k LR drop cleanly.
- **Analysis**: The hypothesis was not supported. Learned transition shortcuts preserved throughput, but accuracy plateaued below the current zero-pad baseline. This suggests the CIFAR ResNet-20 option-A style shortcut is not the current bottleneck at 28/56/112, and adding BatchNorm-projected shortcuts may disturb the validated recipe more than it helps.
- **Key Learning**: Projection shortcuts preserved the step budget but reduced 28/56/112 accuracy to 92.97%, so transition mapping capacity is not the next useful lever.

## Verification
- **Conditions**: Metric improvement condition failed.
- **Review Notes**: Results are trustworthy: the run completed, stayed within `train.py`, preserved the fixed harness, used one GPU, validated once per epoch, and reported all expected metrics.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid result with `best_test_acc=92.97%`, below the 93.23% baseline and below the required 93.33% threshold.

## Unexplored Avenues
- Projection shortcut without BatchNorm could test whether the extra normalization, rather than the learned 1x1 mapping, caused the regression; confidence is low because the current result is substantially below baseline.
- A smaller width step such as 29/58/116 with an earlier first LR drop remains a more plausible capacity path, but should be calibrated against the step budget to avoid repeating EXP-017.
- Sparse late averaging remains plausible for smoothing late low-LR fluctuations, but implementation must avoid the per-step EMA overhead seen in EXP-004.

## Next Steps
- **High confidence**: Return to width/schedule calibration with a smaller width step and earlier first drop, because width remains the strongest validated axis but 30/60/120 was too slow.
- **Medium confidence**: Test low-frequency late weight averaging after the 21k drop, because EXP-016 and EXP-018 both show late best/final gaps without requiring architectural changes.
- **Low confidence**: Try projection shortcuts without shortcut BatchNorm only if future evidence suggests transition mappings are still worth isolating.

## Exit Action Results
