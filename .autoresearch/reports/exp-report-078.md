# Report EXP-078: Pre-Activation BasicBlock
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-078.md
- **Plan**: plans/plan-078.md
- **Log**: logs/exp-log-078.md

## Goal
Maximize CIFAR-10 `best_test_acc` under the fixed single-GPU, fixed-budget benchmark while modifying only `train.py`. The active baseline from the experiment index was 94.11% at commit `1119ff8`; with the +0.10 percentage-point noise guard, EXP-078 needed `best_test_acc >= 94.21%` to count as an improvement.

## Idea & Hypothesis
EXP-078 tested whether replacing the post-activation CIFAR ResNet block with a pre-activation `BasicBlock` would improve residual optimization under the current CutMix anchor. This was selected as a distinct architecture/topology lever after scalar regularizers, classifier-head tweaks, initialization variants, transition smoothing, CutMix brackets, and policy augmentations failed to clear the threshold. The hypothesis was that pre-activation gradient flow and identity behavior would improve post-drop convergence enough to lift the baseline from 94.11% to at least 94.21%.

## Approach
The implementation changed only `train.py`. `BasicBlock` now applies `BN/ReLU` before `conv1`, applies `BN/ReLU` before `conv2`, preserves the existing option-A shortcut from the original input, and returns the residual sum without a final block-level ReLU. `ResNet` adds a final `BatchNorm2d(w3)` plus ReLU after `layer3` and before global average pooling. Startup logging prints `Block topology: pre-activation BasicBlock`. CutMix alpha/probability, endpoint label smoothing, stage widths, optimizer, LR milestones, transforms, batch size, compile/channels-last path, seed, validation cadence, and evaluation harness were preserved.

## Execution
One local foreground run completed on GPU0 with `CUDA_VISIBLE_DEVICES=0`. Preflight checks passed: `git diff --name-only` listed only `train.py`, `python3 -m py_compile train.py` exited 0, and `uv run ruff check train.py` passed. Startup confirmed CUDA, `ResNet-20 | params: 822,846`, the pre-activation marker, and unchanged CutMix settings. The first LR drop was reached at step 21000 in epoch 54 with about 118 seconds remaining, and the run completed normally with no traceback, CUDA, error, NaN, or non-finite signatures.

## Results
- **Primary metric**: 93.92% (baseline: 94.11%, delta: -0.19pp, -0.20%)
- **Observations**: The run followed the usual post-drop climb but plateaued below the anchor. Accuracy rose from a pre-drop best of 88.82% to 91.57% at epoch 54, 93.64% at epoch 64, and peaked at 93.92% in epoch 83. It finished at 92.84%. The topology added only 56 parameters but completed 36,288 steps, fewer than many anchor-region runs.
- **Analysis**: The hypothesis was not supported. Pre-activation produced a valid, clean run, but it did not improve the CutMix anchor and appears to add enough runtime/topology friction to reduce useful fixed-budget optimization. Together with failed projection shortcuts, SE gates, stochastic depth, shallow-wide topology, and transition downsampling smoothing, this result weakens isolated residual-block architecture changes as the immediate path forward.
- **Key Learning**: Pre-activation blocks reached 93.92%, so post-activation topology remains better for the current CutMix anchor.

## Verification
- **Conditions**: All process and hard-constraint checks passed; the metric-improvement condition failed.
- **Review Notes**: Results are trustworthy. The run used one GPU, modified only `train.py`, preserved the fixed evaluation harness, reached the planned LR drop, produced numeric final summary metrics, and stayed under the 10-minute limit.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid run, but `best_test_acc=93.92%` is below the 94.11% baseline and below the 94.21% improvement threshold.

## Unexplored Avenues
- A pre-activation variant without the final BN/ReLU could isolate whether the final normalization contributed to the lower step budget, but the full planned topology already missed baseline by enough that this is low priority.
- A coupled block-topology and schedule recalibration could compensate for lower step count, but isolated schedule work has a long failure history and should not be the immediate next move.
- A shorter CutMix probability ramp remains a different mechanism that builds on EXP-073's near-miss without changing residual topology.

## Next Steps
Medium confidence: test the short CutMix probability ramp from EXP-078's fallback candidates, because it targets early representation noise while preserving static CutMix after warmup.

Medium confidence: try a very narrow training-loop stability change, such as LR warmup, only if it is clearly distinct from failed schedule-only retunes.

Low confidence: revisit architecture only with a coupled mechanism that explains both accuracy and step-budget effects; isolated residual-block topology changes are now less attractive.

## Exit Action Results
