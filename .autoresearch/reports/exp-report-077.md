# Report EXP-077: Anti-Aliased Residual Downsample
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-077.md
- **Plan**: plans/plan-077.md
- **Log**: logs/exp-log-077.md

## Goal
Maximize CIFAR-10 `best_test_acc` under the fixed single-GPU, fixed-budget benchmark while modifying only `train.py`. The active baseline from the experiment index was 94.11% at commit `1119ff8`; with the +0.10 percentage-point noise guard, EXP-077 needed `best_test_acc >= 94.21%` to count as an improvement.

## Idea & Hypothesis
EXP-077 tested anti-aliased downsampling in the learned residual branch of stride-2 `BasicBlock` transitions. The idea was selected because it was a localized architecture change distinct from the recently bracketed CutMix, label-smoothing, classifier-head, and initialization families. The hypothesis was that average-pooling before the learned transition convolution would reduce aliasing enough to improve post-drop features while preserving the validated option-A shortcut and CutMix anchor.

## Approach
The implementation changed only `train.py`. For transition blocks with `stride != 1`, `conv1` now uses stride 1 and the residual branch input is average-pooled before that convolution. Blocks with stride 1 are unchanged. The option-A shortcut path remains the existing strided slicing plus zero-channel padding. A startup marker, `Residual downsample: avgpool before stride-2 conv`, was added so the run log identifies the variant. CutMix alpha/probability, label smoothing, stage widths, optimizer, LR milestones, transforms, batch size, compile/channels-last path, seed, validation cadence, and evaluation harness were preserved.

## Execution
One local foreground run completed on GPU0 with `CUDA_VISIBLE_DEVICES=0`. Preflight checks passed: `git diff --name-only` listed only `train.py`, `python3 -m py_compile train.py` exited 0, and `uv run ruff check train.py` passed. Startup confirmed CUDA, unchanged parameter count `822,790`, unchanged CutMix settings, and the residual-downsample marker. The first LR drop was reached at step 21000 in epoch 54, and the run completed normally with no traceback, CUDA, error, NaN, or non-finite signatures.

## Results
- **Primary metric**: 93.99% (baseline: 94.11%, delta: -0.12pp, -0.13%)
- **Observations**: The run behaved cleanly but trailed the anchor. Accuracy jumped after the first LR drop from a pre-drop best of 88.05% to 91.97% at epoch 54, 93.35% at epoch 56, and 93.91% at epoch 71. A late spike reached 93.99% at epoch 93, then the final checkpoint ended at 93.90%.
- **Analysis**: The hypothesis was not supported. Smoothing the learned residual transition path preserved validity and reached the normal post-drop convergence band, but it did not beat the CutMix anchor. Together with EXP-059's failed shortcut average-pooling test, transition downsampling smoothing now looks weaker than the original strided option-A plus stride-2 residual convolution design under this fixed recipe.
- **Key Learning**: Average-pooling the learned residual downsample path reaches only 93.99%, so transition anti-aliasing lags the CutMix anchor.

## Verification
- **Conditions**: All process and hard-constraint checks passed; the metric-improvement condition failed.
- **Review Notes**: Results are trustworthy. The run used one GPU, modified only `train.py`, preserved the fixed evaluation harness, reached the planned LR drop, produced a numeric metric block, and stayed within the 10-minute limit.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid run, but `best_test_acc=93.99%` is below the 94.11% baseline and below the 94.21% improvement threshold.

## Unexplored Avenues
- A full pre-activation `BasicBlock` remains a separate architecture hypothesis because it changes normalization/activation topology rather than only the downsampling primitive.
- A very small blur kernel before stride-2 convolution could test anti-aliasing more gently, but the two failed average-pool transition probes make this lower priority than other architecture or schedule interactions.

## Next Steps
Medium confidence: test a pre-activation BasicBlock only with a careful plan, since architecture remains one of the few less-exhausted families.

Medium confidence: revisit early-training stability with a shorter CutMix ramp or LR warmup only if it avoids the known schedule-only and clean-warmup failure modes.

Low confidence: explore a coupled architecture-plus-schedule adjustment, because isolated local architecture tweaks are increasingly clustering below the 94.21% threshold.

## Exit Action Results
