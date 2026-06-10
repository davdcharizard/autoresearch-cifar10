# Report EXP-082: Horizontal Flip Probability 0.4
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-082.md
- **Plan**: plans/plan-082.md
- **Log**: logs/exp-log-082.md

## Goal
Improve CIFAR-10 `best_test_acc` under the fixed benchmark harness while modifying only `train.py`. The active baseline before EXP-082 was 94.11% from commit `1119ff8`, and the goal's noise guard required at least +0.10 percentage points, so this experiment needed `best_test_acc >= 94.21%` to count as an improvement.

## Idea & Hypothesis
EXP-082 tested whether the current CutMix anchor is slightly over-regularized by the default horizontal flip probability. The chosen change was to keep the validated reflection crop and CutMix recipe intact while reducing `RandomHorizontalFlip` from the default 0.5 probability to explicit `p=0.4`.

The hypothesis was that a modest reduction in flip frequency would reduce over-regularization enough to raise `best_test_acc` from 94.11% to at least 94.21%.

## Approach
The implementation changed only `train.py`: `transforms.RandomHorizontalFlip()` became `transforms.RandomHorizontalFlip(p=0.4)`. It also added the startup marker `RandomHorizontalFlip p: 0.4` so the run log verifies the intended transform. Reflection crop padding 4, CutMix alpha/probability/label smoothing, clean label smoothing, architecture, optimizer, LR schedule, batch size, seed, compile/channels-last, and validation cadence were unchanged.

## Execution
One local foreground run was launched on GPU0 with output captured to `run.log`. Startup confirmed CUDA, the flip marker, unchanged CutMix settings, `ResNet-20 | params: 822,790`, and 390 batches per epoch. The first LR drop was reached at step 21000 with `lr: 0.0100`, and the run completed the fixed 300s training budget without errors.

## Results
- **Primary metric**: 94.36% (baseline: 94.11%, delta: +0.25pp, above the 94.21% threshold)
- **Observations**: The run crossed threshold at epoch 75 with 94.22%, improved to 94.26% at epoch 81, and peaked at 94.36% at epoch 88. Final accuracy was 93.86%, final loss was 0.2479, and the run completed 39,034 steps in 101 epochs.
- **Analysis**: The result supports the mild over-regularization hypothesis. Unlike reducing reflection crop padding to 3, lowering flip probability produced a margin large enough to clear the explicit noise guard while preserving all other validated anchor components.
- **Key Learning**: Lowering horizontal flip probability to 0.4 improves the CutMix anchor, confirming mild spatial de-regularization can beat the prior best.

## Verification
- **Conditions**: All necessary conditions passed.
- **Review Notes**: Results are trustworthy. The metric came from the intended run, only `train.py` changed, parameter count remained 822,790, the first LR drop was reached, and the fixed training budget was used.
- **Verdict**: improvement
- **Verdict Basis**: EXP-082 reached 94.36%, exceeding both the 94.11% baseline and the 94.21% noise-guard threshold.

## Unexplored Avenues
- Bracket horizontal flip probability around the new anchor, especially `p=0.35` or `p=0.45`, to determine whether 0.4 is a local optimum or just the first useful point.
- Revisit reflection crop padding only as a coupled spatial de-regularization bracket; padding 3 alone was sub-threshold, but the new lower-flip anchor may shift the best crop strength.

## Next Steps
- **High confidence**: Test `RandomHorizontalFlip(p=0.35)` on top of the new anchor to bracket whether further spatial de-regularization helps.
- **Medium confidence**: Test `RandomHorizontalFlip(p=0.45)` if the lower side regresses, to bracket the optimum tightly around 0.4.
- **Low confidence**: Combine lower flip probability with crop padding 3 only after single-axis flip bracketing, because near-miss stacking has regressed before.

## Exit Action Results
- None; the goal has no active exit actions.
