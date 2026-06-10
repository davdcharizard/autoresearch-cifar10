# Report EXP-062: Compact ResNet-14 with Moderate Width Increase
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-062.md
- **Plan**: plans/plan-062.md
- **Log**: logs/exp-log-062.md

## Goal
Improve CIFAR-10 `best_test_acc` under the fixed benchmark harness while modifying only `train.py`. The active baseline from the experiment index is 93.97% at commit `755be2c`, and the goal requires at least a +0.10 percentage-point gain, so EXP-062 needed `best_test_acc >= 94.07%` to count as an improvement.

## Idea & Hypothesis
EXP-062 tested a compact depth/width tradeoff: reduce the model from ResNet-20 to ResNet-14 by setting `NUM_BLOCKS = 2`, while increasing channel widths to `(32, 64, 128)`. The hypothesis was that fewer residual blocks would recover enough throughput to afford wider channels, reach the first LR drop reliably, and improve the fixed-budget capacity/throughput balance.

## Approach
The implementation changed only two constants in `train.py`: `NUM_BLOCKS` from 3 to 2 and `STAGE_WIDTHS` from `(28, 56, 112)` to `(32, 64, 128)`. Optimizer, LR schedule, augmentation, label smoothing, weight decay, compile/channels-last behavior, validation cadence, and timing logic were left unchanged for clean attribution. Startup confirmed `ResNet-14 | params: 685,994`.

## Execution
One local foreground run completed on GPU0 with output captured to `run.log`. The run was healthy: startup completed, compile/style checks passed, no traceback/OOM/runtime-error/non-finite patterns were found, and the first LR drop occurred at `step 21000 ep 54` with `lr: 0.0100`. The compact model reached 132 epochs and 51,471 steps in the 300-second training budget.

## Results
- **Primary metric**: 93.51% (baseline: 93.97%, delta: -0.46pp, -0.49%)
- **Observations**: Throughput improved substantially relative to the current anchor, but the best accuracy peaked shortly after the first LR drop and the final checkpoint decayed to 92.55%.
- **Analysis**: The hypothesis failed. The extra steps from reduced depth did not compensate for lost representational depth, even with wider channels and a lower parameter count than the 822,790-param anchor.
- **Key Learning**: Shallower wider ResNet-14 reached 51,471 steps but peaked at 93.51%, so reduced depth loses too much representation despite speed.

## Verification
- **Conditions**: all execution-integrity conditions passed.
- **Review Notes**: Results are trustworthy: `run.log` contains the expected architecture marker, batch geometry, LR-drop marker, final metrics, and no error patterns; tracked code diff was limited to `train.py`.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid completed run, but `best_test_acc=93.51%` is below the 93.97% baseline and the 94.07% required improvement threshold.

## Unexplored Avenues
- Preserve ResNet-20 depth and redistribute capacity within stages rather than removing one block per stage; EXP-062 suggests depth loss is the dominant failure.
- Try cheaper localized architecture changes, such as stage-limited attention or late-stage width redistribution, where the intervention does not reduce residual depth.

## Next Steps
Medium confidence: move away from naive shallow-wide variants and test a localized architecture perturbation that preserves the proven ResNet-20 depth.

Medium confidence: stage-3-only SE remains a plausible cheaper attention variant because EXP-058 tested broad all-block SE, not a final-stage-only gate.

Low-to-medium confidence: explore scheduled late regularization only if it preserves the high-LR representation-learning phase exactly.

## Exit Action Results
