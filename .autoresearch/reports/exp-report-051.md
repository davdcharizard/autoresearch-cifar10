# Report EXP-051: Partial Residual-Branch BN Scale Initialization
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-051.md
- **Plan**: plans/plan-051.md
- **Log**: logs/exp-log-051.md

## Goal
Maximize CIFAR-10 `best_test_acc` under the fixed benchmark harness. The active baseline is 93.97% at commit `755be2c`, and the goal requires at least +0.10 percentage points to count as an improvement, so EXP-051 needed `best_test_acc >= 94.07%`.

## Idea & Hypothesis
EXP-051 tested partial residual-branch BatchNorm scale initialization: set each `BasicBlock.bn2.weight` to `0.1` after normal initialization. The hypothesis was that a nonzero residual scale could keep useful residual learning active while adding some identity-bias stability, avoiding the undertraining seen in EXP-028's full zero-gamma initialization.

## Approach
The implementation modified `train.py` only. After `self.apply(self._weights_init)` in `ResNet.__init__`, the model loops over `self.modules()` and applies `init.constant_(m.bn2.weight, 0.1)` to every `BasicBlock`. Model width, batch size, optimizer, weight decay, LR milestones, reflection crop padding, label smoothing, compile, channels-last execution, and validation cadence were all preserved.

## Execution
One local foreground run was launched on GPU0 with output captured in `run.log`. Startup was clean, `num_params` stayed at 822,790, and the first LR drop occurred at step 21000 with about 146s remaining. The run completed normally within budget and produced final summary metrics.

## Results
- **Primary metric**: 93.64% (baseline: 93.97%, delta: -0.33pp, -0.35%)
- **Observations**: Pre-drop best was 88.99%, post-drop accuracy climbed to 93.39% by epochs 61-63, and the final best was 93.64% at epoch 94 before late evaluations drifted lower.
- **Analysis**: The partial scale avoided EXP-028's severe full-zero-gamma collapse, but it still lagged the default-scale anchor. This suggests the fixed 300s recipe benefits from fully active residual branches more than from identity-biased initialization.
- **Key Learning**: `bn2.weight=0.1` is trainable but still underperforms the anchor, so residual-branch BN down-scaling is not a useful isolated lever here.

## Verification
- **Conditions**: all passed
- **Review Notes**: Results are trustworthy: only `train.py` changed, compile and ruff passed, the run completed within 402.2s, LR dropped at step 21000, and final metrics were numeric with unchanged parameter count.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid run with `best_test_acc=93.64%`, below the 94.07% improvement threshold.

## Unexplored Avenues
- A less aggressive partial scale such as `0.5` might reduce the initialization shock, but the combined EXP-028/EXP-051 evidence makes residual-branch BN scale retuning lower priority than unrelated no-overhead levers.
- A time-dependent residual scaling schedule could restore full branch strength later, but it would add implementation complexity and a new dynamic mechanism without strong local evidence.

## Next Steps
- **Hybrid post-drop cosine tail (medium confidence)**: Preserve the validated 21k first drop, but smooth only the LR 0.01 tail to test late plateau stability without disrupting the high-LR window.
- **Very mild residual dropout (low confidence)**: Try a tiny residual-branch stochastic regularizer only if it preserves throughput; risk is added noise and slower convergence.
- **Search for no-overhead optimizer or regularization interactions (medium confidence)**: Prioritize changes that preserve the current step budget and avoid already-failed isolated LR, weight-decay, augmentation, EMA, and BN-momentum levers.

## Exit Action Results
- No exit actions defined.
