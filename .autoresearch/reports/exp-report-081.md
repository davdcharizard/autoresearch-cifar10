# Report EXP-081: Reflection Crop Padding 3
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-081.md
- **Plan**: plans/plan-081.md
- **Log**: logs/exp-log-081.md

## Goal
Improve CIFAR-10 `best_test_acc` under the fixed benchmark harness while modifying only `train.py`. The active baseline is 94.11% from commit `1119ff8`, and the goal's noise guard requires at least +0.10 percentage points, so EXP-081 needed `best_test_acc >= 94.21%` to count as an improvement.

## Idea & Hypothesis
EXP-081 tested whether the current CutMix anchor is slightly over-regularized by 4-pixel reflection crop jitter. The chosen change was to preserve `padding_mode="reflect"` but reduce `RandomCrop` padding from 4 to 3, leaving CutMix, label smoothing, architecture, optimizer, LR schedule, batch size, seed, compile/channels-last, and validation cadence unchanged.

The hypothesis was that a small reduction in spatial jitter would improve late clean accuracy enough to raise best accuracy from 94.11% to at least 94.21%.

## Approach
The implementation changed only the training transform in `train.py`: `transforms.RandomCrop(32, padding=4, padding_mode="reflect")` became `padding=3`. It also added the startup marker `RandomCrop padding: 3 reflect` so `run.log` verifies the intended transform setting. There were no deviations from the plan.

## Execution
One local foreground run was launched on GPU0 with output captured to `run.log`. Startup confirmed the CUDA device, the `RandomCrop padding: 3 reflect` marker, unchanged CutMix settings, `ResNet-20 | params: 822,790`, and 390 batches per epoch. Training was healthy, reached the first LR drop at step 21000 with `lr: 0.0100`, and completed the fixed 300s training budget without errors.

## Results
- **Primary metric**: 94.18% (baseline: 94.11%, delta: +0.07pp, below the 94.21% threshold)
- **Observations**: The run peaked at epoch 81, then stayed below that level through epoch 103. Final accuracy was 93.31%, final loss was 0.3047, and the run completed 39,856 steps in 103 epochs.
- **Analysis**: Reducing crop padding produced the best post-baseline near miss so far, but the gain is inside the explicit noise guard. The result suggests full 4-pixel reflection padding may be slightly stronger than optimal, but this single-axis reduction is not enough to justify changing the anchor.
- **Key Learning**: Weaker reflection crop jitter gives only a sub-threshold near miss, so spatial augmentation strength is not solved by padding 3 alone.

## Verification
- **Conditions**: Scope, syntax, style, startup markers, scheduler timing, run completion, and hard constraints passed; the improvement-threshold condition failed.
- **Review Notes**: Results are trustworthy. The metric came from the intended run, only `train.py` changed, parameter count remained 822,790, and the first LR drop was reached.
- **Verdict**: no-improvement
- **Verdict Basis**: EXP-081 reached 94.18%, which is valid but below the required 94.21% threshold for a counted improvement.

## Unexplored Avenues
- Test horizontal flip probability `p=0.4` as a different no-overhead spatial augmentation strength knob. It weakens invariance rather than crop translation, so it probes a related but distinct mechanism.
- Test a coupled crop-padding change only if paired with an independently motivated stabilizer. Padding 3 alone is too small to clear the noise guard.

## Next Steps
- **Medium confidence**: Try lower horizontal flip probability `p=0.4`, preserving padding 4 and the CutMix anchor, to continue the narrow over-regularization probe with a different spatial knob.
- **Medium confidence**: Try final-stage width 104 if the next augmentation micro-tune misses; it may trade slight capacity for more post-drop optimization while staying close to the 28/56/112 anchor.
- **Low confidence**: Revisit near-miss combinations only with a clear interaction rationale; EXP-075 already showed naive near-miss stacking can regress.

## Exit Action Results
- None; the goal has no active exit actions.
