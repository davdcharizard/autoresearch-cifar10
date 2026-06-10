# Report EXP-063: Final-Stage-Only SE Gate
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-063.md
- **Plan**: plans/plan-063.md
- **Log**: logs/exp-log-063.md

## Goal
Improve CIFAR-10 `best_test_acc` under the fixed benchmark harness while modifying only `train.py`. The active experiment-index baseline is 93.97% at commit `755be2c`; with the explicit +0.10 percentage-point noise guard, EXP-063 needed at least 94.07% to count as an improvement.

## Idea & Hypothesis
EXP-063 tested Squeeze-and-Excitation channel recalibration only in the final residual stage. The hypothesis was that `layer3` gates could improve class-semantic feature calibration while avoiding the early-feature perturbation and repeated overhead seen in EXP-058 all-block SE.

## Approach
`train.py` added a channels-last friendly `SEBlock` using adaptive average pooling and two `1x1` convolutions, threaded `use_se` through `BasicBlock` and `_make_layer`, and enabled the gate only for `layer3`. ResNet-20 depth, `(28, 56, 112)` widths, optimizer, LR milestones, augmentation, label smoothing, compile/channels-last behavior, validation cadence, and timing logic were preserved. Startup printed `SE stage: layer3 only, reduction: 16` for verification.

## Execution
One local foreground run completed on GPU0 with output captured to `run.log`. Preflight compile/style checks passed, the tracked code diff was limited to `train.py`, startup confirmed `ResNet-20 | params: 827,851` and unchanged `Batches per epoch: 390`, and the first LR drop occurred at `step 21000 ep 54` with `lr: 0.0100`. No retries or experimental adjustments were needed.

## Results
- **Primary metric**: 93.26% (baseline: 93.97%, delta: -0.71 percentage points, -0.76%)
- **Observations**: The run completed 89 epochs and 34,502 steps in the 300-second training budget, with final accuracy 92.94%, final loss 0.2623, peak VRAM 661.0 MB, and 827,851 parameters.
- **Analysis**: The hypothesis is rejected. Final-stage-only SE preserved the LR milestone and reduced overhead versus EXP-058 all-block SE, but the best accuracy fell to 93.26%, below both EXP-058's 93.71% and the plain anchor's 93.97%.
- **Key Learning**: Layer3-only SE underperformed all-block SE and the anchor, so SE channel attention is a poor fit for this fixed-budget ResNet-20 recipe.

## Verification
- **Conditions**: all process and integrity conditions passed; metric threshold failed.
- **Review Notes**: Results are trustworthy. The run completed, produced numeric summary metrics, modified only `train.py`, preserved the fixed harness, reached the first LR drop, and verified the stage-limited SE marker and batch geometry.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid completed run, but `best_test_acc=93.26%` is below the 93.97% baseline and the 94.07% required improvement threshold.

## Unexplored Avenues
- Test non-SE attention only if it has a materially different mechanism and negligible overhead; two SE variants now provide negative evidence against channel squeeze gates.
- Try localized architecture probes that preserve ResNet-20 depth without attention gates, such as careful stage-width redistribution, though prior width evidence is weak.

## Next Steps
Medium confidence: deprioritize SE and other residual-block attention gates; use the next loop to search for a distinct mechanism that preserves the current anchor's throughput and LR timing.

Medium confidence: consider a very small late-stage capacity redistribution only if the brainstorm can avoid repeating the known width-beyond-anchor failure mode.

Low-to-medium confidence: revisit regional augmentation or scheduled regularization only with a clearly different mechanism from failed mixup, Cutout, dropout, and stochastic-depth variants.

## Exit Action Results
