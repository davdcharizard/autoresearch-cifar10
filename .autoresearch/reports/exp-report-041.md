# Report EXP-041: Weight Decay 1.5e-4 Local Bracket
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-041.md
- **Plan**: plans/plan-041.md
- **Log**: logs/exp-log-041.md

## Goal
Improve CIFAR-10 `best_test_acc` under the fixed `prepare.py` evaluation harness while modifying only `train.py`. The pre-experiment baseline was 93.97% from EXP-038, and the goal's +0.10 percentage-point rule required EXP-041 to reach at least 94.07% to count as an improvement.

## Idea & Hypothesis
EXP-041 tested a local weight-decay bracket around the current successful `WEIGHT_DECAY = 2e-4` anchor. The hypothesis was that `1.5e-4` might preserve the regularization gain while reducing possible over-shrinkage, producing `best_test_acc >= 94.07%`.

## Approach
Only `train.py` changed during the run: `WEIGHT_DECAY = 2e-4` became `WEIGHT_DECAY = 1.5e-4`. Architecture, reflected `RandomCrop`, `BATCH_SIZE = 128`, `LR = 0.1`, `LR_MILESTONES = [21000, 64000]`, `MOMENTUM = 0.9`, `label_smoothing=0.05`, FP32 channels-last compile path, seed, fixed time budget, and once-per-epoch validation were preserved. No deviations from the plan were needed.

## Execution
One local single-GPU run was launched on GPU 1. Startup confirmed CUDA execution, 822,790 parameters, the fixed 300s training budget, and 390 batches per epoch. The first LR drop fired correctly at step 21000 with `lr: 0.0100`, no second LR drop occurred, and the run completed cleanly under the 10-minute wall-clock limit.

## Results
- **Primary metric**: 93.61% (baseline: 93.97%, delta: -0.36 points, -0.38%)
- **Observations**: Pre-drop accuracy reached 89.19% by epoch 53. Post-drop accuracy jumped to 91.58% at epoch 54, peaked at 93.61% by epoch 72, then drifted below the peak through the final epoch.
- **Analysis**: The hypothesis was not supported. Reducing weight decay below `2e-4` weakened the current label-smoothed reflection anchor, while EXP-039 already showed `3e-4` is too strong. Together, EXP-039 and EXP-041 bracket `2e-4` as the local decay setting.
- **Key Learning**: `2e-4` is locally bracketed for the current anchor; both `1.5e-4` and `3e-4` regress, so move away from isolated weight-decay retuning.

## Verification
- **Conditions**: metric improvement condition failed; hard constraints passed
- **Review Notes**: Results are trustworthy: the run completed cleanly, modified only `train.py`, preserved the fixed harness and single-GPU setup, reported a numeric metric, preserved the expected schedule behavior, and stayed under the 10-minute wall-clock cap.
- **Verdict**: no-improvement
- **Verdict Basis**: valid run, but 93.61% is below both the 93.97% baseline and the 94.07% improvement threshold.

## Unexplored Avenues
- Test `LR = 0.08` on the `2e-4` anchor to probe the lower-LR side after EXP-040 ruled out `LR = 0.12`.
- Test a bounded post-drop averaging method only if it avoids the known per-step EMA overhead and long equal-averaging collapse.
- Consider non-weight-decay regularization or optimizer dynamics; isolated decay values around `2e-4` now have a poor local bracket.

## Next Steps
Try `LR = 0.08` with low-to-medium confidence if continuing scalar optimizer probes; it is the remaining simple side of LR bracketing after `0.12` failed.

Try a carefully bounded late post-drop averaging or EMA variant with low confidence only if accepting higher implementation risk.

Avoid further isolated weight-decay retuning with high confidence; `1.5e-4`, `2e-4`, and `3e-4` now map the useful local bracket.

## Exit Action Results
