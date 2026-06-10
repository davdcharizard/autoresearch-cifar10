# Report EXP-067: CutMix Alpha 0.5
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-067.md
- **Plan**: plans/plan-067.md
- **Log**: logs/exp-log-067.md

## Goal
Improve CIFAR-10 `best_test_acc` under the fixed single-GPU, fixed-budget harness while modifying only `train.py`. The active baseline before EXP-067 was 94.11% from EXP-064 at commit `1119ff8`; the goal's noise guard requires at least 94.21% to count as an improvement.

## Idea & Hypothesis
EXP-067 tested the lower CutMix alpha bracket: keep the validated `CUTMIX_PROB=0.5` and lower `CUTMIX_ALPHA` from 1.0 to 0.5. The hypothesis was that a higher-variance patch-area distribution might improve regional-replacement regularization enough to clear 94.21%.

## Approach
The implementation was a one-line `train.py` configuration change, setting `CUTMIX_ALPHA = 0.5` while preserving the EXP-064 anchor's architecture, optimizer, LR schedule, reflection padding, label smoothing, weight decay, compile/channels-last path, validation cadence, and CutMix probability. There were no deviations from the plan.

## Execution
One foreground local run was launched on GPU0 with output captured to `run.log`. Startup confirmed CUDA, ResNet-20 with 822,790 parameters, `CutMix alpha: 0.5, prob: 0.5, label smoothing: 0.05`, the 300s budget, and 390 batches per epoch. The first LR drop occurred at step 21000 with LR 0.0100; the run completed cleanly and reported final metrics.

## Results
- **Primary metric**: 94.07% (baseline: 94.11%, delta: -0.04pp, -0.04%)
- **Observations**: Accuracy climbed quickly after the LR drop, reaching 93.94% by epoch 68 and peaking at 94.07% at epoch 77, but did not improve after that. Final accuracy was 93.65%, with `training_seconds=300.0`, `total_seconds=394.7`, `num_epochs=101`, and `num_steps=39006`.
- **Analysis**: Lowering alpha did not validate the hypothesis. The result was close to the CutMix anchor but below both the 94.11% baseline and the 94.21% improvement threshold, suggesting `alpha=0.5` makes CutMix at least slightly too variable for this fixed-budget recipe.
- **Key Learning**: Lower CutMix alpha preserves most of the regional-mixing benefit but does not beat the `alpha=1.0, p=0.5` anchor.

## Verification
- **Conditions**: All process and scope conditions passed; the improvement threshold condition failed for improvement classification.
- **Review Notes**: Results are trustworthy: the run completed cleanly, reported numeric metrics, used the intended startup configuration, hit the LR milestone, and the tracked code diff was limited to `train.py`.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid result with `best_test_acc=94.07%`, below the required 94.21% threshold.

## Unexplored Avenues
- Test the opposite alpha bracket, `CUTMIX_ALPHA=2.0`, to see whether less-variable patch sizes stabilize late refinement better than both alpha 0.5 and alpha 1.0.
- Test a post-drop CutMix schedule only if the next static alpha bracket also fails; it may preserve early regional mixing while reducing late mixed-label noise, but it is less clean than a scalar bracket.

## Next Steps
Prefer `CUTMIX_ALPHA=2.0` with medium confidence as the next clean local bracket inside the validated CutMix mechanism. If that fails, deprioritize static CutMix strength tuning and move to a more distinct mechanism such as late regional-mixing scheduling or a small optimizer interaction.

## Exit Action Results
