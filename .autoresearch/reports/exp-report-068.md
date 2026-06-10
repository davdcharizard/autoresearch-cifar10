# Report EXP-068: CutMix Alpha 2.0
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-068.md
- **Plan**: plans/plan-068.md
- **Log**: logs/exp-log-068.md

## Goal
Improve CIFAR-10 `best_test_acc` under the fixed single-GPU, fixed-budget harness while modifying only `train.py`. The active baseline before EXP-068 was 94.11% from EXP-064 at commit `1119ff8`; the goal's noise guard requires at least 94.21% to count as an improvement.

## Idea & Hypothesis
EXP-068 tested the upper CutMix alpha bracket: keep the validated `CUTMIX_PROB=0.5` and raise `CUTMIX_ALPHA` from 1.0 to 2.0. The hypothesis was that a less-variable patch-area distribution might reduce extreme mixed-label noise and stabilize late refinement enough to clear 94.21%.

## Approach
The implementation was a one-line `train.py` configuration change, setting `CUTMIX_ALPHA = 2.0` while preserving the EXP-064 anchor's architecture, optimizer, LR schedule, reflection padding, label smoothing, weight decay, compile/channels-last path, validation cadence, and CutMix probability. There were no deviations from the plan.

## Execution
One foreground local run was launched on GPU0 with output captured to `run.log`. Startup confirmed CUDA, ResNet-20 with 822,790 parameters, `CutMix alpha: 2.0, prob: 0.5, label smoothing: 0.05`, the 300s budget, and 390 batches per epoch. The first LR drop occurred at step 21000 with LR 0.0100; the run completed cleanly and reported final metrics.

## Results
- **Primary metric**: 94.00% (baseline: 94.11%, delta: -0.11pp, -0.12%)
- **Observations**: Accuracy climbed after the first LR drop from 91.77% at epoch 54 to 94.00% at epoch 79, but then plateaued below threshold. Final accuracy was 93.66%, with `training_seconds=300.0`, `total_seconds=395.2`, `num_epochs=102`, and `num_steps=39747`.
- **Analysis**: Raising alpha did not validate the hypothesis. Together with EXP-067's alpha 0.5 result, this brackets the simple static CutMix alpha axis around the successful `alpha=1.0, p=0.5` anchor.
- **Key Learning**: CutMix alpha values on both sides of 1.0 miss the threshold, making `alpha=1.0` the local static alpha optimum.

## Verification
- **Conditions**: All process and scope conditions passed; the improvement threshold condition failed for improvement classification.
- **Review Notes**: Results are trustworthy: the run completed cleanly, reported numeric metrics, used the intended startup configuration, hit the LR milestone, and the tracked code diff was limited to `train.py`.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid result with `best_test_acc=94.00%`, below the required 94.21% threshold.

## Unexplored Avenues
- A post-drop CutMix probability taper could preserve early regional mixing while reducing late mixed-label noise; this is a different mechanism than static alpha strength and should be treated as a medium-risk schedule interaction.
- A CutMix-specific smoothing interaction remains possible, but label-smoothing deviations are a high-importance failed family and need a strong rationale before retrying.

## Next Steps
Prefer a post-drop CutMix probability taper with medium confidence if continuing within the successful regional-mixing mechanism. Otherwise move to a distinct optimizer or late-refinement interaction that preserves the current `alpha=1.0, p=0.5` CutMix anchor.

## Exit Action Results
