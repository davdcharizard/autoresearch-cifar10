# Report EXP-069: Post-Drop CutMix Probability Taper
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-069.md
- **Plan**: plans/plan-069.md
- **Log**: logs/exp-log-069.md

## Goal
Improve CIFAR-10 `best_test_acc` under the fixed single-GPU, fixed-budget harness while modifying only `train.py`. The active baseline before EXP-069 was 94.11% from EXP-064 at commit `1119ff8`; the goal's noise guard requires at least 94.21% to count as an improvement.

## Idea & Hypothesis
EXP-069 tested a temporal CutMix schedule: keep the validated `CUTMIX_ALPHA=1.0` and pre-drop `CUTMIX_PROB=0.5`, then reduce CutMix probability to 0.25 after the first LR drop at step 21000. The hypothesis was that early regional mixing should remain useful during high-LR representation learning, while lower post-drop mixing might reduce late mixed-label noise and improve refinement.

## Approach
The implementation added `CUTMIX_POST_DROP_PROB = 0.25` and tied `CUTMIX_TAPER_STEP` to `LR_MILESTONES[0]`. The training loop computes the current CutMix probability from the global step before sampling the augmentation, and startup plus one-time taper markers were added for auditability. Architecture, optimizer, LR schedule, reflection padding, endpoint label smoothing, weight decay, compile/channels-last behavior, and validation cadence were preserved.

## Execution
One foreground local run was launched on GPU0 with output captured to `run.log`. Startup confirmed CUDA, ResNet-20 with 822,790 parameters, `CutMix alpha: 1.0, prob: 0.5, label smoothing: 0.05`, post-drop probability 0.25 after step 21000, the 300s budget, and 390 batches per epoch. The first LR drop occurred at step 21000 with LR 0.0100 and the taper marker printed immediately afterward. The run completed cleanly and reported final metrics.

## Results
- **Primary metric**: 93.73% (baseline: 94.11%, delta: -0.38pp, -0.40%)
- **Observations**: Accuracy climbed after the first LR drop from 91.94% at epoch 54 to 93.72% by epoch 63, then only nudged to 93.73% at epoch 89 before degrading to a final checkpoint accuracy of 92.94%. The run used `training_seconds=300.0`, `total_seconds=394.5`, `num_epochs=102`, `num_steps=39606`, and `peak_vram_mb=660.4`.
- **Analysis**: The temporal-noise hypothesis was not supported. Reducing CutMix probability after the first LR drop weakened the EXP-064 anchor more than static probability brackets did, suggesting continued `p=0.5` regional mixing is not the main late-refinement limiter.
- **Key Learning**: Post-drop CutMix probability tapering to 0.25 underperforms the static `p=0.5` anchor, so temporal CutMix weakening is not promising.

## Verification
- **Conditions**: All process, metric-extraction, and hard-constraint checks passed; the improvement threshold condition failed for improvement classification.
- **Review Notes**: Results are trustworthy: the run completed cleanly, reported numeric metrics, used the intended startup configuration, hit the LR and taper milestone, and the tracked code diff was limited to `train.py`.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid result with `best_test_acc=93.73%`, below the 94.11% baseline and the required 94.21% threshold.

## Unexplored Avenues
- A post-drop CutMix off switch remains the sharper version of this temporal idea, but EXP-069 makes it lower priority because a milder taper already degraded the plateau.
- A CutMix-specific label-smoothing interaction remains technically distinct, but label-smoothing deviations are a high-importance failed family and should require a stronger rationale than late-noise speculation.

## Next Steps
Move away from simple CutMix scalar or temporal-strength tuning with medium confidence. A more defensible next direction is a distinct interaction that preserves `alpha=1.0, p=0.5`, such as a minimal optimizer/refinement adjustment that does not change label-space regularization. Alternatively, leave CutMix as the current anchor and explore non-CutMix mechanisms that have not been locally bracketed.

## Exit Action Results
