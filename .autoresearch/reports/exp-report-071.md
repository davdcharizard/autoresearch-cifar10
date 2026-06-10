# Report EXP-071: CIFAR AutoAugment on CutMix Anchor
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-071.md
- **Plan**: plans/plan-071.md
- **Log**: logs/exp-log-071.md

## Goal
Maximize CIFAR-10 `best_test_acc` under the fixed benchmark harness, with higher better. The active baseline before EXP-071 was EXP-064 at 94.11%, and the goal's noise guard required at least 94.21% to count as an improvement.

## Idea & Hypothesis
The chosen idea was to add torchvision's CIFAR AutoAugment policy to the training transform while preserving the full EXP-064 CutMix anchor. The hypothesis was that a learned CIFAR-specific policy might provide complementary invariance beyond reflection crop, flip, label smoothing, stronger decay, and CutMix.

## Approach
`train.py` was changed only in the training transform and startup logging. `transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10)` was inserted after `RandomHorizontalFlip()` and before `ToTensor()`, keeping AutoAugment in the PIL-image part of the pipeline. Unit-std normalization, CutMix `alpha=1.0` / `p=0.5`, endpoint smoothing 0.05, architecture, optimizer, LR milestones, compile/channels-last, and validation cadence were unchanged.

## Execution
One local foreground run was executed on GPU1 with stdout/stderr captured to `run.log`. Startup markers confirmed CUDA, ResNet-20 with 822,790 parameters, unchanged CutMix settings, active CIFAR AutoAugment, the 300s training budget, and 390 batches per epoch. The run reached the step-21000 LR drop at epoch 54 and completed cleanly with final metrics; no traceback, CUDA OOM, non-finite, `nan`, or `inf` markers were present.

## Results
- **Primary metric**: 93.62% (baseline: 94.11%, delta: -0.49 pp, -0.52%)
- **Observations**: Pre-drop best was 87.50%, lower than the 87.97% pre-drop best from EXP-064. Post-drop accuracy climbed to 93.58% by epoch 72, stayed on a lower plateau, briefly reached 93.62% at epoch 101, and ended at 93.35%.
- **Analysis**: The hypothesis is rejected. CIFAR AutoAugment did not add useful complementary invariance to the CutMix anchor; it underperformed the baseline while slightly reducing step coverage to 39,585 steps and increasing total wall time to 457.7s.
- **Key Learning**: CIFAR AutoAugment stacked with CutMix peaked at 93.62%, below baseline, so policy augmentation over-regularizes the current anchor.

## Verification
- **Conditions**: All hard/process conditions passed; the improvement threshold failed.
- **Review Notes**: Results are trustworthy. The run completed cleanly, reported numeric metrics, reached the LR drop, stayed within the time cap, preserved the fixed harness, and modified only `train.py`.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid completed run with `best_test_acc=93.62%`, below both the 94.11% baseline and the 94.21% improvement threshold.

## Unexplored Avenues
- Weaker or targeted sub-policies could reduce over-regularization, but EXP-044 and EXP-071 now make isolated policy augmentation low priority.
- AutoAugment without CutMix would isolate the policy itself, but removing the current best mechanism is unlikely to beat the baseline and would mainly answer a lower-value interaction question.

## Next Steps
- High confidence: deprioritize isolated policy augmentation on this goal.
- Medium confidence: test fan-out Kaiming conv initialization as a narrow non-augmentation lever.
- Medium confidence: test early CutMix warmup only if the plan clearly preserves static `p=0.5` after warmup and treats EXP-069 as a caution against post-drop weakening.

## Exit Action Results
