# Brainstorm EXP-057
**Created**: 2026-06-09
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

## Web Search & Literature Review

- **Existing knowledge base** (`knowledge/README.md`)
  Existing entries cover CIFAR augmentation, mixup, cosine schedules, wide residual networks, stochastic depth, EMA, residual initialization, and PyTorch throughput. No new external source was needed for EXP-057 because the decision is now dominated by 56 local fixed-harness experiments and the active anchor's narrow remaining gaps.

## Experimental History Review

- Current best remains EXP-038 at `best_test_acc=93.97%`; because the goal now requires at least +0.10 percentage points, EXP-057 must reach `94.07%` to count as an improvement.
- The current anchor is `STAGE_WIDTHS=(28, 56, 112)`, `BATCH_SIZE=128`, `LR=0.1`, `MOMENTUM=0.9`, `WEIGHT_DECAY=2e-4`, `LR_MILESTONES=[21000, 64000]`, reflection crop padding, label smoothing 0.05, FP32 compile, channels-last, and once-per-epoch validation.
- EXP-056 showed that excluding BatchNorm/bias from decay peaks at 93.68%, so the anchor should keep simple global coupled SGD L2 decay.
- Direct regularization and augmentation have weakened recently: very mild stochastic depth peaked at 93.40%, mixup alpha 0.1 peaked at 93.85%, mild RandAugment peaked at 93.83%, and clean ColorJitter peaked at 93.49%.
- Label smoothing remains one of the few positive interventions: EXP-032 improved the reflection anchor with smoothing 0.05, while static smoothing deviations to 0.03 and 0.08 were near but below the improvement threshold. That suggests the smoothing axis is sensitive but not exhausted if changed dynamically rather than globally.
- Late training often plateaus or drifts after the first LR drop, including EXP-056 peaking at epoch 63 and ending lower. A post-drop-only change can preserve the validated high-LR phase while testing whether late refinement benefits from sharper targets.

## Candidate Ideas

### 1. Post-Drop Label Smoothing Anneal
**Summary**: Keep `label_smoothing=0.05` before the step-21000 first LR drop, then lower smoothing to `0.0` after the LR drops to 0.01. Preserve all other anchor settings.

**Reasoning**: Static smoothing 0.05 is validated by EXP-032 and remains in the active anchor, but static deviations around it stayed inside the noise band. A schedule tests a different mechanism: use smoothing during high-LR representation learning, then remove it during low-LR refinement so the classifier can sharpen on true labels. This directly targets the common post-drop plateau/drift without touching architecture, data pipeline, validation cadence, or throughput-sensitive components.

**Sources**: EXP-032 label-smoothing improvement; EXP-033 and EXP-037 smoothing deviations; EXP-056 late plateau; `goal-learnings/maximize-cifar10-best-test-accuracy.md` patterns and failed approaches; current `train.py` loss call.

**Estimated Effort**: low

**Risk Assessment**: Removing smoothing after the first drop may overfit or increase late instability, and static lower smoothing did not clear the threshold. The failure mode should be a valid no-improvement, not a crash, because the code change is a scalar loss parameter branch.

### 2. Tiny Final Classifier Dropout
**Summary**: Add `F.dropout(out, p=0.05, training=self.training)` after global average pooling and before `self.fc` in `ResNet.forward`, preserving every other anchor setting.

**Reasoning**: This targets classifier co-adaptation rather than input augmentation or residual branch dynamics. It is cheap and should preserve model size, step count, and the 21k LR drop.

**Sources**: Recent failed isolated regularization entries EXP-054 and EXP-055; current `train.py` classifier head.

**Estimated Effort**: low

**Risk Assessment**: Recent isolated regularizers have underperformed, and the anchor already has label smoothing plus strong weight decay. Dropout may soften final fitting in the same way mixup and stochastic depth did.

### 3. Mixup Without Label Smoothing
**Summary**: Revisit mild mixup alpha 0.1 but remove label smoothing from the mixup loss so target interpolation is not compounded with softened hard labels.

**Reasoning**: EXP-055 completed cleanly and reached 93.85%, which is below threshold but closer than many recent failures. The result may reflect over-regularization from combining mixup with label smoothing. Removing smoothing only for mixup is a coupled balance test rather than a direct retry.

**Sources**: EXP-055 mixup retry; `knowledge/papers/mixup-beyond-erm.md`; label-smoothing experiments EXP-032, EXP-033, EXP-037.

**Estimated Effort**: medium

**Risk Assessment**: This is still close to a recent no-improvement and adds batch-level target mixing overhead. It may again finish below threshold despite a coherent regularization-balance story.

## Idea Evaluation

Post-Drop Label Smoothing Anneal has the strongest mechanism for the current trajectory. It preserves the validated smoothing during the high-LR phase, avoids the recurring failures around isolated augmentation and decay semantics, and targets the observed late plateau directly. It is also a small `train.py`-only change with little throughput risk.

Tiny Final Classifier Dropout is simple but poorly supported by the recent trajectory. It is another isolated regularizer after multiple regularizers have failed below the anchor.

Mixup Without Label Smoothing has a plausible interaction story and a relatively high recent score, but it immediately revisits a just-failed family and carries more implementation and throughput risk than a scalar smoothing schedule. It remains a backup if late smoothing annealing fails.

The lead candidate is therefore a dynamic loss-sharpening experiment: keep the current anchor unchanged until the first LR drop, then switch label smoothing off for the low-LR refinement phase.

## Chosen Idea
**Selected**: Post-Drop Label Smoothing Anneal

**Why this idea**:
It is the most targeted remaining low-risk probe: it preserves the validated label-smoothing anchor during high-LR training while testing whether late low-LR refinement is being held back by overly soft targets. Prior static smoothing deviations show this axis matters but do not rule out a phase-specific schedule.

**Hypothesis**:
Using `label_smoothing=0.05` before step 21000 and `label_smoothing=0.0` afterward will improve late refinement enough to reach at least `94.07%`, while preserving throughput, the 21k LR drop, and the rest of the current anchor.
