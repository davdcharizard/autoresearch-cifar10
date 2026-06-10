# Brainstorm EXP-052
**Created**: 2026-06-09
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

## Web Search & Literature Review

- **SGDR cosine schedule note** (`knowledge/papers/sgdr-cosine-schedule.md`)
  Cosine annealing is relevant to CIFAR SGD scheduling, but EXP-046 showed full-budget time-fraction cosine underperforms locally; any retry must preserve the validated 21k high-LR window.
- **Wide residual network note** (`knowledge/papers/wide-residual-networks.md`)
  Capacity scaling can work for CIFAR, but local experiments already found the 28/56/112 anchor and showed larger width variants fail under the fixed recipe.
- **RandAugment note** (`knowledge/papers/randaugment-augmentation.md`)
  Policy augmentation is supported by torchvision, but EXP-044 and EXP-050 show isolated augmentation additions have not cleared the current 94.07% threshold.

No new external search was needed. Existing knowledge plus the 53-experiment local trajectory gives the strongest guidance for the next step.

## Experimental History Review

- Current best remains EXP-038 at `best_test_acc=93.97%`; EXP-052 must reach at least `94.07%` to count as an improvement.
- The current anchor is `STAGE_WIDTHS=(28, 56, 112)`, batch size 128, LR 0.1, momentum 0.9, weight decay 2e-4, first LR drop at step 21000, reflection crop padding, label smoothing 0.05, FP32 compile, and channels-last.
- Recent clean failures narrow the space: EXP-048 lower BN momentum, EXP-049 decoupled weight decay, EXP-050 ColorJitter, and EXP-051 partial residual BN scaling all completed cleanly below threshold.
- High-importance failures warn against isolated second LR drops and weight averaging. Medium-importance failures now include residual-branch BN down-scaling, scalar initial-LR deviations, smaller batches, and label-smoothing deviations.
- The most defensible remaining gap is a schedule shape that is not an abrupt second drop and does not disturb the proven first 21k-step high-LR phase. A gentle post-drop tail could address late plateau drift while preserving the anchor's early learning.

## Candidate Ideas

### 1. Hybrid Post-Drop Cosine Tail to a Nonzero Floor
**Summary**: Preserve LR 0.1 until the validated step-21000 first drop, then replace the flat LR 0.01 tail with a smooth cosine decay from 0.01 toward a small nonzero floor such as 0.002 over the expected remaining step budget.

**Reasoning**: Full time-fraction cosine failed because it changed the whole schedule, while abrupt second drops have repeatedly failed because they reduce exploration too sharply. A tail-only cosine keeps the known-good high-LR phase and first drop intact, but gently reduces late update size during the plateau where EXP-051 and other runs drift around 93.1-93.6%.

**Sources**: `knowledge/papers/sgdr-cosine-schedule.md`; EXP-046; goal-learnings `Schedule-only second LR drops fail`; EXP-051 late plateau observations.

**Estimated Effort**: medium

**Risk Assessment**: Schedule-only work has a negative local prior. If the nonzero floor is too low, the run may behave like a failed second-drop variant; if too high, it may match the flat 0.01 tail and produce no gain.

### 2. Very Mild Residual Drop-Path Regularization
**Summary**: Add a tiny stochastic-depth style mask to residual branches during training only, with a small maximum drop probability, leaving evaluation deterministic and preserving architecture shape.

**Reasoning**: External image regularizers have mostly failed, but an internal residual regularizer could reduce co-adaptation without changing data transforms or label smoothing. It directly targets generalization while keeping parameter count unchanged.

**Sources**: `train.py` `BasicBlock`; goal-learnings failed augmentation and cutout entries; EXP-051 showing residual initialization is trainable but not enough.

**Estimated Effort**: medium

**Risk Assessment**: This may reduce effective residual capacity and hurt the already tight fixed-budget convergence. It also adds a small per-forward operation and has less direct local support than a schedule-tail test.

### 3. Larger Batch Size 160 with Current Schedule
**Summary**: Increase `BATCH_SIZE` from 128 to 160 while preserving the optimizer, LR milestones, model, augmentation, and label smoothing.

**Reasoning**: Smaller batches have failed by losing useful coverage, but a modest larger batch may improve image throughput and gradient stability while still reaching the 21k LR drop. It is a simple way to test whether the current anchor is update-limited or image-coverage-limited.

**Sources**: EXP-025 and EXP-036 smaller-batch failures; `train.py` batch-size and step-schedule structure; goal-learnings smaller-batch entry.

**Estimated Effort**: low

**Risk Assessment**: Larger batches may reduce update count or change optimization noise unfavorably. If step time rises too much, it could weaken the first-drop/tail coverage and become a confounded no-improvement.

## Idea Evaluation

The hybrid post-drop cosine tail has the clearest mechanism for the current trajectory: most clean anchor-like runs reach the first drop and then plateau below threshold, so a smoother late tail can target overfitting or noisy late updates without disturbing the validated early schedule. Its main weakness is the negative prior on schedule-only changes, but it is distinct from both full-budget cosine and abrupt second drops.

Very mild residual drop-path is a reasonable future test, but it has weaker evidence and a more direct convergence risk. The latest EXP-051 result makes residual-branch dynamics look sensitive under this budget, so another residual-branch intervention should wait.

Batch size 160 is cheap and plausible, yet its mechanism is ambiguous: the goal needs higher accuracy, and the current anchor already has adequate throughput and reaches the LR drop. A batch-size experiment is best saved for a deliberate image-coverage-vs-update-count bracket.

## Chosen Idea
**Selected**: Hybrid Post-Drop Cosine Tail to a Nonzero Floor

**Why this idea**:
It tests the most specific remaining schedule gap: keep the successful high-LR phase and first drop, but replace the long flat LR 0.01 plateau with a gentle tail that may reduce late drift. The nonzero floor distinguishes it from failed abrupt second-drop experiments and avoids the full-budget cosine change that underperformed in EXP-046.

**Hypothesis**:
A cosine tail from LR 0.01 toward roughly 0.002 after step 21000 will improve late post-drop refinement while preserving early convergence, raising `best_test_acc` to at least 94.07%.
