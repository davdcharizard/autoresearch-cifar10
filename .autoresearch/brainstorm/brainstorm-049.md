# Brainstorm EXP-049
**Created**: 2026-06-09
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

## Web Search & Literature Review

- **Existing PyTorch throughput context** (`knowledge/references/pytorch-throughput-tools.md`)
  The current anchor should keep the validated FP32 compile/channels-last path. Candidate ideas should avoid reduced precision and avoid extra runtime overhead unless the expected accuracy gain is large.
- **Existing CIFAR architecture context** (`knowledge/papers/wide-residual-networks.md`)
  Wider residual networks can improve CIFAR accuracy, but local experiments show this repo's `28/56/112` width is already the best calibrated capacity point under the fixed budget.
- **Existing ResNet initialization context** (`knowledge/references/resnet-zero-init-residual.md`)
  Residual-branch BN initialization is a known lever, but EXP-028's full zero-gamma trial created strong local negative evidence for aggressive identity-bias initialization.

No new external search was needed for EXP-049. The most relevant constraints are now local: 49 prior experiments have bracketed many generic CIFAR levers, and the next useful ideas should be low-overhead modifications to the validated anchor.

## Experimental History Review

- Current best remains EXP-038 at `best_test_acc=93.97%`; the active threshold is 94.07% because improvements must clear +0.10 percentage points.
- The validated anchor is `STAGE_WIDTHS=(28, 56, 112)`, batch size 128, LR 0.1, momentum 0.9, weight decay 2e-4, first LR drop at step 21000, reflection crop padding, label smoothing 0.05, FP32 compile, and channels-last.
- Recent no-improvements narrowed nearby knobs: `WEIGHT_DECAY=3e-4` over-regularized, `1.5e-4` underperformed, LR 0.12 and 0.08 weakened the anchor, lower BN momentum lagged the baseline, and time-budget cosine was substantially worse.
- Strong recurring negative evidence rules out isolated second LR drops, EMA/SWA-style weight averaging, width above `28/56/112`, smaller batch sizes, cutout masking, and simple smoothing deviations.
- EXP-047 showed that missed LR drops confound step-schedule comparisons, but EXP-048 had clean GPU conditions and a valid first LR drop. The next experiment should continue to check step 21000 explicitly.
- The most promising remaining space is optimizer dynamics that preserves the current anchor but changes how the successful regularization acts during momentum SGD.

## Candidate Ideas

### 1. Decoupled SGD Weight Decay at 2e-4
**Summary**: Keep SGD momentum, LR schedule, model, augmentation, label smoothing, and the numeric decay magnitude `2e-4`, but remove `weight_decay` from `optim.SGD` and apply decoupled multiplicative weight decay manually after each optimizer step.

**Reasoning**: EXP-038 proved that stronger shrinkage helps the label-smoothed reflection anchor, while EXP-039 and EXP-041 showed nearby coupled L2 magnitudes are worse. Decoupling tests a different mechanism: preserve the successful shrinkage magnitude while preventing the decay term from flowing through the gradient and momentum-buffer path. This is low overhead, modifies only `train.py`, and should keep the same step budget and first LR drop behavior.

**Sources**: `train.py` optimizer path; EXP-038/039/041 reports; goal-learnings pattern that `2e-4` is the current regularization anchor.

**Estimated Effort**: medium

**Risk Assessment**: The main risk is semantic miscalibration: decoupled `2e-4` may be weaker or stronger than coupled SGD L2 over this short horizon. The failure mode should still be a clean no-improvement, and the implementation can be tightly scoped.

### 2. Partial Residual-Branch BN Scale Initialization
**Summary**: Initialize each residual block's final BatchNorm scale to a small positive value such as `0.1` instead of zeroing it fully.

**Reasoning**: Full zero-gamma initialization in EXP-028 undertrained badly, but a partial scale could keep some early residual learning while gently biasing blocks toward stable residual updates. It is a no-parameter, low-overhead initialization tweak.

**Sources**: `knowledge/references/resnet-zero-init-residual.md`; EXP-028 report; `BasicBlock.bn2` in `train.py`.

**Estimated Effort**: low

**Risk Assessment**: The local negative prior is strong. Even partial scaling may still slow representation learning under the fixed budget, so this should not outrank optimizer-dynamics tests with cleaner positive evidence.

### 3. Clean Mild ColorJitter Retry
**Summary**: Retry the mild ColorJitter augmentation from EXP-047 under clean GPU conditions so the first LR drop is reached.

**Reasoning**: EXP-047 was not a clean attribution against photometric augmentation because severe contention caused the run to miss the 21k LR drop. A clean retry could determine whether mild color perturbation is actually useful on the current anchor.

**Sources**: EXP-047 report and protocol finding; current `train_tf` augmentation pipeline in `train.py`.

**Estimated Effort**: low

**Risk Assessment**: This is scientifically reasonable but lower priority: augmentation has already been weak locally, and EXP-044 RandAugment reached only 93.83% despite a valid run. A retry may spend another cycle on a low-probability family.

## Idea Evaluation

Decoupled SGD weight decay has the best risk-adjusted value for EXP-049. It directly builds on the strongest recent success, EXP-038, without retrying the failed nearby scalar values. The mechanism is clear: change how shrinkage interacts with SGD momentum while preserving the regularization magnitude that currently anchors the best result.

Partial residual BN scale initialization is cheap, but EXP-028 makes the family suspect. A nonzero scale may avoid the worst undertraining, yet the expected impact is uncertain and the mechanism is less tied to the latest successful recipe.

A clean ColorJitter retry would resolve an attribution gap left by EXP-047, but it is less promising than optimizer dynamics. EXP-044's valid RandAugment result suggests isolated augmentation is unlikely to clear the 94.07% threshold unless the policy is more targeted.

## Chosen Idea
**Selected**: Decoupled SGD Weight Decay at 2e-4

**Why this idea**:
It is the most targeted test of the current regularization bottleneck. The numeric weight-decay anchor is validated, nearby coupled magnitudes failed, and decoupling changes the optimizer dynamics without changing model size, data pipeline, evaluation, or runtime-heavy components.

**Hypothesis**:
Applying `WEIGHT_DECAY=2e-4` as decoupled multiplicative shrinkage after each SGD step will preserve the useful regularization found in EXP-038 while reducing momentum-buffer coupling, improving `best_test_acc` to at least 94.07% under the fixed CIFAR-10 harness.
