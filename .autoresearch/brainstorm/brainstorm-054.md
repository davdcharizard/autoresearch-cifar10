# Brainstorm EXP-054
**Created**: 2026-06-09
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

## Web Search & Literature Review

- **Existing knowledge base** (`knowledge/README.md`)
  Current entries cover cutout, mixup, RandAugment, cosine schedules, Wide ResNets, throughput tools, EMA, residual initialization, and crop padding. The most relevant existing open mechanism is mixup, but EXP-042 crashed before a final metric and carries throughput risk.
- **Deep Networks with Stochastic Depth** (`knowledge/papers/stochastic-depth-resnets.md`; source: https://arxiv.org/abs/1603.09382)
  Stochastic depth randomly bypasses residual branches during training and uses the full residual network at evaluation time. It is directly compatible with residual blocks, changes no evaluation harness code, and can regularize co-adaptation without changing parameter count.
- **DropBlock: A regularization method for convolutional networks** (source: https://arxiv.org/abs/1810.12890)
  DropBlock is structured feature dropout for convolutional networks. It is relevant as another train-time regularizer, but the local history already shows isolated cutout and photometric augmentation underperform, and DropBlock is implementation-heavier than stochastic depth.

## Experimental History Review

- Current best remains EXP-038 at `best_test_acc=93.97%`; EXP-054 must reach at least `94.07%` to count as an improvement.
- The current anchor is `STAGE_WIDTHS=(28, 56, 112)`, `BATCH_SIZE=128`, `LR=0.1`, `MOMENTUM=0.9`, `WEIGHT_DECAY=2e-4`, `LR_MILESTONES=[21000, 64000]`, reflection crop padding, label smoothing 0.05, FP32 compile, channels-last, and once-per-epoch validation.
- EXP-053 closed the isolated batch-size bracket: smaller batches 96/112 and larger batch 160 all lag the batch-128 anchor, so batch size should not be retuned alone.
- High-importance failed spaces now include schedule-only second LR drops, weight averaging variants, batch-size deviations, and width increases beyond 28/56/112.
- Medium recurring failures include cosine schedules, residual-branch BN down-scaling, scalar LR deviations, cutout, and label-smoothing deviations.
- A remaining untested category is train-time residual regularization that leaves the evaluation graph and parameter count unchanged. This differs from failed residual initialization because it does not force residual branches near identity at initialization; it only adds mild stochastic branch omission during training.

## Candidate Ideas

### 1. Very Mild Residual Stochastic Depth
**Summary**: Add training-only stochastic depth inside `BasicBlock`, with a very small per-block drop probability that increases linearly across the residual stack and is disabled during evaluation.

**Reasoning**: The current anchor may be limited by late generalization rather than capacity or throughput. Stochastic depth is designed for residual networks and regularizes branch co-adaptation while preserving the full network at test time. A conservative maximum drop probability, such as 0.03, should avoid the undertraining seen in residual BN down-scaling while adding little compute overhead and no parameter change.

**Sources**: `knowledge/papers/stochastic-depth-resnets.md`; residual failure entries EXP-028 and EXP-051; current anchor row EXP-038; failed batch/schedule brackets through EXP-053.

**Estimated Effort**: medium

**Risk Assessment**: The model is shallow, so even mild block dropping can undertrain or add compile/runtime overhead. If applied too strongly it may resemble the residual identity-bias failures; the plan must keep the probability very low and verify the first LR drop.

### 2. Reliable Mild Mixup Retry
**Summary**: Retry mild mixup with a foreground attached run now that full foreground execution is proven, preserving the EXP-042 concept but treating prior failures as run-control artifacts.

**Reasoning**: Mixup has direct CIFAR generalization support and EXP-042 never produced a final metric, so it remains scientifically unmeasured. It targets image/label interpolation rather than another local hyperparameter bracket and could improve the late plateau if the run survives to the post-drop phase.

**Sources**: `knowledge/papers/mixup-beyond-erm.md`; `reports/exp-report-042.md`; full successful foreground runs EXP-052 and EXP-053.

**Estimated Effort**: medium

**Risk Assessment**: Prior partial logs showed step timing degraded enough that the 21k LR drop was barely reachable. Even with a reliable foreground session, mixup may miss or barely reach the first LR drop, making attribution weaker.

### 3. Tiny Late-Stage DropBlock
**Summary**: Add a very small train-time DropBlock-style feature regularizer after the final residual stage only, disabled during evaluation.

**Reasoning**: DropBlock targets spatially correlated convolutional activations more directly than standard dropout. Restricting it to the final stage could regularize high-level features without changing input augmentation or parameter count.

**Sources**: DropBlock paper (https://arxiv.org/abs/1810.12890); cutout and ColorJitter failure rows EXP-005, EXP-009, EXP-050.

**Estimated Effort**: medium

**Risk Assessment**: Local augmentation-like regularizers have repeatedly underperformed, and a custom DropBlock implementation risks overhead or shape bugs. This is less clean than stochastic depth because it adds spatial masking logic not already present in the model.

## Idea Evaluation

Very mild residual stochastic depth has the best combination of mechanism clarity, distinctness, and feasibility. It targets the residual model itself through a known residual-network regularization mechanism while preserving inference-time parameters and the fixed evaluation harness. It also avoids the now weak spaces: batch size, scalar LR, weight decay brackets, second drops, cosine schedules, and width increases.

Reliable mixup remains worthwhile because EXP-042 was a crash rather than a negative result, but it is operationally riskier. The prior partial logs suggest throughput degradation that could threaten first-drop reachability, while stochastic depth should preserve most of the anchor's step budget. Tiny DropBlock has external support, but it overlaps more with failed spatial regularizers and requires more custom masking code.

The lead candidate should therefore be stochastic depth, implemented conservatively enough that a no-improvement remains interpretable rather than a predictable undertraining failure.

## Chosen Idea
**Selected**: Very Mild Residual Stochastic Depth

**Why this idea**:
It is the clearest untested train-time regularization mechanism left around the current anchor. It has literature support for residual networks, modifies only `train.py`, preserves evaluation behavior, and avoids the recurring failed scalar, schedule, width, batch-size, and augmentation brackets.

**Hypothesis**:
Adding very mild training-only residual stochastic depth with a maximum block drop probability around 0.03 will improve generalization enough to raise `best_test_acc` to at least `94.07%` while preserving the step-21000 LR drop and unchanged `num_params=822,790`.
