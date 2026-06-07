# Brainstorm EXP-020
**Created**: 2026-05-29
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

- Prior sources sufficient. Airbench uses 6-view TTA with spatial shifts (https://arxiv.org/html/2404.00498v2). EXP-016 validated hflip TTA (+0.66%).

## Experimental History Review

- **21 experiments** (BASE through EXP-019), baseline 96.39%
- **EXP-019 key learning**: CosineAnnealingLR is periodic — channels_last's extra epochs cause LR restart. Channels_last cannot be used without fixing the LR schedule.
- **EXP-016**: hflip TTA gave +0.66%. The model benefits significantly from prediction averaging.
- **EXP-019 confound**: 6-view TTA was tested WITH channels_last (LR restart). Cannot determine TTA effect. Must test TTA alone.
- **Critical constraint**: NO training changes. Only modify eval-mode forward(). This avoids all training-related confounds (LR schedule, epochs, convergence).

## Candidate Ideas

### 1. Extended TTA (Spatial Shifts) — Eval-Only Change
**Summary**: Add ±1px spatial shift TTA to the eval-mode forward() only. No training changes at all. This isolates the TTA effect on the exact baseline model. 6 views: original, hflip, shift-left, shift-right, shift-up, shift-down. All averaged equally.

**Reasoning**: EXP-016's hflip TTA gave +0.66% on top of 95.73% training accuracy. The model was trained with RandomCrop(32, padding=4), so it's invariant to small translations. Spatial shifts at eval time exploit this learned invariance. Since eval time doesn't count against the 300s budget, this is truly free. Tested on airbench with success.

**Sources**: EXP-016, airbench, EXP-019 (need to isolate TTA from channels_last confound)

**Estimated Effort**: low — ~8 lines in forward()

**Risk Assessment**: Extremely low. Zero training changes. Cannot affect training accuracy. Worst case: no improvement over hflip-only TTA.

### 2. Weighted TTA (Higher Weight on Original + Hflip)
**Summary**: Same 6 views but weight original and hflip higher (0.25 each) and shifts lower (0.125 each). The original and hflip are higher-quality views; shifts may introduce minor artifacts from padding.

**Reasoning**: The original and horizontally-flipped views are the most natural transformations the model has seen during training. Spatial shifts are valid but may introduce slight edge artifacts from reflect padding. Weighting accordingly respects this hierarchy.

**Sources**: Common practice in ensemble methods

**Estimated Effort**: low — same as Idea 1 with weighted sum

**Risk Assessment**: Very low. But adds a hyperparameter (weights) that may need tuning.

## Idea Evaluation

Idea 1 (equal-weight 6-view TTA) is the simpler and cleaner approach. It matches the airbench pattern exactly. Idea 2 adds unnecessary complexity with the weighting hyperparameter. If the equal-weight version doesn't work, we can try weighting in a follow-up.

## Chosen Idea
**Selected**: Extended TTA (Spatial Shifts) — Eval-Only Change

**Why this idea**: Cleanest possible test of spatial-shift TTA. Zero training changes means zero risk and zero confound with LR schedule. Directly isolates whether adding spatial-shift views to the existing hflip TTA improves accuracy.

**Hypothesis**: Adding 4 spatial-shift views to the existing hflip TTA will reduce prediction variance further, improving best_test_acc from 96.39% to ~96.5-96.6%.
