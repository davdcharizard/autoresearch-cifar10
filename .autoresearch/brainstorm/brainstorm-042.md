# Brainstorm EXP-042
**Created**: 2026-06-09
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

## Web Search & Literature Review

- **mixup: Beyond Empirical Risk Minimization** (https://arxiv.org/abs/1710.09412)
  The original mixup paper reports improved generalization on CIFAR-10 and other vision datasets by training on convex combinations of examples and labels. This is relevant because the current anchor appears bounded by late generalization plateau behavior rather than throughput or model capacity alone.
- **Existing knowledge base** (`.autoresearch/knowledge/README.md`)
  Existing entries cover CIFAR regularization, cosine scheduling, width scaling, throughput, EMA, initialization, and crop padding. None records a prior mixup trial in this repo, so mixup is a distinct untested regularization lever.

## Experimental History Review

- Current baseline is `best_test_acc=93.97%` from EXP-038 / commit `755be2c`; the active goal requires at least +0.10 percentage points, so EXP-042 must reach `94.07%` to count.
- The current anchor is `STAGE_WIDTHS=(28,56,112)`, `LR=0.1`, `MOMENTUM=0.9`, `WEIGHT_DECAY=2e-4`, `LR_MILESTONES=[21000,64000]`, reflection crop padding, `label_smoothing=0.05`, FP32 channels-last compile.
- EXP-039 and EXP-041 bracket isolated weight decay around `2e-4`: `3e-4` fell to 93.55% and `1.5e-4` fell to 93.61%, so isolated decay retuning is low priority.
- EXP-040 showed higher initial LR `0.12` peaked at 93.70%; lower LR remains untested but has weaker evidence than moving to a distinct regularization mechanism.
- Repeated schedule-only second drops, smaller batches, cutout, smoothing deviations, and width increases beyond 28/56/112 are known poor next priorities.
- The late plateau in EXP-041 peaked at 93.61% and then drifted down, suggesting that stability/generalization levers remain plausible if they avoid the over-regularization seen with cutout and stronger smoothing.

## Candidate Ideas

### 1. Mild Mixup Alpha 0.1
**Summary**: Add batch-level mixup during training with `MIXUP_ALPHA = 0.1`. For each training batch, sample a mixing coefficient, permute the batch, mix images linearly, and compute `lam * CE(outputs, y_a) + (1 - lam) * CE(outputs, y_b)` while preserving `label_smoothing=0.05`.

**Reasoning**: Mixup has direct CIFAR-10 generalization evidence and is a qualitatively different regularizer from weight decay, crop padding, label smoothing, and cutout. A mild alpha is intentionally chosen to reduce the risk of fixed-budget undertraining or excessive label smoothing interaction. It should add little memory pressure, preserve evaluation semantics, and target the late generalization plateau rather than model capacity.

**Sources**: mixup paper (https://arxiv.org/abs/1710.09412); current failed-approach bracket in `.autoresearch/goal-learnings/maximize-cifar10-best-test-accuracy.md`; EXP-038 through EXP-041 index rows.

**Estimated Effort**: medium

**Risk Assessment**: Mixup may underfit within the 300s budget or interact poorly with existing label smoothing and `2e-4` weight decay. It also adds a few operations per step, so throughput must be checked. Worst case is a valid no-improvement with lower accuracy.

### 2. Initial LR 0.08 on the 2e-4 Anchor
**Summary**: Change only `LR` from `0.1` to `0.08`, leaving milestones unchanged at `[21000, 64000]`.

**Reasoning**: EXP-040 ruled out higher initial LR. A lower LR could reduce noisy high-LR updates and improve the post-drop plateau. It is very low effort and easy to interpret, but the current anchor’s first LR drop and decay settings are already well calibrated, and scalar probes have recently regressed.

**Sources**: EXP-040 report and index row; `.autoresearch/goal-learnings/maximize-cifar10-best-test-accuracy.md` LR failure entry.

**Estimated Effort**: low

**Risk Assessment**: Lower LR may reduce useful pre-drop exploration and simply undertrain. It is safe but likely has a smaller upside than a new regularization mechanism.

### 3. Bounded Late EMA Averaging
**Summary**: Add a low-frequency late-training EMA that only starts after the first LR drop and evaluates a bounded averaged model, avoiding per-step full-run EMA.

**Reasoning**: The late plateau shows noise and drift below the peak, and averaging could stabilize the final solution. However, EXP-004 showed per-step EMA overhead, while EXP-021 showed naive long equal averaging collapse, so any averaging attempt needs careful bounds and introduces more implementation risk than mixup.

**Sources**: `.autoresearch/knowledge/references/pytorch-ema-averaging.md`; EXP-004 and EXP-021 failed-approach entries.

**Estimated Effort**: medium

**Risk Assessment**: The approach could add overhead, disturb evaluation cadence, require BN recalibration, or repeat known averaging failures in a slightly different form.

## Idea Evaluation

Mild mixup has the best evidence-to-novelty ratio among the candidates. The current anchor’s strongest scalar levers are increasingly bracketed: weight decay has a local optimum at `2e-4`, higher LR failed, smoothing deviations are recurring no-improvements, and schedule-only refinement is high-importance failed space. Mixup is supported by CIFAR literature and tests a distinct mechanism: smoothing the empirical training distribution through input and label interpolation.

LR 0.08 is safer and simpler, but its expected impact is modest because the current LR/milestone setup is already validated and the higher-LR side failed clearly. Bounded EMA remains interesting for late stability but has more implementation risk and directly borders known failed averaging variants. A mild mixup alpha is the best next shot because it can plausibly improve generalization by more than the +0.10 point threshold without changing architecture, evaluation, dependencies, or resource class.

## Chosen Idea
**Selected**: Mild Mixup Alpha 0.1

**Why this idea**:
It is the most distinct remaining generalization lever with direct CIFAR evidence and manageable implementation scope. The recent scalar brackets suggest the loop should stop over-tuning weight decay/LR and test a mild data/label interpolation regularizer that may improve the post-drop plateau.

**Hypothesis**:
Adding mild mixup with `MIXUP_ALPHA = 0.1` to the current `2e-4` label-smoothed reflection anchor will improve generalization enough to reach `best_test_acc >= 94.07%` while preserving throughput, schedule behavior, parameter count, and the fixed evaluation harness.
