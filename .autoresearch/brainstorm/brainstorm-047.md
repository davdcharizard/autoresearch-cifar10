# Brainstorm EXP-047
**Created**: 2026-06-09
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

## Web Search & Literature Review

- **RandAugment: Practical Automated Data Augmentation** (`knowledge/papers/randaugment-augmentation.md`)
  Policy augmentation can improve CIFAR-style image training, and torchvision provides built-in transforms without dependency changes. EXP-044 showed the broad policy setting was not enough here, so any augmentation retry should be narrower and lower-overhead.

## Experimental History Review

- Current best remains EXP-038 at `best_test_acc=93.97%`; the active improvement threshold is 94.07% because the goal requires at least +0.10 percentage points.
- The validated anchor remains `STAGE_WIDTHS=(28, 56, 112)`, `LR=0.1`, `MOMENTUM=0.9`, `WEIGHT_DECAY=2e-4`, `LR_MILESTONES=[21000, 64000]`, reflection crop padding, label smoothing 0.05, FP32 compile, and channels-last.
- Recent isolated failures narrow the search space: weight averaging variants collapsed or lost useful steps, full-budget time-fraction cosine underperformed the 21k step schedule, isolated LR retunes weakened the anchor, and nearby weight decay brackets were worse than `2e-4`.
- Augmentation evidence is mixed but not exhausted. Erased-patch masking over-regularized, and mild RandAugment reached 93.83% but added overhead and stayed below threshold. EXP-044 explicitly left targeted color-only augmentation as the untested narrower alternative.
- The main current gap is a low-overhead perturbation that improves color/illumination invariance while preserving crop geometry, step schedule, label smoothing, and the `2e-4` weight-decay anchor.

## Candidate Ideas

### 1. Mild ColorJitter After Crop/Flip
**Summary**: Add a small `transforms.ColorJitter` after `RandomHorizontalFlip()` and before `ToTensor()`, while preserving all architecture, optimizer, schedule, smoothing, and weight-decay settings. Use conservative photometric ranges so the transform changes illumination/color statistics without geometric distortion or erasure.

**Reasoning**: This directly tests whether the harmful part of EXP-044 was broad policy augmentation rather than photometric invariance. Color-only jitter should add less CPU overhead and less semantic distortion than RandAugment, and it avoids the known failures from cutout-style masking.

**Sources**: EXP-044 report; `knowledge/papers/randaugment-augmentation.md`; cutout failures EXP-005 and EXP-009; current anchor patterns in `goal-learnings/maximize-cifar10-best-test-accuracy.md`.

**Estimated Effort**: low

**Risk Assessment**: The improvement may be too small to clear +0.10 points, or the jitter may still over-regularize the already-regularized anchor. Worst case is a valid no-improvement run with clean attribution to targeted photometric augmentation.

### 2. Hybrid Step-Hold Cosine Refinement
**Summary**: Keep LR at 0.1 until the historical 21k first-drop region, then replace the abrupt 0.01 phase with a cosine tail. This tests whether cosine can help only after preserving the validated high-LR exploration window.

**Reasoning**: EXP-046 showed full-budget cosine decays too aggressively, but a hybrid could retain the useful part of the step schedule while smoothing late refinement. This is distinct from isolated second-drop tuning because it changes the shape of the post-drop region rather than merely inserting another milestone.

**Sources**: EXP-046 report; `knowledge/papers/sgdr-cosine-schedule.md`; schedule failures EXP-003, EXP-024, EXP-030.

**Estimated Effort**: medium

**Risk Assessment**: Schedule-only experiments are a high-importance recurring failure class, so this has weaker prior odds than a new augmentation axis. It also may be sensitive to GPU contention because 21k may be reached late or not reached.

### 3. High-Capacity Budget-Matched Cosine Bundle
**Summary**: Run a broad, attribution-poor bundle inspired by the local sibling run: much larger width, projection shortcuts, stronger augmentation/regularization, and a budget-matched LR schedule.

**Reasoning**: The local sibling result suggests a retuned high-capacity recipe can beat the current range, but this contradicts many local isolated failures and would not explain which component mattered.

**Sources**: `../v2.9.6-opus-4-8/train.py`; `../v2.9.6-opus-4-8/run.log`; width failures EXP-017/019/020; BF16/TF32/cutout/projection/Nesterov failures.

**Estimated Effort**: medium

**Risk Assessment**: Highest upside but highest attribution risk. It is better reserved for when clean, targeted tests stop producing actionable candidates.

## Idea Evaluation

Mild ColorJitter has the best risk-adjusted value for EXP-047. It targets a distinct untested augmentation mechanism, preserves every validated anchor setting, and directly follows the unexplored avenue from EXP-044 and EXP-046. Its failure mode is clean: if it underperforms, targeted photometric perturbation can be deprioritized without disturbing optimizer or architecture conclusions.

Hybrid step-hold cosine is technically plausible but fights the strongest recent evidence: schedule-only changes have repeatedly underperformed, and EXP-046 just showed that cosine needs careful handling. It should wait until non-schedule, low-overhead levers are exhausted.

The high-capacity bundle remains a useful future escape hatch, but it is too broad for the immediate next loop because local learnings already warn against most of its components in isolation.

## Chosen Idea
**Selected**: Mild ColorJitter After Crop/Flip

**Why this idea**:
It is the cleanest remaining augmentation test after RandAugment and cutout failures. It keeps the validated step-schedule and regularization anchor intact while probing a narrower photometric-invariance mechanism.

**Hypothesis**:
A conservative color-only jitter will improve CIFAR-10 generalization enough to reach at least 94.07% `best_test_acc`, because it adds mild photometric diversity without the geometric distortion, erasure, or heavier policy overhead that hurt prior augmentation attempts.
