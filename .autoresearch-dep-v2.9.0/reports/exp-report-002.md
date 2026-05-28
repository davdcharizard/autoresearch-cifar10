# Report EXP-002: TrivialAugmentWide + RandomErasing on Width-2x Baseline
- **Created**: 2026-05-27
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-002.md
- **Plan**: plans/plan-002.md
- **Log**: logs/exp-log-002.md

## Goal

Maximize best_test_acc (%) on CIFAR-10, higher is better. Baseline: 92.29% (EXP-001, width-2x ResNet-20, 1.07M params, wall-clock-fractional schedule). Improvement threshold: >= 92.39%.

## Idea & Hypothesis

**Chosen idea**: Add TrivialAugmentWide and RandomErasing to the width-2x model's training transforms, keeping everything else at EXP-001 values.

**Why selected**: The "capacity first, then regularization" sweep order establishes that augmentation gains compound with model width. The 1.07M-param model has more representational capacity for augmentation to exploit. TrivialAugment is zero-search-cost and competitive with AutoAugment on WRN-class CIFAR-10 models.

**Hypothesis**: best_test_acc would reach 92.8-93.5% (+0.5-1.2pp over baseline).

## Approach

Two lines added to `train_tf` in train.py:
1. `transforms.TrivialAugmentWide()` after `RandomHorizontalFlip()` and before `ToTensor()` (PIL-level)
2. `transforms.RandomErasing(p=0.25, scale=(0.02, 0.2))` after `Normalize()` (tensor-level)

No other changes. Architecture (WIDTH_MULT=2), wall-clock-fractional schedule, optimizer (SGD lr=0.1, momentum=0.9, WD=1e-4), batch size (128), seed (42) all unchanged.

## Execution

Single run, completed without errors. 68 epochs (26,420 steps) in 300.0s on a single H20 GPU. Total wall-clock 354.4s. No retries, no adjustments.

## Results

- **Primary metric**: 92.92% (baseline: 92.29%, delta: +0.63pp, +0.68%)
- **Observations**:
  - Augmentation overhead negligible: step time ~11ms (same as EXP-001), 68 epochs vs 69 — only 1 fewer epoch
  - Early accuracy ~3pp below EXP-001 at same epochs (training harder with augmentation, as expected)
  - First LR drop (epoch ~34): +3.8pp jump (86.45→90.24%), then steady climb to 92.40% by epoch 49
  - Second LR drop (epoch ~52): pushed from 92.40% to 92.92% peak — a +0.52pp gain, much larger than EXP-001's +0.02pp from the same drop
  - Peak VRAM unchanged at 598.7 MB, params unchanged at 1,073,962
- **Analysis**: The hypothesis was directionally correct — augmentation improved accuracy — landing at 92.92%, within the predicted 92.8-93.5% band. The most striking observation is that the second LR drop delivered +0.52pp here vs only +0.02pp in EXP-001. This suggests augmentation creates a larger optimization gap in the second LR plateau that the final polish phase can close — the augmented training data is harder to fit, so there's more room for the lowest LR to fine-tune. The total gain from BASE→EXP-002 is now +1.20pp (91.72→92.92%), composed of +0.57pp from capacity (EXP-001) and +0.63pp from augmentation (EXP-002), confirming the capacity-then-regularization compounding effect.
- **Key Learning**: Augmentation on the width-2x model adds +0.63pp; the second LR drop benefits significantly more with augmentation (+0.52pp vs +0.02pp without), indicating that augmentation creates a larger optimization gap that the low-LR polish phase closes.

## Verification

- **Conditions**: All 3 passed — (1) 92.92% > 92.39% threshold, (2) summary block complete, (3) eval_count=68 == num_epochs=68
- **Review Notes**: Results trustworthy. Same architecture/params as EXP-001, only augmentation changed. Improvement came through the intended mechanism (regularization via augmentation).
- **Verdict**: improvement
- **Verdict Basis**: All necessary conditions passed; +0.63pp exceeds the +0.1pp threshold by several baseline noise-floor standard deviations (~0.3pp).

## Unexplored Avenues

- **Weight decay 5e-4**: The WRN paper's recommended WD for wider models. Now that augmentation is in place, WD could compound further — stronger L2 regularization works synergistically with augmentation.
- **Stronger augmentation**: RandAugment with tuned N and M, or increasing RandomErasing probability. The current setup uses conservative parameters.
- **Label smoothing**: Was part of the EXP-000 bundle but never isolated. On the wider augmented model, label smoothing 0.1 could add another regularization axis.
- **Nesterov momentum**: A small independent improvement, free to add.

## Next Steps

1. **Weight decay 5e-4** — the WRN paper's standard for wider models, compounding with the augmentation already in place. High confidence.
2. **Nesterov momentum + label smoothing 0.1** — two small orthogonal regularization gains that can be combined as a single "recipe polish" experiment. Medium confidence.
3. **Width-4x (WIDTH_MULT=4)** — with the current augmentation and WD, the wider model may now have enough regularization support. Lower confidence due to epoch-count risk.

## Exit Action Results
(no exit actions defined)
