# Report EXP-000: Modern Training Recipe (Cosine LR + CutOut + Label Smoothing)
- **Created**: 2026-05-28
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-000.md
- **Plan**: plans/plan-000.md
- **Log**: logs/exp-log-000.md

## Goal

Maximize CIFAR-10 test accuracy (best_test_acc, %, higher is better) by modernizing the ResNet-20 training recipe. Baseline: 91.81% (commit 34e8ab2).

## Idea & Hypothesis

Replace the misaligned step LR schedule with cosine annealing + warmup, add CutOut augmentation (16px), and add label smoothing (0.1). These are individually proven techniques on CIFAR-10 with small ResNets. Hypothesis: combined effect would push accuracy to 93-94%, primarily from proper LR scheduling and CutOut regularization.

## Approach

Changes to train.py only (ResNet-20 architecture unchanged):
1. Replaced MultiStepLR (milestones [32000, 48000]) with SequentialLR: LinearLR warmup (5 epochs, start_factor=0.1) + CosineAnnealingLR (T_max=90). Scheduler steps per-epoch instead of per-step.
2. Added CutOut class masking 16x16 pixel patches, applied after Normalize in training transforms.
3. Replaced F.cross_entropy with nn.CrossEntropyLoss(label_smoothing=0.1).
4. Removed MAX_STEPS=64000 cap; TIME_BUDGET_S is sole termination condition.

Key deviation from plan: initially set T_max=200 (plan's value), which caused a regression. Fixed to T_max=90 to match actual epoch count.

## Execution

Two runs total:
- **Run 1** (T_max=200): Failed — 89.26% accuracy, a 2.55% regression. LR stayed at ~0.058 at epoch 90 because cosine only completed 45% of its cycle.
- **Run 2** (T_max=90): Succeeded — 92.10% accuracy, a 0.29% improvement over baseline. LR properly decayed to near zero.

No infrastructure errors. Single code fix between runs (T_max hyperparameter).

## Results

- **Primary metric**: 92.10% (baseline: 91.81%, delta: +0.29%, +0.32%)
- **Observations**: The gain was modest — well below the hypothesized 93-94%. The model is still a 270K-param ResNet-20 and the modern recipe only provides incremental regularization improvements at this model capacity. Final test accuracy (91.91%) was slightly below best (92.10%), suggesting some instability in the final epochs.
- **Analysis**: The hypothesis was partially validated — the modern recipe does improve over the baseline, but the 93-94% prediction was too optimistic for a model this small. The biggest bottleneck is likely model capacity, not training recipe. The LR schedule fix was the most impactful component (89.26% → 92.10% with proper T_max), while CutOut and label smoothing provided modest additional regularization. The fact that the combined gains were only +0.29% suggests diminishing returns from recipe-only improvements on this small architecture.
- **Key Learning**: Cosine T_max must match actual epoch count; combined modern recipe yields only modest gains (+0.29%) on ResNet-20 — model capacity is likely the real bottleneck for further accuracy improvements.

## Verification

- **Conditions**: All 4 passed (run completion, time budget, accuracy improvement >= 0.1%, eval frequency)
- **Review Notes**: Results confirmed trustworthy — metric improvement is real, training ran within budget, no eval tampering.
- **Verdict**: improvement
- **Verdict Basis**: All necessary conditions passed and primary metric improved by 0.29% (exceeds 0.1% threshold).

## Unexplored Avenues

- **Wider ResNet**: Increase channel widths from {16, 32, 64} to {64, 128, 256}. The modern recipe is now established — the next lever is model capacity. Evidence: airbench 96% model uses 128/512 channels.
- **Different augmentation combinations**: Try Mixup or CutMix instead of or in addition to CutOut. These provide different regularization signals (label mixing vs. spatial masking).
- **Nesterov momentum or different optimizer**: AdamW or Nesterov SGD might improve convergence speed, allowing more effective training in the fixed time budget.
- **Tuning CutOut size**: 16px might be too aggressive or too mild for this model. Try 8px or 12px.

## Next Steps

1. **Wider ResNet with modern recipe** (high confidence): Increase model width to {64, 128, 256} while keeping all EXP-000 recipe improvements. This directly targets the capacity bottleneck identified in this experiment.
2. **Architecture change to custom ConvNet** (medium confidence): Replace ResNet-20 with a speedrun-style architecture (whitening layers, wider convolutions) from airbench. Higher ceiling but higher risk.
3. **Augmentation tuning** (low confidence): Experiment with different CutOut sizes or switch to CutMix/Mixup. Likely marginal gains without model capacity increase.

## Exit Action Results
