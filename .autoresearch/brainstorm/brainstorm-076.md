# Brainstorm EXP-076
**Created**: 2026-06-09
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

## Web Search & Literature Review

- **CutMix regularization** (`knowledge/papers/cutmix-regularization.md`)
  CutMix remains the validated regional-mixing mechanism in this repo; the current best EXP-064 keeps `CUTMIX_ALPHA=1.0`, `CUTMIX_PROB=0.5`, and endpoint label smoothing 0.05.
- **ResNet initialization context** (`knowledge/references/resnet-zero-init-residual.md`, `reports/exp-report-072.md`, `reports/exp-report-075.md`)
  Initialization changes can be tested without throughput or evaluation changes, but residual/Conv2d initialization variants have mostly been too small or regressive under this 300s budget.
- **Wide residual network context** (`knowledge/papers/wide-residual-networks.md`)
  Capacity changes can improve CIFAR residual models in general, but this repo's current width/depth experiments show fixed-budget and schedule calibration are major constraints.

## Experimental History Review

- Current best remains EXP-064: probabilistic CutMix reached `best_test_acc=94.11%`; the active noise guard requires `best_test_acc >= 94.21%`.
- The static CutMix anchor is locally bracketed: EXP-065/066 probability brackets, EXP-067/068 alpha brackets, EXP-069 post-drop taper, EXP-073 warmup, EXP-074 endpoint hard labels, and EXP-075 fan-out plus hard endpoints all failed to clear 94.21%.
- Label-smoothing deviations are now High Importance failures; EXP-075 specifically showed hard CutMix endpoints do not compose with Conv2d fan-out initialization.
- Conv2d/residual initialization is weak: EXP-072 reached a sub-threshold 94.16%, EXP-075 regressed to 93.92%, and residual BN down-scaling failed. However, final classifier initialization remains untested and is mechanically different from residual identity bias.
- Classifier-head dropout failed in EXP-061, so the next classifier experiment should avoid training-time feature dropout. A pure initialization calibration has no per-step overhead and does not add regularization during training.
- The current `_weights_init` applies Kaiming normal to both Conv2d and Linear weights, even though the final Linear layer produces logits and is not followed by a ReLU. This is a narrow untested code-path mismatch.

## Candidate Ideas

### 1. Xavier Classifier Init With Zero Bias
**Summary**: Keep the CutMix anchor unchanged, but split `_weights_init` so Conv2d keeps the current default Kaiming normal behavior while the final Linear classifier uses Xavier/Glorot initialization and zero bias. Add a startup marker confirming the classifier initialization.

**Reasoning**: The current Linear layer receives pooled ReLU features and directly emits logits, yet it is initialized with the same Kaiming normal call used for convolutional ReLU stacks. A classifier-specific Xavier initialization is a standard logit-head calibration choice and is mechanically distinct from the failed Conv2d fan-out and residual-BN initialization variants. It has no throughput, parameter-count, augmentation, schedule, or evaluation impact, and it preserves the validated CutMix and label-smoothing anchor.

**Sources**: `train.py` `_weights_init`; `reports/exp-report-072.md`; `reports/exp-report-075.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`

**Estimated Effort**: low

**Risk Assessment**: The effect may be too small to clear a +0.10pp threshold, and classifier-head dropout already suggests the final head is not the main bottleneck. Worst case is a clean no-improvement with unchanged runtime.

### 2. Bias-Free Classifier Head
**Summary**: Change `self.fc = nn.Linear(w3, num_classes)` to `nn.Linear(w3, num_classes, bias=False)` while preserving the existing Kaiming normal weight initialization and every CutMix-anchor setting.

**Reasoning**: CIFAR-10 classes are balanced, and the classifier bias is a tiny unregularized class-prior degree of freedom. Removing it could force logits to rely on learned features rather than early bias calibration, with almost no implementation or runtime cost. It tests classifier calibration without changing representation learning or the loss.

**Sources**: `train.py` classifier definition; `reports/exp-report-061.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`

**Estimated Effort**: low

**Risk Assessment**: This may be a negligible or harmful change because the bias can also help calibrate logits. It changes parameter count slightly, which is acceptable but less directly motivated than changing only initialization.

### 3. Shorter 1000-Step Clean Warmup Before CutMix
**Summary**: Preserve CutMix endpoint smoothing 0.05 and all anchor settings, but disable CutMix only for the first 1000 optimizer steps before restoring static `CUTMIX_PROB=0.5`.

**Reasoning**: EXP-073's 2000-step warmup reached 94.14%, above baseline but below threshold. A shorter warmup could reduce very-early mixed-label noise while preserving more total CutMix exposure. This is the least bad remaining CutMix-timing variant, but it remains inside a weak local family.

**Sources**: `reports/exp-report-073.md`; `reports/exp-report-074.md`; `reports/exp-report-075.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`

**Estimated Effort**: low

**Risk Assessment**: Temporal CutMix refinements have repeatedly stayed in the noise band, and further schedule-like CutMix tuning risks chasing sub-threshold variance.

## Idea Evaluation

The classifier Xavier/zero-bias initialization has the cleanest mechanism among the candidates: it targets the only remaining untested initialization mismatch in `train.py` without touching the now-bracketed CutMix, label-smoothing, optimizer, schedule, or architecture settings. It is also more precise than a bias-free classifier because it keeps the classifier's representational form and parameter count unchanged while changing only the initial logit scale/offset behavior.

The bias-free classifier is nearby and low cost, but it is less compelling because the bias can be useful calibration and because changing parameter count, however slightly, is not necessary to test classifier initialization. The shorter warmup has the strongest direct recent metric among alternatives, but EXP-073, EXP-074, and EXP-075 collectively argue that CutMix-internal refinements are too small or non-additive under the 94.21% guard.

## Chosen Idea
**Selected**: Xavier Classifier Init With Zero Bias

**Why this idea**:
It is the narrowest distinct experiment left after closing Conv2d fan-out plus hard CutMix endpoints. It preserves the successful CutMix recipe and all high-value anchors while testing a plausible initialization mismatch in the final logit layer. The effect size is uncertain, but it avoids the stronger negative priors around more smoothing, CutMix scheduling, width/depth, residual initialization, and optimizer-only retuning.

**Hypothesis**:
If the current Kaiming-normal Linear initialization produces a slightly miscalibrated initial logit scale for the non-ReLU classifier head, then Xavier classifier weights with zero bias will improve post-drop calibration enough to raise `best_test_acc` from 94.11% to at least 94.21%.
