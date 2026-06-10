# Brainstorm EXP-059
**Created**: 2026-06-09
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

## Web Search & Literature Review

- **Deep Residual Learning for Image Recognition** (CVPR 2016, https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)
  The original ResNet paper describes CIFAR option-A shortcuts as parameter-free identity mappings with zero-padding for increased dimensions and stride-2 behavior across spatial-size changes. This matches the current code's no-parameter shortcut family and frames shortcut changes as architecture changes rather than recipe changes.
- **Bag of Tricks for Image Classification with Convolutional Neural Networks** (CVPR 2019, https://openaccess.thecvf.com/content_CVPR_2019/papers/He_Bag_of_Tricks_for_Image_Classification_with_Convolutional_Neural_Networks_CVPR_2019_paper.pdf)
  The paper revisits ResNet downsampling and notes that stride-based downsampling can ignore input feature positions; its ResNet-D tweak uses average pooling in the shortcut path as part of a downsampling refinement. Although the setting is ImageNet bottleneck ResNets, the mechanism suggests a low-overhead transition-quality probe for this CIFAR model.
- **Existing knowledge base** (`knowledge/README.md`)
  Existing entries now include augmentation, mixup, cosine, stochastic depth, SE attention, EMA, residual initialization, and throughput notes. There is no existing downsample-shortcut reference, and EXP-058 just showed broad per-block attention is weaker than the anchor.

## Experimental History Review

- Current best remains EXP-038 at `best_test_acc=93.97%`; EXP-059 must reach `94.07%` to count as an improvement under the +0.10 percentage-point rule.
- The active anchor remains `STAGE_WIDTHS=(28, 56, 112)`, `BATCH_SIZE=128`, `LR=0.1`, `MOMENTUM=0.9`, `WEIGHT_DECAY=2e-4`, `LR_MILESTONES=[21000, 64000]`, reflection crop padding, full-run label smoothing 0.05, FP32 compile, channels-last, and once-per-epoch validation.
- Recent no-improvements have closed many scalar recipe axes: LR brackets, weight-decay brackets, batch-size deviations, label-smoothing deviations, cosine tails, mixup, stochastic depth, ColorJitter, RandAugment, BN momentum, no-decay parameter groups, and SE gates.
- Architecture changes are mixed: width scaling to 28/56/112 is validated, but widening beyond that fails, projection shortcuts hurt, residual BN down-scaling undertrains, and all-block SE reached only 93.71%. This leaves a narrow gap for no-parameter transition-quality changes that do not add broad overhead.
- The current downsampling shortcut uses strided slicing (`shortcut[:, :, ::stride, ::stride]`) plus zero-channel padding. This discards most spatial samples at stage transitions, while an average-pool shortcut can preserve local information before channel padding without adding trainable parameters.

## Candidate Ideas

### 1. Average-Pool Option-A Downsample Shortcut
**Summary**: In `BasicBlock.forward`, replace stride-2 shortcut slicing with `F.avg_pool2d(shortcut, kernel_size=2, stride=2)` before zero-channel padding. Keep the shortcut parameter-free and preserve all anchor settings.

**Reasoning**: This targets transition information loss, not capacity or regularization. EXP-018 showed learned projection shortcuts hurt, but that does not rule out a parameter-free downsample that preserves more local evidence than raw strided slicing. The Bag of Tricks downsampling discussion supports the general mechanism that shortcut/downsample details can affect accuracy, and this version avoids the per-block overhead that hurt SE.

**Sources**: CVPR 2016 ResNet paper option-A shortcut; CVPR 2019 Bag of Tricks ResNet-D downsampling tweak; EXP-018 projection shortcut failure; EXP-058 SE overhead result; current `train.py` shortcut code.

**Estimated Effort**: low

**Risk Assessment**: Average pooling may blur useful high-frequency features or add enough overhead to reduce steps. It also changes a sensitive CIFAR ResNet convention, so the likely failure mode is a valid no-improvement rather than a crash.

### 2. Mixup Without Label Smoothing
**Summary**: Retry mild mixup alpha 0.1, but remove label smoothing from the mixup loss so target interpolation is not compounded with soft hard-label targets.

**Reasoning**: EXP-055 reached 93.85%, closer than many recent failures, and may have been over-regularized by combining mixup with `label_smoothing=0.05`. This is a coupled regularization-balance test rather than a direct repeat.

**Sources**: EXP-055 mixup retry; `knowledge/papers/mixup-beyond-erm.md`; smoothing failures EXP-033, EXP-037, EXP-057.

**Estimated Effort**: medium

**Risk Assessment**: It revisits a recently failed family and may again underperform while adding batch-mixing overhead. It also temporarily violates the "keep full-run smoothing" lesson, though only because mixup supplies its own label softening.

### 3. Stage-3-Only Squeeze-and-Excitation
**Summary**: Add SE gates only in the final residual stage instead of all blocks, preserving the rest of the anchor.

**Reasoning**: EXP-058 rejected broad all-block SE, but its failure may partly reflect overhead and early-stage gating. Restricting SE to high-level channels reduces overhead while retaining a channel-calibration mechanism.

**Sources**: EXP-058 report; `knowledge/papers/squeeze-and-excitation-networks.md`; current stage structure in `train.py`.

**Estimated Effort**: medium

**Risk Assessment**: This is close to a just-failed idea and likely has lower expected value than a distinct shortcut/downsample probe. It would require a small API change to pass stage context into blocks.

## Idea Evaluation

Average-Pool Option-A Downsample Shortcut has the best balance of novelty, mechanism clarity, and cost. It is not a scalar recipe bracket, not an isolated regularizer, and not a broad parameter-adding architecture change. It directly targets a concrete code path where the current shortcut discards spatial samples, and it has external support from ResNet downsampling refinements while staying closer to parameter-free CIFAR option-A than EXP-018's learned projection shortcuts.

Mixup Without Label Smoothing is plausible because EXP-055 was a relatively high no-improvement, but it stays in a regularization family that has repeatedly missed the anchor and adds more implementation complexity.

Stage-3-Only SE is a reasonable narrower follow-up, but EXP-058 is too fresh a negative signal to prioritize another attention variant immediately. It is better held as a backup if shortcut/downsample changes fail.

The lead candidate is therefore a no-parameter average-pool shortcut transition: improve stage transition information flow while preserving the proven training recipe and avoiding trainable shortcut projections.

## Chosen Idea
**Selected**: Average-Pool Option-A Downsample Shortcut

**Why this idea**:
It is a distinct architecture-quality probe with low overhead and clear code localization. It preserves the validated recipe and option-A no-parameter shortcut family while testing whether stage transitions are losing useful spatial information through strided slicing.

**Hypothesis**:
Replacing strided shortcut slicing with average pooling before zero-padding will improve downsample transition quality enough to reach at least `94.07%` best test accuracy, while preserving a valid single-GPU run, the step-21000 LR drop, and all anchor training settings.
