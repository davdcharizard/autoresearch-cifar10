# Brainstorm EXP-064
**Created**: 2026-06-09
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

## Web Search & Literature Review

- **CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features** (`knowledge/papers/cutmix-regularization.md`, https://arxiv.org/abs/1905.04899)
  CutMix mixes rectangular image patches and labels by area, retaining real pixels unlike Cutout while providing a spatially local alternative to global mixup.
- **mixup: Beyond Empirical Risk Minimization** (`knowledge/papers/mixup-beyond-erm.md`)
  Mixup is easy to implement inside `train.py`, but this repo's completed alpha-0.1 runs stayed below the anchor, so any interpolation idea needs a different mechanism.
- **ResNet downsampling tweaks** (`knowledge/papers/resnet-downsampling-tweaks.md`)
  Shortcut/downsampling choices can affect ResNet accuracy, but both learned projection shortcuts and average-pool option-A shortcuts have already underperformed here.
- **Wide Residual Networks** (`knowledge/papers/wide-residual-networks.md`)
  Width/depth tradeoffs can help CIFAR residual models in general, but this repo's current anchor has strong negative evidence against further naive widening or depth removal.

## Experimental History Review

- Current best remains EXP-038 at `best_test_acc=93.97%`; with the explicit +0.10 percentage-point rule, EXP-064 must reach at least `94.07%` to count as an improvement.
- The active anchor is ResNet-20 with `(28,56,112)` widths, `BATCH_SIZE=128`, `LR=0.1`, `MOMENTUM=0.9`, `WEIGHT_DECAY=2e-4`, `LR_MILESTONES=[21000,64000]`, reflection crop padding, full-run label smoothing 0.05, FP32 compile, channels-last, and once-per-epoch validation.
- High- and medium-importance failed families now include schedule-only second drops, weight averaging, batch-size deviations, label-smoothing deviations, mixup variants, cosine schedules, residual BN down-scaling, initial LR deviations, no-decay BN/bias groups, Cutout, and SE attention.
- EXP-055 and EXP-060 make direct mild mixup a no-improvement family, but EXP-060 explicitly left regional mixing as a mechanistically different unexplored avenue.
- EXP-005 and EXP-009 make erased-patch Cutout low priority, but CutMix differs because it replaces erased pixels with real patches and area-adjusted labels.
- Recent architecture probes are weak: average-pool shortcuts, classifier dropout, shallow-wide ResNet-14, all-block SE, and layer3-only SE all underperformed. Any next architecture candidate needs a stronger rationale than more capacity or local gates.

## Candidate Ideas

### 1. Probabilistic CutMix Regional Mixing
**Summary**: Add `CUTMIX_ALPHA = 1.0` and `CUTMIX_PROB = 0.5` to the training loop. On half of training batches, sample a beta-distributed lambda, paste a clipped rectangular patch from a permuted batch into the current batch, recompute lambda from the actual patch area, and train with a weighted two-target cross-entropy loss. Keep the anchor architecture, optimizer, schedule, reflection crop padding, batch size, compile path, and validation cadence unchanged.

**Reasoning**: CutMix is the most distinct remaining augmentation mechanism with strong external support. It addresses Cutout's information-loss failure by retaining real pixels, and it differs from mixup's whole-image convex interpolation by preserving local image patches. A `0.5` application probability bounds regularization strength and overhead, leaving half of batches as the current anchor recipe while still testing regional label mixing.

**Sources**: `knowledge/papers/cutmix-regularization.md`; `knowledge/papers/mixup-beyond-erm.md`; reports/exp-report-055.md; reports/exp-report-060.md; goal-learnings failed-approach entries for mixup and Cutout.

**Estimated Effort**: medium

**Risk Assessment**: The negative mixup/Cutout history is a serious warning, and label mixing may still soften supervision under the short fixed budget. The failure mode should be a valid no-improvement if the LR drop is reached and metrics are captured. Implementation must avoid CPU-side overhead and preserve exactly one validation per epoch.

### 2. CIFAR AutoAugment Policy Probe
**Summary**: Insert torchvision's CIFAR-10 AutoAugment policy after crop/flip and before `ToTensor`, preserving all model and optimizer settings. This tests whether a stronger learned policy can improve generalization beyond the failed mild RandAugment setting.

**Reasoning**: EXP-044 mild RandAugment reached 93.83%, close to the anchor but below threshold. AutoAugment's CIFAR-specific policy is a distinct augmentation schedule and may have a better operation distribution than the conservative one-op RandAugment probe.

**Sources**: reports/exp-report-044.md; `knowledge/papers/randaugment-augmentation.md`; active goal constraints.

**Estimated Effort**: low

**Risk Assessment**: CPU transform overhead may reduce useful training steps, and stronger policy augmentation may over-regularize the already tuned label-smoothed anchor. This is more likely to repeat a policy-augmentation miss than CutMix is to test a genuinely different mechanism.

### 3. Tiny Final-Stage Width Redistribution
**Summary**: Preserve ResNet-20 depth but redistribute a small amount of channel capacity toward the final stage, such as `(27, 54, 120)`, keeping all optimizer and schedule settings unchanged.

**Reasoning**: EXP-062 showed removing depth is too costly, while the current anchor emerged from useful width scaling. A tiny late-stage-biased redistribution could increase semantic capacity without fully repeating broad widening.

**Sources**: `knowledge/papers/wide-residual-networks.md`; reports/exp-report-062.md; goal-learnings width-scaling entries.

**Estimated Effort**: low

**Risk Assessment**: This is close to the high-importance width-beyond-anchor failure. It may reduce early feature capacity and throughput without enough final-stage benefit, so it should rank below a more mechanistically distinct augmentation.

## Idea Evaluation

Probabilistic CutMix has the strongest combination of external evidence, novelty relative to local failures, and bounded implementation risk. It is not a direct retry of mixup because the image content remains locally coherent rather than globally blended, and it is not a direct retry of Cutout because no image region is replaced by blank pixels. The `CUTMIX_PROB=0.5` choice is conservative enough to avoid turning every batch into a strongly mixed target, while still producing a clear test of regional mixing.

CIFAR AutoAugment is easy to implement, but EXP-044 already showed that adding policy augmentation overhead and difficulty can stay below the threshold. It may still be worth trying later, especially if CutMix fails cleanly, but its mechanism is less distinct from the failed RandAugment probe.

Tiny final-stage width redistribution has weak local support. The width family produced the current anchor historically, but recent evidence says extra capacity above `(28,56,112)` and depth/width swaps are not promising. It should wait until more distinct data-regularization options are closed.

The lead candidate is therefore Probabilistic CutMix Regional Mixing. It is the clearest remaining regional-augmentation experiment, it can be implemented entirely in `train.py`, and it has a clean verification path against the 94.07% threshold.

## Chosen Idea
**Selected**: Probabilistic CutMix Regional Mixing

**Why this idea**:
It is the most distinct high-signal augmentation left after direct mixup and Cutout failures. CutMix preserves real image patches while mixing labels by area, so it tests whether local regional mixing can add useful invariance without the information loss of Cutout or the global blending behavior of mixup.

**Hypothesis**:
Applying CutMix to 50% of batches with `alpha=1.0` will preserve the step-21000 LR drop while improving regional robustness enough to reach at least `94.07%` `best_test_acc`.
