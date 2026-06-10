# Brainstorm EXP-072
**Created**: 2026-06-09
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

## Web Search & Literature Review

- **ResNet initialization reference** (`knowledge/references/resnet-zero-init-residual.md`)
  Initialization can be changed inside `train.py` without parameter-count, optimizer, schedule, or harness changes. Prior identity-biased residual initialization variants are risky under the 300s budget, but the file also points to standard ResNet initialization practice as a narrow code lever.
- **CutMix knowledge entry** (`knowledge/papers/cutmix-regularization.md`)
  The current baseline depends on preserving regional CutMix, so non-CutMix experiments should keep `CUTMIX_ALPHA=1.0` and `CUTMIX_PROB=0.5`.
- **Project knowledge index** (`knowledge/README.md`)
  Recent saved knowledge covers augmentation, CutMix, residual initialization, SE, stochastic depth, schedules, and throughput. After EXP-071, the highest-signal untried low-cost lever is initialization rather than another isolated augmentation/policy variant.

## Experimental History Review

- Current best remains EXP-064 at `best_test_acc=94.11%`, commit `1119ff8`, from probabilistic CutMix on the label-smoothed, `2e-4` weight-decay, reflection-padding anchor.
- The improvement threshold is now `best_test_acc >= 94.21%`; ties and smaller gains are `no-improvement`.
- CutMix scalar/probability, post-drop taper, standard CIFAR std, and CIFAR AutoAugment all failed after EXP-064. The anchor should keep static CutMix `alpha=1.0`, `p=0.5`, unit-std normalization, and no policy augmentation.
- Recurring failures discourage schedule-only second drops, EMA/SWA, batch-size deviations, label-smoothing deviations, direct mixup, cosine schedules, SE, residual-BN down-scaling, scalar LR changes, BN/bias decay exceptions, Cutout, and policy augmentation.
- The current `_weights_init` applies `init.kaiming_normal_(m.weight)` to both Conv2d and Linear modules using PyTorch defaults. A conv-only fan-out ReLU initialization has not been tried on the current CutMix anchor.

## Candidate Ideas

### 1. Fan-Out Kaiming Conv Initialization
**Summary**: Change convolution initialization to explicit ReLU Kaiming normal with `mode="fan_out"` while leaving linear initialization on the current default Kaiming normal path.

**Reasoning**: The local model is a residual CNN with post-activation BasicBlocks. Fan-out conv initialization is common in residual CNN implementations because it preserves backward signal scaling through convolutional stacks. This is narrower than prior residual-BN down-scaling failures: it changes only initial weight variance for convolutions, not residual branch output scale, architecture, optimizer, schedule, augmentation, or evaluation.

**Sources**: `train.py` `_weights_init`; `knowledge/references/resnet-zero-init-residual.md`; EXP-028/051 as cautionary residual-init failures; EXP-064 as the CutMix anchor to preserve.

**Estimated Effort**: low

**Risk Assessment**: Expected effect size may be small and could stay inside noise. It may also slightly miscalibrate the LR=0.1 early phase. Worst case is a valid no-improvement run; code risk is very low.

### 2. Early CutMix Warmup
**Summary**: Use clean label-smoothed batches for a short initial warmup, then enable the validated static `CUTMIX_PROB=0.5` recipe for the rest of training.

**Reasoning**: Static CutMix is validated, but it may add early representation noise. A short warmup is distinct from EXP-069's post-drop weakening because it preserves the successful CutMix regularization during post-drop refinement.

**Sources**: `reports/exp-report-064.md`; `reports/exp-report-069.md`; CutMix bracket results EXP-065 through EXP-069.

**Estimated Effort**: low

**Risk Assessment**: It adds a schedule branch and may simply reduce helpful regularization. EXP-069 makes temporal CutMix reductions lower confidence unless the mechanism is clearly different.

### 3. Lightweight Classifier Weight Normalization Probe
**Summary**: Apply a narrow final-layer initialization adjustment, such as explicitly initializing `fc.weight` with Kaiming normal for linear fan-in while keeping convolutions unchanged.

**Reasoning**: Final classifier calibration can affect peak test accuracy without throughput or architecture changes. The current initializer treats Conv2d and Linear the same, so a classifier-specific init could be tested without touching the harness.

**Sources**: `train.py` `_weights_init`; failed classifier-head dropout EXP-061 as a caution that final-head regularization is weak.

**Estimated Effort**: low

**Risk Assessment**: This is likely too small to clear the +0.10pp threshold and may be dominated by seed noise. It is less grounded than fan-out conv initialization.

## Idea Evaluation

Fan-out Kaiming conv initialization is the best next candidate because it is a clean non-augmentation, non-schedule lever after many regularization and policy variants failed. It has a clear mechanism in residual CNN signal scaling, touches only initialization, and should preserve runtime and all validated anchor settings.

Early CutMix warmup is plausible but weaker because several CutMix temporal/static variants have already failed, and its effect could simply reduce a validated regularizer. Classifier-only initialization is safe but probably too small to clear the noise guard.

The next experiment should therefore test conv-only fan-out initialization while explicitly preserving linear initialization and every EXP-064 anchor setting.

## Chosen Idea
**Selected**: Fan-Out Kaiming Conv Initialization

**Why this idea**:
It is the clearest remaining low-risk lever that avoids the now-recurring failed families. It tests whether the residual CNN benefits from conv fan-out scaling without changing parameter count, throughput, augmentation, optimizer, schedule, or evaluation.

**Hypothesis**:
If the current default fan-in-style conv initialization is slightly miscalibrated for this residual stack, switching Conv2d layers to fan-out ReLU Kaiming normal will improve `best_test_acc` from 94.11% to at least 94.21% while preserving the CutMix anchor.
