# Brainstorm EXP-071
**Created**: 2026-06-09
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

## Web Search & Literature Review

- **RandAugment / policy augmentation knowledge entry** (`knowledge/papers/randaugment-augmentation.md`)
  Torchvision already exposes policy augmentation transforms, so augmentation policies can be tested by editing only the existing `train.py` training transform. Prior guidance says to use conservative settings because the current recipe already has crop/flip/reflection, label smoothing, stronger decay, and CutMix.
- **CutMix knowledge entry** (`knowledge/papers/cutmix-regularization.md`)
  CutMix is now validated locally and provides regional patch/label mixing that differs from both failed direct mixup and failed Cutout masking. New augmentation ideas should preserve this anchor unless the experiment is specifically testing its interaction.
- **ResNet initialization reference** (`knowledge/references/resnet-zero-init-residual.md`)
  Residual initialization can be changed inside `train.py` without modifying the harness or parameter count, but prior residual-branch scale experiments show that identity-biased variants can undertrain in the 300s budget.

## Experimental History Review

- Current best is EXP-064 at `best_test_acc=94.11%`, commit `1119ff8`, from `CUTMIX_ALPHA=1.0`, `CUTMIX_PROB=0.5`, endpoint label smoothing 0.05, `WEIGHT_DECAY=2e-4`, reflection crop padding, and first LR drop at step 21000.
- The active goal requires `best_test_acc >= 94.21%`; ties or smaller gains are `no-improvement`.
- CutMix scalar/probability and a post-drop weakening schedule are now bracketed: EXP-065/066/067/068/069 all missed threshold. The anchor should keep static `p=0.5`, `alpha=1.0` unless a new experiment tests a clearly different interaction.
- EXP-070 showed that standard CIFAR channel std is not a benign input-conditioning fix; it collapsed peak accuracy to 75.03%, so unit-std input scaling should be preserved.
- Recurring high-importance failures discourage isolated second-drop schedules, EMA/SWA averaging, batch-size deviations, and label-smoothing deviations. Medium-importance failures discourage static CutMix-strength changes, SE, direct mixup, cosine schedules, residual-BN down-scaling, scalar LR changes, BN/bias decay exceptions, and Cutout.
- Policy augmentation is only partially explored: mild RandAugment (`num_ops=1`, `magnitude=5`) underperformed EXP-038 at 93.83%, but CIFAR AutoAugment's learned CIFAR-specific policy has not been tried on the stronger EXP-064 CutMix anchor.

## Candidate Ideas

### 1. CIFAR AutoAugment on the CutMix Anchor
**Summary**: Add torchvision's CIFAR AutoAugment policy after reflection crop and horizontal flip, before `ToTensor()`, while preserving unit-std normalization and the complete EXP-064 CutMix anchor.

**Reasoning**: EXP-064 established that regional CutMix is the strongest current regularizer, and all static CutMix brackets have now been exhausted. CIFAR AutoAugment is a distinct policy-augmentation mechanism with a learned CIFAR-specific operation distribution, unlike the locally failed single-parameter RandAugment probe. It can be added in one transform line with no dependency, parameter-count, optimizer, schedule, or evaluation-harness changes.

**Sources**: `knowledge/papers/randaugment-augmentation.md`; `reports/exp-report-044.md`; `reports/exp-report-064.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`; `train.py`.

**Estimated Effort**: low

**Risk Assessment**: AutoAugment may over-regularize when stacked with CutMix, and policy transforms may add CPU dataloader overhead that reduces step coverage. Worst case is a valid no-improvement run; code risk is low if the startup log records the policy.

### 2. Fan-Out Kaiming Conv Initialization
**Summary**: Change convolution initialization from default `init.kaiming_normal_(m.weight)` to explicit `init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")`, leaving linear initialization and the rest of the CutMix anchor unchanged.

**Reasoning**: The current `_weights_init` uses PyTorch defaults, which imply fan-in mode. Many residual CNN recipes use fan-out mode for convolution layers to preserve backward variance through residual stacks. This is a narrow initialization-only probe that avoids the repeatedly failed schedule, smoothing, batch-size, normalization, and CutMix-strength families.

**Sources**: `train.py` `_weights_init`; `knowledge/references/resnet-zero-init-residual.md`; failed residual-init context from EXP-028 and EXP-051 in `goal-learnings/maximize-cifar10-best-test-accuracy.md`.

**Estimated Effort**: low

**Risk Assessment**: Initialization-only changes may be too small to clear the +0.10pp noise guard, and residual initialization variants have a weak local record. It should not affect runtime or scope, but the expected impact is modest.

### 3. Early CutMix Warmup
**Summary**: Keep `CUTMIX_PROB=0.5` after a short initial warmup, but use clean label-smoothed batches only for the first few epochs or first several thousand steps.

**Reasoning**: Static CutMix is validated, but it may perturb very early representation formation. A short clean warmup could preserve the anchor's regional mixing benefits while reducing early optimization noise. This is distinct from EXP-069's post-drop weakening, which reduced CutMix after the first LR drop and underperformed.

**Sources**: `reports/exp-report-064.md`; `reports/exp-report-069.md`; CutMix bracket failures EXP-065 through EXP-069 in the experiment index.

**Estimated Effort**: low

**Risk Assessment**: The static `p=0.5` anchor may already be well calibrated, and removing early CutMix could simply reduce regularization without improving later peak accuracy. It adds a schedule branch to the training loop and is less directly supported by literature than AutoAugment.

## Idea Evaluation

CIFAR AutoAugment has the best expected upside among the remaining distinct levers. It is not the same as the failed mild RandAugment setting: RandAugment used a simple fixed-strength policy, while CIFAR AutoAugment uses a CIFAR-specific learned policy. The main risk is over-regularization and dataloader overhead, but the code change is localized and the failure mode should be a clean no-improvement rather than invalid or crash.

Fan-out Kaiming initialization is cleaner and cheaper, but its likely effect size is small relative to the +0.10pp improvement threshold, and prior residual-initialization probes were negative. Early CutMix warmup is mechanistically plausible, but the CutMix anchor has already been locally bracketed and EXP-069 weakens confidence in temporal CutMix reductions unless paired with a stronger mechanism.

Given the current search history, the best next move is to test the one untried CIFAR-specific policy augmentation on top of the validated CutMix anchor before falling back to smaller initialization probes.

## Chosen Idea
**Selected**: CIFAR AutoAugment on the CutMix Anchor

**Why this idea**:
It is the most distinct remaining one-file augmentation lever with external support, no dependency changes, and direct compatibility with the fixed harness. It tests whether a CIFAR-specific policy can add complementary invariance after static CutMix strength, schedule, smoothing, decay, and normalization probes have been exhausted.

**Hypothesis**:
If CIFAR AutoAugment supplies complementary invariances without excessive CPU overhead or over-regularization, adding it to the EXP-064 CutMix anchor will raise `best_test_acc` from 94.11% to at least the 94.21% improvement threshold.
