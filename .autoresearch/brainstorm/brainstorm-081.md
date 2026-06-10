# Brainstorm EXP-081
**Created**: 2026-06-09
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

## Web Search & Literature Review

- **Torchvision RandomCrop padding modes** (`knowledge/references/torchvision-randomcrop-padding.md`)
  RandomCrop padding can be changed with a one-line `train.py` edit, preserving the evaluation harness, optimizer, architecture, dependencies, and training loop.
- **CutMix regularization** (`knowledge/papers/cutmix-regularization.md`)
  The current best anchor uses regional label/image mixing; future perturbations should preserve this validated mechanism unless directly testing CutMix itself.
- **Wide residual networks** (`knowledge/papers/wide-residual-networks.md`)
  Width/capacity can improve CIFAR models in general, but local fixed-budget evidence says architecture changes need careful step-budget tradeoffs.
- **ResNet downsampling tweaks** (`knowledge/papers/resnet-downsampling-tweaks.md`)
  Shortcut/downsampling details are plausible low-overhead architecture levers, but local transition-smoothing variants now underperform the anchor.

No new external search was needed; the local 80-experiment trajectory is more specific than additional generic CIFAR literature for this narrow next choice.

## Experimental History Review

- Current best remains EXP-064 at `best_test_acc=94.11%` from commit `1119ff8`; the +0.10pp noise guard requires `best_test_acc >= 94.21%`.
- EXP-080 closed the early optimizer-softening lane: a 500-step LR warmup peaked at 94.08%, and LR startup deviations are now a high-importance failed family.
- CutMix strength and timing are well bracketed: probability 0.25/0.75, alpha 0.5/2.0, late probability taper, clean warmup, and short probability ramp all missed threshold.
- Near misses EXP-072, EXP-073, and EXP-074 reached 94.16%, 94.14%, and 94.17%, but EXP-075 showed combining two near misses can regress rather than add.
- Added augmentation is usually too strong: Cutout, mild ColorJitter, RandAugment, and AutoAugment all missed; however, the current anchor still uses reflection crop padding and has not bracketed padding magnitude.
- Architecture/topology families are weak locally: SE, shortcut smoothing, residual-branch downsampling, pre-activation, classifier-head tweaks, and shallow-wide depth reduction all missed.
- The remaining plausible gap is over-regularization around an otherwise strong CutMix anchor. A tiny reduction in spatial jitter is distinct from adding policy augmentation or weakening CutMix itself.

## Candidate Ideas

### 1. Slightly Weaker Reflection Crop Jitter
**Summary**: Keep `padding_mode="reflect"` but reduce `RandomCrop(32, padding=4, padding_mode="reflect")` to `padding=3`. Preserve CutMix alpha/probability/smoothing, clean label smoothing, architecture, optimizer, LR schedule, batch size, compile/channels-last, seed, and validation cadence.

**Reasoning**: The anchor already combines strong regional CutMix regularization with crop/flip augmentation. Several added regularizers have underperformed, suggesting the remaining gain may come from slightly reducing augmentation strength rather than adding more. This directly brackets an untested degree of freedom in the validated reflection crop mechanism, while preserving the padding mode that beat nearby boundary-fill siblings.

**Sources**: `knowledge/references/torchvision-randomcrop-padding.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`; EXP-029, EXP-031, EXP-064, EXP-071, EXP-080

**Estimated Effort**: low

**Risk Assessment**: The effect may be too small to clear a +0.10pp threshold, or reduced translation jitter may hurt generalization. The failure mode should be a clean no-improvement with no infrastructure risk.

### 2. Narrow Layer3 Width Rebalance
**Summary**: Keep ResNet-20 depth and the first two stage widths unchanged, but reduce final stage width from 112 to 104: `STAGE_WIDTHS=(28, 56, 104)`. Preserve the current CutMix, optimizer, LR schedule, transforms, batch size, compile/channels-last path, seed, and validation cadence.

**Reasoning**: Fixed-budget performance depends on both capacity and post-drop optimization coverage. A small final-stage reduction could trade a little representational capacity for more steps after the 21k LR drop while staying much closer to the validated 28/56/112 anchor than prior shallow-wide or topology changes.

**Sources**: `knowledge/papers/wide-residual-networks.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`; EXP-062, EXP-078, EXP-080; `train.py` `STAGE_WIDTHS`

**Estimated Effort**: low

**Risk Assessment**: Architecture changes are a weak local family, and reducing final-stage width may lower the accuracy ceiling more than it improves step coverage. This is low code risk but lower expected value than the crop-jitter probe.

### 3. Lower Horizontal Flip Probability
**Summary**: Replace default `transforms.RandomHorizontalFlip()` with `transforms.RandomHorizontalFlip(p=0.4)` while preserving reflection crop padding 4 and all CutMix, optimizer, schedule, architecture, and training-loop settings.

**Reasoning**: Like crop padding, flip probability is a no-overhead spatial augmentation strength knob. If the current CutMix anchor is slightly over-regularized, a modest reduction in flip frequency could improve clean late accuracy without touching CutMix frequency or LR behavior.

**Sources**: `train.py` transform pipeline; `knowledge/papers/cutmix-regularization.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`; EXP-071, EXP-080

**Estimated Effort**: low

**Risk Assessment**: CIFAR horizontal flips are usually a strong default, so reducing them may simply weaken invariance. It is also less directly grounded in local experiments than padding magnitude, because reflection crop behavior has explicit prior support.

## Idea Evaluation

The slight reflection-crop padding reduction has the strongest project-specific evidence. It targets a narrow unbracketed augmentation-strength knob inside the validated transform pipeline, and it directly responds to the pattern that extra regularization often underperforms on top of CutMix. It also avoids closed families: it is not a CutMix alpha/probability/timing retune, not an LR schedule change, and not a topology rewrite.

The layer3 width rebalance is attractive because it might improve step coverage without a broad architecture rewrite, but architecture experiments have repeatedly underperformed. Since the current best is only 0.10pp below the required threshold, lowering capacity is a risky way to chase a small gain.

The horizontal-flip probability test is simple, but it has weaker evidence. Reflection padding has already been validated and only its magnitude remains untested; flip probability lacks that local bracket and may remove a useful invariance. It is a reasonable later augmentation micro-tune if padding 3 is neutral or slightly positive.

## Chosen Idea
**Selected**: Slightly Weaker Reflection Crop Jitter

**Why this idea**:
It is the cleanest remaining low-risk probe of over-regularization around the CutMix anchor. It changes one transform parameter, preserves all successful recipe components, and tests a knob not covered by the closed CutMix, LR, label-smoothing, architecture, and policy-augmentation families.

**Hypothesis**:
If the current CutMix anchor is marginally over-regularized by full 4-pixel reflection crop jitter, then reducing reflection padding to 3 will improve late clean accuracy enough to raise `best_test_acc` from 94.11% to at least 94.21%.
