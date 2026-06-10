# Brainstorm EXP-065
**Created**: 2026-06-09
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

## Web Search & Literature Review

- **CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features** (`knowledge/papers/cutmix-regularization.md`, https://arxiv.org/abs/1905.04899)
  CutMix replaces rectangular regions with real patches from another image and mixes labels by actual patch area. EXP-064 validated this mechanism locally at `CUTMIX_ALPHA=1.0`, `CUTMIX_PROB=0.5`.
- **mixup: Beyond Empirical Risk Minimization** (`knowledge/papers/mixup-beyond-erm.md`)
  Direct global mixup remains a failed local family, so the next idea should preserve the regional CutMix mechanism rather than return to whole-image interpolation.
- **Cutout regularization** (`knowledge/papers/cutout-cifar-regularization.md`)
  Erased-patch masking remains a failed local family, reinforcing that CutMix's advantage likely comes from replacing regions with real image content and area-adjusted labels.

No new external search was needed; the relevant paper notes are already in the knowledge base and EXP-064 just supplied local validation.

## Experimental History Review

- Current best is EXP-064 at `best_test_acc=94.11%` on commit `1119ff8`. With the explicit +0.10 percentage-point rule, EXP-065 must reach at least `94.21%` to count as an improvement.
- The new anchor is ResNet-20 with `(28,56,112)` widths, reflection crop padding, `label_smoothing=0.05`, `WEIGHT_DECAY=2e-4`, `LR_MILESTONES=[21000,64000]`, FP32 compile, channels-last, and `CUTMIX_ALPHA=1.0`, `CUTMIX_PROB=0.5`.
- EXP-064 reached 94.11% late at epoch 96 but finished at 93.02%, so CutMix improved the peak checkpoint but did not make late training monotonic. This suggests CutMix strength may need bracketing before adding another mechanism.
- Failed families to avoid as isolated retries include direct mixup, Cutout, RandAugment, ColorJitter, schedule-only second drops, cosine schedules, batch-size deviations, label-smoothing deviations, SE attention, residual stochastic depth, dropout, shortcut changes, and width beyond `(28,56,112)`.
- The most useful new pattern is that probabilistic regional mixing works where global mixup and erased-patch masking did not. The cleanest follow-up is a one-axis CutMix bracket that keeps all other anchors fixed.

## Candidate Ideas

### 1. Lower CutMix Probability to 0.25
**Summary**: Keep the EXP-064 CutMix implementation but change only `CUTMIX_PROB` from `0.5` to `0.25`. This applies regional mixing to one quarter of batches while leaving three quarters on the clean label-smoothed anchor path.

**Reasoning**: EXP-064 barely cleared the threshold and had a much lower final checkpoint than its best epoch, which is consistent with a useful but possibly strong regularization/noisy-target pressure. A lower probability preserves the validated regional mechanism while reducing how often mixed labels shape gradients. It is the most conservative local bracket because it changes exactly one scalar and should preserve throughput and the 21k LR drop.

**Sources**: `reports/exp-report-064.md`; `knowledge/papers/cutmix-regularization.md`; goal-learnings CutMix pattern; failed direct-mixup and Cutout entries.

**Estimated Effort**: low

**Risk Assessment**: Reducing probability may remove too much of the newly useful regularization and revert toward the 93.97% pre-CutMix anchor. The failure mode should still be a clean no-improvement with final metrics.

### 2. Raise CutMix Probability to 0.75
**Summary**: Keep `CUTMIX_ALPHA=1.0` and label smoothing unchanged, but increase `CUTMIX_PROB` from `0.5` to `0.75` so most batches receive regional patch mixing.

**Reasoning**: If EXP-064's late 94.11% peak indicates that regional robustness remains under-applied, stronger CutMix exposure could push the post-drop peak above 94.21%. This tests the opposite side of the probability bracket and directly estimates whether the validated mechanism wants more or less strength.

**Sources**: `reports/exp-report-064.md`; `knowledge/papers/cutmix-regularization.md`.

**Estimated Effort**: low

**Risk Assessment**: Local history is full of over-regularization failures, including direct mixup, Cutout, label-smoothing deviations, and strong augmentation. Increasing the probability may soften supervision too much under the short fixed budget.

### 3. Lower CutMix Alpha to 0.5
**Summary**: Keep `CUTMIX_PROB=0.5`, but change `CUTMIX_ALPHA` from `1.0` to `0.5` to alter the patch-area distribution while preserving the same frequency of CutMix batches.

**Reasoning**: EXP-064 validated regional mixing but did not test patch-size distribution. Lower alpha changes the beta distribution and may increase the frequency of more asymmetric area ratios, potentially making some mixed batches closer to clean supervised examples while still regularizing with regional content.

**Sources**: `knowledge/papers/cutmix-regularization.md`; `reports/exp-report-064.md`.

**Estimated Effort**: low

**Risk Assessment**: The effect is less directly interpretable than changing probability because both patch area and label weighting distribution shift. It should wait until at least one probability bracket is measured.

## Idea Evaluation

All three candidates respect the hard constraints because they modify only `train.py`, keep the fixed harness, preserve one validation per epoch, and change a single CutMix scalar. The evidence now favors local exploitation: EXP-064 is the first successful experiment since EXP-038 and directly introduced the CutMix mechanism.

Lowering `CUTMIX_PROB` to 0.25 is the best first bracket. The main concern from EXP-064 is not missing the LR drop or insufficient steps; it is that the gain appeared as a late peak while the final checkpoint fell far below the peak. Less frequent CutMix is a plausible way to retain regional regularization while giving more batches the stable clean-anchor loss. This also aligns with the broader local failure pattern that too much regularization usually underperforms under the 300s budget.

Raising `CUTMIX_PROB` to 0.75 is still worth trying if the lower bracket fails, but it is higher risk because direct mixup, Cutout, stronger smoothing changes, and policy augmentation have generally failed by adding difficulty or target softness. Lowering `CUTMIX_ALPHA` is useful but less diagnostic than a probability bracket; if probability is the main strength knob, measure it first.

The lead candidate is therefore Lower CutMix Probability to 0.25.

## Chosen Idea
**Selected**: Lower CutMix Probability to 0.25

**Why this idea**:
It is the cleanest one-scalar exploitation of the newly successful CutMix anchor. EXP-064 showed regional mixing is useful, but the weak final checkpoint suggests the current `p=0.5` may be stronger than needed.

**Hypothesis**:
Changing `CUTMIX_PROB` from `0.5` to `0.25` will preserve enough regional mixing benefit while reducing mixed-label pressure, allowing the post-drop peak to reach at least `94.21%` `best_test_acc`.
