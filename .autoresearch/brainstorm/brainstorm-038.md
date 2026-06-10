# Brainstorm EXP-038
**Created**: 2026-06-09
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

<!-- This file is focused on IDEATION only.
     Goal statement, primary metric, direction, hard constraints, and verification criteria
     live in the goal file (see pointer above). Baseline lives in experiment-indices/maximize-cifar10-best-test-accuracy.tsv.
     Do not duplicate those fields here — always point to the source of truth. -->

## Web Search & Literature Review

- **EXP-023 report** (`reports/exp-report-023.md`)
  Lowering weight decay to `5e-5` weakened the 28/56/112 anchor, suggesting the current recipe is not over-regularized by `1e-4`.
- **EXP-032 report** (`reports/exp-report-032.md`)
  `label_smoothing=0.05` improved the reflection anchor to 93.70% without throughput loss and remains the current baseline.
- **EXP-037 report** (`reports/exp-report-037.md`)
  Raising label smoothing to 0.08 reached 93.73%, only +0.03 over baseline and still inside the +0.10 noise band.
- **PyTorch EMA averaging note** (`knowledge/references/pytorch-ema-averaging.md`)
  Averaged weights remain a possible late-stability mechanism, but prior local evidence warns that update frequency and windowing matter.

## Experimental History Review

- Current baseline is EXP-032 at `best_test_acc=93.70%`; under the +0.10 percentage-point rule, EXP-038 must reach at least `93.80%`.
- The current anchor is `STAGE_WIDTHS = (28, 56, 112)`, reflected `RandomCrop`, `label_smoothing=0.05`, `BATCH_SIZE = 128`, `WEIGHT_DECAY = 1e-4`, `LR_MILESTONES = [21000, 64000]`, FP32, channels-last, cuDNN benchmark, and `torch.compile`.
- Recent local probes exhausted adjacent smoothing and schedule composition: 0.03 smoothing, 0.08 smoothing, 22k first drop, and their combination all failed to clear the +0.10 threshold.
- Smaller batch sizes are now a medium-importance failed family after batch size 96 and 112 both lost useful coverage or plateaued below baseline.
- Lower weight decay failed, but the opposite direction has not been tested on the stronger reflection plus label-smoothing anchor. This is a distinct no-throughput scalar regularization probe.
- More complex late averaging remains possible, but EXP-004 and EXP-021 show per-step EMA overhead and long equal averaging can both fail badly.

## Candidate Ideas

### 1. Increase Weight Decay to 2e-4
**Summary**: Preserve the current reflection-padding, label-smoothed 28/56/112 anchor and change only `WEIGHT_DECAY` from `1e-4` to `2e-4`.

**Reasoning**: Lower weight decay underperformed, so the model may benefit more from stronger shrinkage than weaker shrinkage. This is a one-scalar, no-throughput regularization probe that avoids the failed smoothing bracket and smaller-batch coverage loss. It also keeps the validated architecture, augmentation, label smoothing, schedule, and optimizer class intact.

**Sources**: `reports/exp-report-023.md`; `reports/exp-report-032.md`; `reports/exp-report-037.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md` § Failed Approaches and Patterns; current `train.py`.

**Estimated Effort**: low

**Risk Assessment**: Stronger weight decay could over-regularize and reduce peak top-1 accuracy. The failure mode should be a clean no-improvement result because the change is one scalar and does not alter runtime structure.

### 2. Initial LR 0.12 with Existing Milestones
**Summary**: Preserve the current anchor but raise `LR` from `0.1` to `0.12`, keeping `LR_MILESTONES = [21000, 64000]`.

**Reasoning**: A slightly larger high-LR phase could improve exploration before the proven 21k first drop, without changing model size or epoch geometry. Since schedule-only adjacent first-drop changes have plateaued, this tests high-LR dynamics rather than milestone timing.

**Sources**: `experiment-indices/maximize-cifar10-best-test-accuracy.tsv`; `goal-learnings/maximize-cifar10-best-test-accuracy.md` § Failed Approaches and Patterns.

**Estimated Effort**: low

**Risk Assessment**: Higher LR may destabilize early training or worsen calibration. It has less direct evidence than weight decay and may interact with the fixed 21k schedule in a harder-to-attribute way.

### 3. Post-Drop Low-Frequency EMA
**Summary**: Add an EMA model only after the first LR drop and update it once per epoch, evaluating only the EMA model after activation.

**Reasoning**: Prior per-step EMA was too expensive and prior long equal averaging collapsed. A low-frequency post-drop EMA could target late stability while avoiding per-step overhead and long equal-average drift.

**Sources**: `reports/exp-report-004.md`; `reports/exp-report-021.md`; `knowledge/references/pytorch-ema-averaging.md`.

**Estimated Effort**: medium

**Risk Assessment**: This touches more code and sits near two failed averaging variants. BatchNorm buffer handling is a known pitfall, and a bad averaging setup could repeat EXP-021's collapse.

## Idea Evaluation

The best next experiment is increased weight decay because it is a distinct, no-throughput scalar after the smoothing and batch-size spaces both produced repeated no-improvements. It also tests a clear bracket: `5e-5` was too weak, while `2e-4` may improve generalization if the current wider, label-smoothed model remains slightly under-regularized by weight shrinkage.

Initial LR 0.12 is simple but less grounded; schedule-only work has repeatedly disappointed, and LR changes may be harder to distinguish from milestone effects. Post-drop EMA remains interesting but has higher code risk and two nearby failure modes, so it should wait until simpler scalar levers are exhausted.

EXP-038 should therefore change only `WEIGHT_DECAY` from `1e-4` to `2e-4`, preserving all other anchor choices and verifying the usual schedule, batch, parameter, and metric conditions.

## Chosen Idea
**Selected**: Increase Weight Decay to 2e-4

**Why this idea**:
It is a one-scalar, no-throughput probe that tests a distinct regularization axis after smoothing, batch size, and adjacent schedule probes failed to clear the tightened threshold. The lower-weight-decay failure gives a concrete reason to test the stronger side.

**Hypothesis**:
Increasing `WEIGHT_DECAY` from `1e-4` to `2e-4` will improve generalization on the reflection-padding, label-smoothed 28/56/112 anchor enough to raise `best_test_acc` from 93.70% to at least 93.80%.
