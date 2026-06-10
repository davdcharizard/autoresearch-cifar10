# Brainstorm EXP-037
**Created**: 2026-06-09
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

<!-- This file is focused on IDEATION only.
     Goal statement, primary metric, direction, hard constraints, and verification criteria
     live in the goal file (see pointer above). Baseline lives in experiment-indices/maximize-cifar10-best-test-accuracy.tsv.
     Do not duplicate those fields here — always point to the source of truth. -->

## Web Search & Literature Review

- **EXP-032 report** (`reports/exp-report-032.md`)
  Isolated `label_smoothing=0.05` improved the reflection-padding anchor to 93.70% without throughput loss.
- **EXP-033 report** (`reports/exp-report-033.md`)
  Lower smoothing at 0.03 nearly cleared the current threshold at 93.79%, showing smoothing magnitude remains a sensitive scalar.
- **EXP-035 report** (`reports/exp-report-035.md`)
  Combining lower smoothing with a later first drop regressed to 93.63%, so adjacent near-miss composition is not reliable.
- **EXP-036 report** (`reports/exp-report-036.md`)
  Batch size 112 preserved the first drop but reduced useful coverage and peaked at 93.43%, deprioritizing smaller-batch stochasticity.

## Experimental History Review

- Current baseline is EXP-032 at `best_test_acc=93.70%`; under the +0.10 percentage-point rule, EXP-037 must reach at least `93.80%`.
- The current anchor is `STAGE_WIDTHS = (28, 56, 112)`, reflected `RandomCrop`, `label_smoothing=0.05`, `BATCH_SIZE = 128`, `LR_MILESTONES = [21000, 64000]`, FP32, channels-last, cuDNN benchmark, and `torch.compile`.
- EXP-033's `label_smoothing=0.03` and EXP-034's 22k first drop each reached 93.79%, but EXP-035 showed the two effects do not add.
- EXP-036 promoted smaller-batch tuning to a medium-importance failed approach; batch size 96 and 112 both lost useful coverage or plateaued below baseline.
- Strong avoid signals remain: no width beyond 28/56/112, no isolated second-drop retuning, no cutout masking, no smaller batches, no projection shortcuts, no zero-gamma init, no momentum 0.95, and no BN/bias no-decay.
- Remaining low-scope space favors no-throughput scalar regularization or carefully bounded late stability mechanisms.

## Candidate Ideas

### 1. Raise Label Smoothing to 0.08
**Summary**: Preserve the current reflection-padding 28/56/112 anchor and change only the training loss from `label_smoothing=0.05` to `label_smoothing=0.08`.

**Reasoning**: Isolated mild label smoothing is the current successful anchor, and smoothing changes do not affect throughput, schedule reachability, model size, or evaluation cadence. The 0.03 result was close to the threshold but still inside the noise band; testing the opposite side of 0.05 checks whether the useful mechanism is stronger confidence regularization rather than weaker smoothing.

**Sources**: `reports/exp-report-032.md`; `reports/exp-report-033.md`; `reports/exp-report-035.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md` § Patterns.

**Estimated Effort**: low

**Risk Assessment**: Stronger smoothing may suppress peak top-1 accuracy. The failure mode should be a clean no-improvement result because the change is one scalar and does not alter runtime structure.

### 2. Increase Weight Decay to 2e-4
**Summary**: Preserve all current anchor settings except changing `WEIGHT_DECAY` from `1e-4` to `2e-4`.

**Reasoning**: This is a no-throughput regularization scalar that could improve generalization if the current larger anchor still benefits from stronger weight shrinkage. It avoids the smaller-batch throughput loss and schedule-composition failures.

**Sources**: `reports/exp-report-023.md`; `reports/exp-report-032.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md` § Failed Approaches.

**Estimated Effort**: low

**Risk Assessment**: Lower weight decay already failed and the learnings currently say to keep `1e-4`; increasing weight decay could over-regularize or simply move away from the validated anchor. This is plausible but weaker than the smoothing-value bracket.

### 3. Short-Window Late Weight Averaging
**Summary**: Add a bounded late averaging mechanism that starts after the first LR drop and averages only a short recent window before once-per-epoch evaluation.

**Reasoning**: The current anchor has a stable post-drop plateau, and a short window might improve evaluated weights without touching the harness. This targets late stability rather than regularization strength.

**Sources**: `knowledge/references/pytorch-ema-averaging.md`; `reports/exp-report-021.md`; `reports/exp-report-032.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md` § Failed Approaches.

**Estimated Effort**: medium

**Risk Assessment**: Prior averaging approaches either lost throughput or collapsed as snapshots accumulated. This needs more code and carries higher risk of repeating a known failed family if not carefully bounded.

## Idea Evaluation

The best next experiment is the stronger-smoothing scalar because it stays within the only recent successful regularization family while avoiding the failed batch-size and schedule-composition paths. It is also a clean bracket around the current `0.05` anchor: 0.03 nearly succeeded but did not clear the noise rule, and 0.08 tests whether the best point is on the stronger side.

Higher weight decay is also no-throughput, but it contradicts the current `1e-4` anchor guidance more directly after lower weight decay failed. Short-window averaging remains interesting but has a worse complexity-to-evidence ratio and sits near prior averaging failures.

EXP-037 should therefore change only `label_smoothing` from 0.05 to 0.08 and preserve every other anchor setting. If this fails, smoothing-value bracketing is likely exhausted for now, and future work should move to a distinct late-stability mechanism.

## Chosen Idea
**Selected**: Raise Label Smoothing to 0.08

**Why this idea**:
It is a one-scalar, no-throughput probe inside the only regularization family that has produced a recent valid improvement. It avoids smaller-batch coverage loss and adjacent schedule/smoothing composition, both of which now have fresh negative evidence.

**Hypothesis**:
Increasing `label_smoothing` from 0.05 to 0.08 will improve late generalization on the reflection-padding 28/56/112 anchor enough to raise `best_test_acc` from 93.70% to at least 93.80%.
