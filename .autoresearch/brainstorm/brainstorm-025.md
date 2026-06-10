# Brainstorm EXP-025
**Created**: 2026-06-08
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

<!-- This file is focused on IDEATION only.
     Goal statement, primary metric, direction, hard constraints, and verification criteria
     live in the goal file (see pointer above). Baseline lives in experiment-indices/maximize-cifar10-best-test-accuracy.tsv.
     Do not duplicate those fields here — always point to the source of truth. -->

## Web Search & Literature Review

- **Wide Residual Networks** (`knowledge/papers/wide-residual-networks.md`)
  Wider CIFAR residual networks remain useful background, but local fixed-budget evidence says the current 28/56/112 anchor is the capacity ceiling for now.

- **SGDR / cosine scheduling** (`knowledge/papers/sgdr-cosine-schedule.md`)
  Smooth scheduling remains possible, but local schedule-only step changes are now tightly bounded.

No new external search was needed. The next useful move is driven by local evidence: architecture, first-drop timing, second-drop timing, lower weight decay, and averaging have all been tested around the current anchor without clearing the threshold.

## Experimental History Review

- Current baseline is EXP-016 at `best_test_acc=93.23%`; with the goal's +0.10 percentage-point rule, EXP-025 must reach at least `93.33%`.
- The current anchor remains `STAGE_WIDTHS = (28, 56, 112)` with `WEIGHT_DECAY = 1e-4` and first LR drop near the middle of the reachable step budget.
- Capacity above 28/56/112 is a recurring high-importance failure.
- First-drop schedule-only bracketing is bounded: 20k is slightly early, 21k is best, and 23k is too late for the anchor.
- A reachable second drop at 36k improved late refinement to 93.13% but did not beat the 93.23% baseline or 93.33% threshold.
- Explicit regularization changes have mostly hurt: cutout variants, combined label smoothing/cosine/cutout, lower weight decay, and naive averaging all missed the threshold.
- Batch size remains an untested lever that changes update count and stochasticity without adding explicit augmentation or model capacity.

## Candidate Ideas

### 1. Batch Size 96 with Step-Budget-Aware Milestones
**Summary**: Reduce `BATCH_SIZE` from 128 to 96 on the unchanged 28/56/112 anchor. Recalibrate milestones to `[26000, 44000]` so the first drop remains near the middle of the expected step budget and the second drop remains reachable late.

**Reasoning**: Smaller batches can change the optimization noise scale and may improve generalization without explicit regularization. They may also increase the number of optimizer steps under the fixed 300s training budget. Because batch size changes step count and epoch length, the LR milestones should be moved in step units rather than reusing the 128-batch schedule blindly.

**Sources**: `train.py` `BATCH_SIZE` and `LR_MILESTONES`; `experiment-indices/maximize-cifar10-best-test-accuracy.tsv` EXP-016, EXP-024; `goal-learnings/maximize-cifar10-best-test-accuracy.md`.

**Estimated Effort**: medium

**Risk Assessment**: This couples batch size and schedule, so attribution is less isolated. Smaller batches may reduce image throughput enough to offset any update-count or noise-scale benefit.

### 2. Momentum 0.95 on the Current Anchor
**Summary**: Keep the anchor and schedule unchanged, but increase classical SGD momentum from `0.9` to `0.95`.

**Reasoning**: Higher momentum may smooth noisy updates and improve late convergence without changing model size or explicit regularization. This is distinct from EXP-010, which tested Nesterov rather than the momentum coefficient.

**Sources**: `train.py` optimizer setup; `experiment-indices/maximize-cifar10-best-test-accuracy.tsv` EXP-010 and EXP-016.

**Estimated Effort**: low

**Risk Assessment**: The expected effect may be too small to clear the +0.10 threshold, and higher momentum can overshoot after LR drops if not jointly retuned.

### 3. Isolated Label Smoothing 0.05
**Summary**: Add mild label smoothing to the cross-entropy loss on the current anchor while preserving architecture and schedule.

**Reasoning**: Label smoothing can improve calibration/generalization without masking data. The only local label-smoothing trial was confounded with cutout and cosine, so an isolated weaker smoothing value remains untested.

**Sources**: `experiment-indices/maximize-cifar10-best-test-accuracy.tsv` EXP-000; `goal-learnings/maximize-cifar10-best-test-accuracy.md` regularization failures; `train.py` loss call.

**Estimated Effort**: low

**Risk Assessment**: Explicit regularization has repeatedly hurt this fixed-budget recipe, so this has a lower prior than batch-size noise-scale changes.

## Idea Evaluation

Batch size 96 has the strongest remaining mechanism. It is the only candidate that changes the stochastic optimization regime without increasing capacity or adding explicit regularization, and it may create more optimizer updates within the same time budget. It does require schedule recalibration, but the chosen milestones are mechanically tied to the expected higher step budget rather than arbitrary first-drop bracketing.

Momentum 0.95 is clean and cheap, but its expected impact is likely small and could need further retuning. Label smoothing is also cheap but runs against the accumulated evidence that explicit regularization additions hurt this benchmark under the fixed budget.

## Chosen Idea
**Selected**: Batch Size 96 with Step-Budget-Aware Milestones

**Why this idea**:
It targets an untested optimization-noise lever while preserving the validated 28/56/112 architecture, `1e-4` weight decay, and FP32 compile/channels-last path. Recalibrating milestones to `[26000, 44000]` keeps the schedule meaningful if the smaller batch increases reachable optimizer steps.

**Hypothesis**:
Changing `BATCH_SIZE` from 128 to 96 and `LR_MILESTONES` from `[21000, 64000]` to `[26000, 44000]` will improve the noise/update-count tradeoff and raise `best_test_acc` to at least 93.33% while respecting all hard constraints.
