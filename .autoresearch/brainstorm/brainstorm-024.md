# Brainstorm EXP-024
**Created**: 2026-06-08
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

<!-- This file is focused on IDEATION only.
     Goal statement, primary metric, direction, hard constraints, and verification criteria
     live in the goal file (see pointer above). Baseline lives in experiment-indices/maximize-cifar10-best-test-accuracy.tsv.
     Do not duplicate those fields here — always point to the source of truth. -->

## Web Search & Literature Review

- **SGDR / cosine scheduling** (`knowledge/papers/sgdr-cosine-schedule.md`)
  Smooth LR decay remains a plausible schedule alternative, but the only local cosine trial was confounded with strong regularization and undertrained.

- **Wide Residual Networks** (`knowledge/papers/wide-residual-networks.md`)
  Wider CIFAR residual networks are generally effective, but local evidence now shows a fixed-budget ceiling above the 28/56/112 anchor.

No new external search was needed. The current decision is driven by local experimental evidence: the 28/56/112, 21k first-drop anchor remains best; width increases above it, adjacent first-drop brackets, averaging, and lower weight decay have failed.

## Experimental History Review

- Current baseline is EXP-016 at `best_test_acc=93.23%`; with the goal's +0.10 percentage-point rule, EXP-024 must reach at least `93.33%`.
- The proven anchor remains `STAGE_WIDTHS = (28, 56, 112)` with first LR drop at step 21000.
- First-drop bracketing around the anchor is now tight: 20k reached 93.18%, 21k reached 93.23%, and 23k reached 92.88%; schedule-only exploration should avoid more adjacent first-drop moves.
- Width beyond 28/56/112 is a high-importance recurring failure across proportional, minimal, and final-stage-only variants.
- Lower weight decay (`5e-5`) preserved throughput but dropped to 92.83%, so the current anchor does not look over-regularized by `1e-4`.
- The current schedule's second milestone at 64000 is unreachable under the fixed 300s budget. Recent runs complete roughly 41k-43k steps, leaving an untested opportunity for a short late LR 0.001 refinement phase.

## Candidate Ideas

### 1. Add a Reachable Second LR Drop to the 28/56/112, 21k Anchor
**Summary**: Preserve the current architecture, batch size, optimizer, weight decay, augmentation, FP32 compile/channels-last path, and first LR drop at 21000. Move the second LR milestone from 64000 to 36000 so the run gets several thousand late steps at `lr=0.001`.

**Reasoning**: The current anchor never reaches the second LR drop, so its late training remains at `lr=0.01` until time expires. Recent completed runs reach about 42k steps, so a 36k second drop should provide a real but not dominant final refinement window. This tests a schedule dimension not yet tried on the 28/56/112, 21k anchor while avoiding the exhausted first-drop bracket.

**Sources**: `experiment-indices/maximize-cifar10-best-test-accuracy.tsv` EXP-016, EXP-022, EXP-023; `goal-learnings/maximize-cifar10-best-test-accuracy.md` schedule and width entries; `train.py` `LR_MILESTONES`.

**Estimated Effort**: low

**Risk Assessment**: EXP-003 showed an earlier second drop hurt the smaller FP32 ResNet-20, so this may also over-decay. The difference is that EXP-024 keeps the validated 21k first drop and gives a longer LR 0.01 phase before the second drop.

### 2. Batch Size 96 with Step-Budget-Aware Schedule
**Summary**: Reduce `BATCH_SIZE` from 128 to 96 and recalibrate milestones after estimating reachable steps, testing whether more gradient updates and higher optimization noise improve generalization on the anchor.

**Reasoning**: Batch size changes offer a non-explicit regularization lever distinct from cutout and weight decay. They could increase update count under the fixed time budget and shift the noise scale, which may help escape the current plateau.

**Sources**: `train.py` data loader and optimizer setup; `experiment-indices/maximize-cifar10-best-test-accuracy.tsv` EXP-016, EXP-023.

**Estimated Effort**: medium

**Risk Assessment**: This confounds batch noise, image throughput, epoch length, and milestone semantics. It needs careful schedule calibration, making it less isolated than a reachable second LR drop.

### 3. Isolated Cosine Decay on the 28/56/112 Anchor
**Summary**: Replace the unreachable second-step schedule with a cosine schedule over the expected reachable step budget while keeping the anchor architecture and regularization unchanged.

**Reasoning**: Cosine decay can provide smoother late refinement than abrupt steps. The local failed cosine run also included cutout and label smoothing, so isolated cosine on the validated anchor remains technically untested.

**Sources**: `knowledge/papers/sgdr-cosine-schedule.md`; `experiment-indices/maximize-cifar10-best-test-accuracy.tsv` EXP-000 and EXP-016.

**Estimated Effort**: medium

**Risk Assessment**: The fixed time budget makes exact horizon selection brittle, and EXP-000 suggests slow cosine can undertrain if the decay is not calibrated. A simple second drop is a cleaner first test.

## Idea Evaluation

The reachable second LR drop has the best evidence-to-risk balance. It preserves every currently validated component and changes only an unreachable schedule milestone into a reachable late-refinement phase. It also directly addresses a visible recipe issue: `LR_MILESTONES = [21000, 64000]` means the current anchor effectively has only one LR drop despite completing over 40k steps.

Batch size 96 is interesting, but it changes several mechanisms at once and would require a measured schedule recalibration before it is interpretable. Isolated cosine is also plausible, but it introduces horizon tuning and has one locally negative, though confounded, prior result. The second-drop experiment is the smallest clean test of whether the current anchor needs lower LR late in training.

## Chosen Idea
**Selected**: Add a Reachable Second LR Drop to the 28/56/112, 21k Anchor

**Why this idea**:
It targets an untested schedule gap while preserving the validated first drop, architecture, regularization, and throughput path. With recent runs reaching roughly 42k steps, a second drop at 36k should create a short LR 0.001 phase that may improve late generalization without sacrificing the proven LR 0.01 refinement window.

**Hypothesis**:
Changing `LR_MILESTONES` from `[21000, 64000]` to `[21000, 36000]` on the unchanged 28/56/112 anchor will improve late refinement and raise `best_test_acc` to at least 93.33% while preserving all hard constraints.
