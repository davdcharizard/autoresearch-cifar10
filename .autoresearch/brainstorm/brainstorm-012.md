# Brainstorm EXP-012
**Created**: 2026-06-08
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

<!-- This file is focused on IDEATION only.
     Goal statement, primary metric, direction, hard constraints, and verification criteria
     live in the goal file (see pointer above). Baseline lives in experiment-indices/maximize-cifar10-best-test-accuracy.tsv.
     Do not duplicate those fields here — always point to the source of truth. -->

## Web Search & Literature Review

- **Wide Residual Networks** (`knowledge/papers/wide-residual-networks.md`)
  Widening is a supported CIFAR residual-network scaling direction, but fixed-budget runs need runtime-aware schedules.

- **EXP-011 ResNet-20 Width 1.25x report** (`reports/exp-report-011.md`)
  The 20/40/80 stage widths preserved a 43,713-step budget and reached 92.12%, with the final epoch setting the best accuracy.

- **EXP-004 EMA report** (`reports/exp-report-004.md`)
  Per-step EMA gave a small accuracy gain but lost too many steps; any averaging retry should avoid per-step overhead.

No new external search was needed; the next experiment is a direct exploitation step from the latest successful local result and the saved WRN note.

## Experimental History Review

- Current baseline is EXP-011 at `best_test_acc=92.12%`; the tightened success rule requires EXP-012 to reach at least `92.22%`.
- The validated recipe is now FP32 compile/channels-last ResNet-20 with stage widths 20/40/80 and milestones `[24000, 64000]`.
- EXP-011 reached the first LR drop at step 24000, then improved through the LR 0.01 phase and set its best accuracy at the final epoch. That suggests either more LR 0.01 refinement or a nearby milestone retune could matter.
- EXP-008 showed that moving the first drop earlier on the old 16/32/64 ResNet-20 was harmful, but EXP-011 changed both capacity and timing; the widened model appears to need a much earlier drop than the old baseline.
- Larger capacity is plausible but risky: EXP-006 depth scaling failed by missing the first LR drop, while EXP-011 width scaling worked because the schedule was explicitly calibrated.
- Recurring cutout failures and Nesterov/TF32 failures should be avoided for now. EMA remains a possible future variant only if the update overhead is sharply reduced.

## Candidate Ideas

### 1. Earlier First LR Drop on Widened ResNet-20
**Summary**: Keep the successful EXP-011 widened architecture and all other recipe choices, but move the first LR milestone from 24000 to 22000 while leaving the second milestone unreachable at 64000. This gives the model roughly 2000 additional LR 0.01 steps under the same 300s training budget.

**Reasoning**: EXP-011 reached its best accuracy at the final epoch, not earlier in the run, and completed 19,713 LR 0.01 refinement steps after the drop. If the widened model was still refining at the end, a slightly earlier first drop may increase the low-LR refinement window enough to exceed the new 92.22% threshold without changing architecture or throughput. This is not a retry of EXP-008 because the architecture and previous optimal milestone regime changed after EXP-011.

**Sources**: `reports/exp-report-011.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`; `experiment-indices/maximize-cifar10-best-test-accuracy.tsv`.

**Estimated Effort**: low

**Risk Assessment**: The first drop may become too early and reduce the high-LR exploration phase, recreating EXP-008's schedule-only failure in the widened regime. Worst case is a valid no-improvement with clean schedule calibration evidence.

### 2. Slightly Wider ResNet-20 with Calibrated First Drop
**Summary**: Increase stage widths from 20/40/80 to 24/48/96 and set an earlier first LR milestone, likely around 20000-22000, to account for the lower expected step budget. Keep depth, optimizer, augmentation, FP32 compile, channels-last, and once-per-epoch validation fixed.

**Reasoning**: EXP-011 validated width as a working capacity direction, and WRN evidence supports width over depth for CIFAR residual networks. A second width step may raise the ceiling above 92.22%, but it must be schedule-calibrated because the extra channels will increase per-step cost.

**Sources**: `knowledge/papers/wide-residual-networks.md`; `reports/exp-report-011.md`; `reports/exp-report-006.md`.

**Estimated Effort**: medium

**Risk Assessment**: The wider model may lose too many steps or need a different milestone than can be guessed in one run. It also raises VRAM and parameter count; that is allowed only if accuracy improves meaningfully.

### 3. Low-Overhead Late Averaging on Widened ResNet-20
**Summary**: Keep EXP-011 architecture and schedule, but add an averaging mechanism only late in training or at low frequency after the LR drop, avoiding per-step EMA. Evaluate the averaged weights once per epoch using the same evaluator cadence.

**Reasoning**: EXP-004 showed that averaging can slightly improve accuracy but per-step EMA overhead delayed the schedule. EXP-011 already reaches a stronger low-LR plateau, so a late-only average might smooth final weights without losing thousands of steps.

**Sources**: `reports/exp-report-004.md`; `references/pytorch-ema-averaging.md`; `reports/exp-report-011.md`.

**Estimated Effort**: medium

**Risk Assessment**: Even low-frequency averaging adds implementation complexity and may still reduce throughput or mishandle batch norm buffers. It is also less direct than schedule tuning after the final-epoch EXP-011 peak.

## Idea Evaluation

The earlier-drop experiment has the clearest causal mechanism and the smallest blast radius: EXP-011's final-epoch peak implies the widened model may benefit from more LR 0.01 refinement, and the change can be isolated to one scheduler constant. Its expected impact is modest, but the current threshold only needs +0.10 points over a near-miss successful baseline.

The slightly wider model has higher upside, but its schedule is less certain and it may repeat the capacity-undertraining failure mode from EXP-006 if the step budget falls more than expected. It is a good follow-up after one local schedule exploitation run establishes whether 20/40/80 still has unused headroom.

Late averaging is interesting, but EXP-004 already showed the overhead trap. It should be revisited only with a carefully designed low-overhead implementation, ideally after deciding whether the widened baseline's schedule can be improved without new machinery.

## Chosen Idea
**Selected**: Earlier First LR Drop on Widened ResNet-20

**Why this idea**:
It is the cleanest exploitation of the new best recipe. EXP-011 set its best accuracy at the end of the run, so moving the first drop from 24000 to 22000 directly tests whether more low-LR refinement can clear the new 92.22% threshold while preserving the successful architecture and throughput path.

**Hypothesis**:
A 22000-step first LR drop on the 20/40/80 ResNet-20 will preserve the fixed-budget step count near EXP-011 while giving enough extra LR 0.01 refinement to reach `best_test_acc >= 92.22%`.
