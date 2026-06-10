# Brainstorm EXP-014
**Created**: 2026-06-08
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

<!-- This file is focused on IDEATION only.
     Goal statement, primary metric, direction, hard constraints, and verification criteria
     live in the goal file (see pointer above). Baseline lives in experiment-indices/maximize-cifar10-best-test-accuracy.tsv.
     Do not duplicate those fields here — always point to the source of truth. -->

## Web Search & Literature Review

- **Wide Residual Networks** (`knowledge/papers/wide-residual-networks.md`)
  Widening residual networks is a supported CIFAR scaling direction and has now produced two local improvements when paired with runtime-aware LR milestones.

- **EXP-013 ResNet-20 Width 1.5x report** (`reports/exp-report-013.md`)
  The 24/48/96 model reached 92.49% with 41,825 steps, 605,026 parameters, and the 24k first drop, establishing the new baseline.

- **Goal learnings** (`goal-learnings/maximize-cifar10-best-test-accuracy.md`)
  Width scaling is now the strongest recurring positive pattern, but capacity experiments must explicitly calibrate milestones to reachable steps.

No new external search was needed. The local evidence from EXP-011 and EXP-013 is more directly relevant than generic CIFAR recipes.

## Experimental History Review

- Current baseline is EXP-013 at `best_test_acc=92.49%`; the tightened success rule requires EXP-014 to reach at least `92.59%`.
- EXP-011 improved the FP32 throughput baseline by widening to 20/40/80 and using a 24k first drop, reaching 92.12%.
- EXP-013 improved again by widening to 24/48/96 while keeping the 24k drop, reaching 92.49% despite a lower 41,825-step budget.
- EXP-012 showed that a 22k first drop is too early for 20/40/80, but that does not rule out an earlier drop for a larger and slower model whose LR 0.01 window would otherwise shrink.
- Depth scaling failed because it missed the planned first drop; further capacity scaling should stay width-based and choose a first milestone that is safely reachable.
- Cutout, Nesterov, TF32, per-step EMA, and early second drops remain unattractive relative to the width path.

## Candidate Ideas

### 1. ResNet-20 Width 1.75x with Earlier 22k First Drop
**Summary**: Increase stage widths from 24/48/96 to 28/56/112 and move the first LR milestone from 24000 to 22000, keeping the second milestone unreachable at 64000. This tests whether another width step can raise the accuracy ceiling while an earlier first drop preserves enough LR 0.01 refinement under the likely lower step budget.

**Reasoning**: Width scaling is the only direction with repeated confirmed improvements. EXP-013 showed 24/48/96 can clear the threshold with only 41,825 steps, so a 28/56/112 model may have enough capacity to reach 92.59% if it still enters LR 0.01 with adequate time left. The earlier 22k drop is not a retry of EXP-012 because the larger model changes the step-budget tradeoff; it is a schedule calibration for a slower capacity point.

**Sources**: `knowledge/papers/wide-residual-networks.md`; `reports/exp-report-013.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`; `reports/exp-report-006.md`.

**Estimated Effort**: low

**Risk Assessment**: The model may be too wide for the 300s budget or the 22k drop may still be too early, causing under-exploration. Worst case is a clean no-improvement that maps the next width boundary.

### 2. Earlier 22k First Drop on Current 24/48/96 Baseline
**Summary**: Keep stage widths at 24/48/96 and move the first LR drop from 24000 to 22000. This isolates whether the new best model benefits from more LR 0.01 refinement time without adding capacity.

**Reasoning**: EXP-013 crossed the threshold shortly after the LR drop and peaked during the low-LR phase, so extra refinement might lift the same model above 92.59%. However, EXP-012 already showed that a 22k drop can be too early on the previous widened model, so this is a local schedule test with modest expected upside.

**Sources**: `reports/exp-report-012.md`; `reports/exp-report-013.md`; `experiment-indices/maximize-cifar10-best-test-accuracy.tsv`.

**Estimated Effort**: low

**Risk Assessment**: This may reproduce the EXP-012 failure mode and gain less than the required +0.10 points. It is safer than further widening but likely lower upside.

### 3. Low-Frequency Late Averaging on 24/48/96
**Summary**: Keep the EXP-013 architecture and schedule, but add sparse or epoch-level weight averaging after the first LR drop to smooth the final LR 0.01 trajectory without per-step overhead.

**Reasoning**: EXP-013 peaked at 92.49% and ended lower at 92.25%, suggesting final weights fluctuate or overfit late. Averaging may capture a more stable solution, and the stronger 24/48/96 baseline gives this mechanism more headroom than the old ResNet-20 baseline.

**Sources**: `reports/exp-report-004.md`; `references/pytorch-ema-averaging.md`; `reports/exp-report-013.md`.

**Estimated Effort**: medium

**Risk Assessment**: Averaging risks implementation complexity, batch-norm handling issues, and overhead. Per-step EMA already missed the threshold once, so only sparse late averaging would be defensible.

## Idea Evaluation

The 1.75x width candidate has the highest expected impact because the threshold moved to 92.59%, and schedule-only gains may not be large enough. It follows the strongest positive pattern: width scaling has produced both EXP-011 and EXP-013 improvements. Moving the first drop to 22k is a calculated schedule calibration for a slower model, not a generic earlier-drop retry.

The 22k schedule retune on 24/48/96 is cleaner and lower risk, but its ceiling may be too close to EXP-013. Since EXP-013 already ended below its peak, more low-LR time could help, but EXP-012 warns that earlier drops can hurt.

Late averaging is plausible but more complex and has a known overhead failure mode. It should be kept as a future option if width scaling starts to plateau or if a no-overhead implementation is planned carefully.

## Chosen Idea
**Selected**: ResNet-20 Width 1.75x with Earlier 22k First Drop

**Why this idea**:
It best matches the new difficulty of the goal. The next success requires at least 92.59%, and repeated local evidence says width scaling is the clearest path to meaningful gains. Pairing 28/56/112 with a 22k first drop gives the larger model a safer LR 0.01 refinement window under the fixed 300s budget.

**Hypothesis**:
A 28/56/112 ResNet-20 with first LR drop at step 22000 will retain enough steps to refine at LR 0.01 and improve `best_test_acc` to at least `92.59%`.
