# Brainstorm EXP-010
**Created**: 2026-06-08
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

<!-- This file is focused on IDEATION only.
     Goal statement, primary metric, direction, hard constraints, and verification criteria
     live in the goal file (see pointer above). Baseline lives in experiment-indices/maximize-cifar10-best-test-accuracy.tsv.
     Do not duplicate those fields here — always point to the source of truth. -->

## Web Search & Literature Review

- **Wide Residual Networks** (`knowledge/papers/wide-residual-networks.md`)
  Widening can improve CIFAR residual-network accuracy more efficiently than simply increasing depth, but this repo's fixed 300s budget means any capacity experiment needs runtime-aware scheduling.

- **SGDR / cosine scheduling** (`knowledge/papers/sgdr-cosine-schedule.md`)
  Cosine schedules can improve anytime SGD performance on CIFAR-style training, but local schedule-only experiments have weakened the case for more scheduler work around the current recipe.

- **EXP-000 recipe bundle** (`reports/exp-report-000.md`)
  Nesterov momentum and label smoothing were only tested inside a confounded bundle with strong cutout and slow cosine decay, so neither component has been isolated under the successful EXP-002 schedule.

- **EXP-009 weak cutout report** (`reports/exp-report-009.md`)
  Weak 8x8 cutout preserved throughput but still missed the threshold, making more erased-patch regularization lower priority.

No new external search was needed; the next ideas are grounded in saved paper notes and completed local experiment reports.

## Experimental History Review

- Current baseline is still EXP-002 at `best_test_acc=91.95%`; the goal requires `>=92.05%` to count as a real improvement.
- EXP-002 remains the only accepted improvement: FP32 compile/channels-last ResNet-20 with the original `[32000, 48000]` schedule.
- Precision changes are unattractive: BF16 missed baseline in EXP-001 and TF32 slowed the known-good path in EXP-007.
- Schedule-only retuning is weakening: EXP-003's reachable second drop and EXP-008's earlier first drop both reduced peak accuracy.
- Cutout-style masking is now a recurring failed approach: EXP-005 full 16x16 cutout peaked at 91.72%, and EXP-009 weak 8x8 cutout peaked at 91.87%.
- ResNet-32 depth scaling undertrained badly in EXP-006, so architecture experiments should prefer smaller width changes or measured schedule calibration rather than another depth-only increase.
- Nesterov momentum and label smoothing remain unisolated from the failed EXP-000 bundle and can be tested without major throughput risk.

## Candidate Ideas

### 1. Isolated Nesterov Momentum on FP32 Baseline
**Summary**: Preserve the full EXP-002 recipe and change only the SGD optimizer from classical momentum to Nesterov momentum by setting `nesterov=True` with the existing `momentum=0.9`, LR, weight decay, augmentation, schedule, and model.

**Reasoning**: Nesterov was present in EXP-000, but that run also added strong cutout, label smoothing, and a slow cosine schedule, so the optimizer effect was completely confounded. This is a minimal isolated change with essentially no expected throughput cost. If Nesterov improves late low-LR refinement or stabilizes the high-LR phase, the effect only needs to clear +0.10 points.

**Sources**: `reports/exp-report-000.md`; `reports/exp-report-002.md`; `experiment-indices/maximize-cifar10-best-test-accuracy.tsv`.

**Estimated Effort**: low

**Risk Assessment**: The effect may be too small or slightly negative under the existing step schedule. Worst case is a clean no-improvement with useful isolation of the optimizer component.

### 2. Isolated Mild Label Smoothing
**Summary**: Preserve the EXP-002 recipe and add only mild cross-entropy label smoothing, such as `label_smoothing=0.03` or `0.05`, without cutout, cosine decay, Nesterov, or architecture changes.

**Reasoning**: Label smoothing can improve calibration/generalization in some classifiers, and EXP-000 did not isolate it from stronger regularization and schedule changes. It has no meaningful throughput cost and no evaluation-harness impact. The risk is that smoothing may reduce top-1 peak accuracy even if loss or calibration improves.

**Sources**: `reports/exp-report-000.md`; `reports/exp-report-009.md`; goal-learnings failed-approach entries for cutout and schedule failures.

**Estimated Effort**: low

**Risk Assessment**: Label smoothing can trade confidence for calibration and may lower the exact `best_test_acc` target. The best value is likely narrow, so a single smoothing coefficient may miss.

### 3. Compact Width Increase with Schedule Calibration
**Summary**: Keep the ResNet-20 depth but widen channels modestly, for example 16/32/64 -> 20/40/80 or a width multiplier near 1.25, and set LR milestones based on a conservative expected step budget.

**Reasoning**: The WRN note suggests width can improve CIFAR residual networks more efficiently than depth, and EXP-006 only tested a deeper ResNet-32 that missed its first LR drop. A small width increase may offer a better capacity/runtime tradeoff than depth. However, it needs careful planning because even modest capacity changes can alter step count and schedule timing.

**Sources**: `knowledge/papers/wide-residual-networks.md`; `reports/exp-report-006.md`; `reports/exp-report-002.md`.

**Estimated Effort**: medium

**Risk Assessment**: Increased channel count may reduce steps enough to miss the useful low-LR refinement window or increase compile overhead. It also has a larger code and schedule surface than optimizer-only changes.

## Idea Evaluation

The strongest near-term candidate is isolated Nesterov. It directly resolves an EXP-000 confound, preserves the only proven FP32 throughput path, and has the cleanest failure mode. The required margin is small enough that an optimizer dynamics change could matter, and if it fails the result will be easy to interpret.

Mild label smoothing is similarly cheap, but its mechanism is less clearly aligned with maximizing exact top-1 accuracy under a short budget. It may improve loss while reducing peak accuracy, and the coefficient choice introduces an extra small hyperparameter guess.

The compact width idea has the highest theoretical ceiling, but EXP-006 showed that capacity changes can fail for schedule/runtime reasons before testing their actual accuracy potential. It should remain a candidate after one more cheap optimizer or loss ablation, ideally with a measured step-budget plan.

## Chosen Idea
**Selected**: Isolated Nesterov Momentum on FP32 Baseline

**Why this idea**:
It is the most interpretable next ablation: a single optimizer flag that was previously confounded, with almost no throughput or scope risk, while avoiding the recurring cutout failure and the weakened schedule-only path.

**Hypothesis**:
Adding Nesterov momentum to the EXP-002 FP32 compile/channels-last ResNet-20 recipe will improve optimization or late-stage refinement enough to reach at least `92.05%` `best_test_acc` without reducing the fixed-budget step count.
