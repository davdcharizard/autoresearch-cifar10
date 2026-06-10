# Brainstorm EXP-009
**Created**: 2026-06-08
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

<!-- This file is focused on IDEATION only.
     Goal statement, primary metric, direction, hard constraints, and verification criteria
     live in the goal file (see pointer above). Baseline lives in experiment-indices/maximize-cifar10-best-test-accuracy.tsv.
     Do not duplicate those fields here — always point to the source of truth. -->

## Web Search & Literature Review

- **Cutout regularization note** (`knowledge/papers/cutout-cifar-regularization.md`)
  Cutout-style masking is a known CIFAR augmentation that can improve generalization, and the closest local implementation is tensor-space `transforms.RandomErasing` inside `train.py`.

- **EXP-005 report** (`reports/exp-report-005.md`)
  Isolated 16x16 cutout preserved throughput but over-regularized the fixed-budget ResNet-20 recipe, peaking at 91.72%. The report explicitly leaves smaller or lower-probability cutout as an unexplored avenue.

- **EXP-008 report** (`reports/exp-report-008.md`)
  Schedule-only retuning around the first LR drop reduced accuracy, making another schedule-only experiment lower priority.

- **WRN note** (`knowledge/papers/wide-residual-networks.md`)
  Compact wider architectures remain a possible higher-ceiling direction, but prior depth scaling undertrained badly and should be approached carefully.

No new external search was needed; the next idea is grounded in existing local paper notes and completed experiment reports.

## Experimental History Review

- Current baseline remains EXP-002 at `best_test_acc=91.95%`; the tightened goal requires `>=92.05%`.
- EXP-002 remains the only accepted recipe improvement: FP32 compile/channels-last ResNet-20 with the original `[32000, 48000]` schedule.
- EXP-003 and EXP-008 both weaken the schedule-only path: a reachable LR 0.001 phase hurt, and an earlier LR 0.01 phase also hurt.
- EXP-005 shows full 16x16 cutout is too strong, but it did not test weaker masking.
- EXP-000 bundled Nesterov, label smoothing, cutout, and cosine, so it does not isolate optimizer-only or weak-regularization effects.
- EXP-006 shows depth-only capacity scaling can undertrain badly under this budget.
- EXP-007 shows TF32 should stay disabled for this small CNN.

## Candidate Ideas

### 1. Weak 8x8 Cutout on FP32 Baseline
**Summary**: Preserve the EXP-002 ResNet-20 recipe and add a much weaker RandomErasing/Cutout transform than EXP-005: fixed 8x8 square area, lower probability, after normalization using the same implementation style as the prior cutout experiment.

**Reasoning**: EXP-005 showed cutout does not materially hurt throughput, but 16x16 masks delay convergence too much. Reducing the masked area to one quarter of EXP-005 and reducing probability should retain some generalization pressure while lowering the underfitting cost. This directly tests the open avenue identified by EXP-005 rather than repeating the failed full-strength version.

**Sources**: `knowledge/papers/cutout-cifar-regularization.md`; `reports/exp-report-005.md`; `reports/exp-report-002.md`.

**Estimated Effort**: low

**Risk Assessment**: It may still regularize too much or be too weak to matter. The expected failure mode is clean no-improvement, with little throughput risk.

### 2. Nesterov Momentum Only
**Summary**: Keep the EXP-002 recipe and change SGD from classical momentum to Nesterov momentum, with all schedule, augmentation, precision, and architecture settings preserved.

**Reasoning**: Nesterov was part of EXP-000's failed bundle, but that run was confounded by cutout, label smoothing, and slow cosine decay. A one-line isolated optimizer change could improve optimization dynamics without adding evaluation overhead or changing the data distribution.

**Sources**: `reports/exp-report-000.md`; `reports/exp-report-002.md`.

**Estimated Effort**: low

**Risk Assessment**: The effect may be small or interact poorly with the existing step schedule. Because the required margin is +0.10 points, a small optimizer effect could matter, but evidence is weaker than for the cutout-size ablation.

### 3. Compact Width Increase
**Summary**: Modify the ResNet channel widths modestly, such as using a small width multiplier, while keeping depth at ResNet-20 and preserving the EXP-002 schedule initially.

**Reasoning**: WRN evidence suggests width can be more efficient than depth for CIFAR. EXP-006 only tested deeper ResNet-32, not a compact width increase, so this could provide a higher capacity ceiling without the same depth overhead.

**Sources**: `knowledge/papers/wide-residual-networks.md`; `reports/exp-report-006.md`.

**Estimated Effort**: medium

**Risk Assessment**: Any capacity increase risks reducing step count or missing schedule phases. It should probably be preceded by a small warmup estimate or planned schedule calibration.

## Idea Evaluation

Weak 8x8 cutout is the best next experiment because it directly tests an identified gap from EXP-005 while preserving the validated FP32 baseline. It is not a repeat of the failed full cutout experiment: the mask area is one quarter as large and the probability can be lower, targeting the over-regularization mechanism. It also avoids the now-weakened schedule-only path and does not introduce capacity/schedule confounds.

Nesterov-only is attractive because it is tiny and cheap, but it has less direct evidence in the local trajectory. Compact width is higher ceiling, but after EXP-006 it should be planned with more care than a quick single-variable run. Since the current goal needs only +0.10 points, a carefully weakened regularizer is a reasonable low-risk attempt before architecture work.

## Chosen Idea
**Selected**: Weak 8x8 Cutout on FP32 Baseline

**Why this idea**:
It is a focused ablation of the failed cutout direction, keeping the same fast baseline and same implementation family while reducing the specific regularization strength that likely caused EXP-005 to miss.

**Hypothesis**:
Adding low-probability fixed 8x8 RandomErasing to the EXP-002 FP32 throughput ResNet-20 recipe will improve generalization enough to reach at least `92.05%` without the convergence delay caused by 16x16 cutout.
