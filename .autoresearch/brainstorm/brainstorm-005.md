# Brainstorm EXP-005
**Created**: 2026-06-08
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

<!-- This file is focused on IDEATION only.
     Goal statement, primary metric, direction, hard constraints, and verification criteria
     live in the goal file (see pointer above). Baseline lives in experiment-indices/maximize-cifar10-best-test-accuracy.tsv.
     Do not duplicate those fields here — always point to the source of truth. -->

## Web Search & Literature Review

- **Cutout regularization note** (`knowledge/papers/cutout-cifar-regularization.md`)
  Cutout-style masking has direct CIFAR evidence and can be implemented as tensor-space `RandomErasing` inside `train.py` without dependencies or evaluation changes.

- **PyTorch throughput tools note** (`knowledge/references/pytorch-throughput-tools.md`)
  The current best recipe depends on FP32 `torch.compile` plus channels-last; new experiments should preserve this validated throughput path unless the hypothesis requires changing it.

- **Wide Residual Networks note** (`knowledge/papers/wide-residual-networks.md`)
  Architecture scaling remains a higher-ceiling option, but fixed-budget runtime and schedule calibration make it a riskier next experiment than an isolated augmentation change.

- **EXP-004 EMA report** (`reports/exp-report-004.md`)
  Per-step EMA reached 91.98% but missed the +0.10 threshold and lost about 6,800 optimizer steps, so the next idea should avoid material per-step overhead.

## Experimental History Review

- Current best baseline remains EXP-002 at `best_test_acc=91.95%`, commit `6743174`; because of the updated goal, EXP-005 needs at least `92.05%`.
- EXP-000 failed at 90.45% with cutout bundled with label smoothing, Nesterov, and a slow 64k-step cosine schedule. The failure points to undertraining from the combined recipe, not necessarily cutout alone.
- EXP-001 showed BF16 throughput missed baseline; preserve FP32 arithmetic.
- EXP-002 showed FP32 compile/channels-last is the only validated improvement and should remain the foundation.
- EXP-003 showed an early second LR drop at 40k hurts this baseline; avoid retuning the second milestone without stronger evidence.
- EXP-004 showed per-step EMA is too costly in this time-budgeted loop; avoid additions that materially reduce optimizer steps unless the expected accuracy gain is large.
- Gap: no experiment has isolated augmentation-only regularization on the successful EXP-002 throughput baseline.

## Candidate Ideas

### 1. Isolated Cutout on the FP32 Throughput Baseline
**Summary**: Keep EXP-002's model, optimizer, LR schedule, batch size, FP32 arithmetic, `torch.compile`, and channels-last path unchanged, and add only cutout-style masking to the training transform. Use tensor-space `transforms.RandomErasing` after `ToTensor()` and before normalization, with a square mask matching the 16x16 CIFAR cutout component that was previously bundled with other changes.

**Reasoning**: Cutout has direct CIFAR evidence and targets generalization rather than throughput. EXP-000's failure bundled multiple optimization-slowing changes, while this variant isolates the augmentation component on a faster, validated baseline. It has a plausible effect size above +0.10 points and should avoid the large per-step overhead seen in EXP-004.

**Sources**: `knowledge/papers/cutout-cifar-regularization.md`; EXP-000 report `reports/exp-report-000.md`; EXP-002 report `reports/exp-report-002.md`; EXP-004 report `reports/exp-report-004.md`.

**Estimated Effort**: low

**Risk Assessment**: Cutout may still regularize too strongly under the 300 second budget or add enough CPU transform overhead to reduce steps. The change is narrow and any failure should be a clean no-improvement.

### 2. Observed-Horizon Cosine Schedule without Extra Regularization
**Summary**: Replace the step LR schedule with a cosine schedule calibrated to the observed EXP-002 step budget rather than `MAX_STEPS=64000`, while preserving augmentation, optimizer, architecture, FP32 compile, and channels-last.

**Reasoning**: EXP-000's cosine failed because it was slow over a 64k horizon and bundled with other regularization. A cosine schedule over roughly the observed 43k-step budget could provide smoother late optimization without changing data or precision.

**Sources**: `knowledge/papers/sgdr-cosine-schedule.md`; EXP-000 report; EXP-002 report.

**Estimated Effort**: low

**Risk Assessment**: Schedule-only changes have already been fragile: EXP-003's milestone adjustment hurt accuracy. Cosine could again keep LR too high or decay too early, and a single run may be sensitive to horizon choice.

### 3. Compact Higher-Capacity ResNet Variant
**Summary**: Increase capacity modestly, such as moving from ResNet-20 to ResNet-32, while keeping FP32 compile/channels-last and retuning milestones to the expected lower step budget.

**Reasoning**: A higher-capacity CIFAR ResNet has a larger potential accuracy ceiling than incremental recipe changes. The H20 has sufficient VRAM, and architecture changes are explicitly allowed inside `train.py`.

**Sources**: README baseline notes; `knowledge/papers/wide-residual-networks.md`; TASK.md.

**Estimated Effort**: medium

**Risk Assessment**: More compute per step could reduce epochs and delay LR drops, repeating EXP-004's fixed-budget failure mode. Schedule calibration would make this a multi-variable experiment.

## Idea Evaluation

Isolated cutout is the best next experiment. It has stronger CIFAR-specific evidence than schedule retuning, does not alter arithmetic precision, and should not impose the per-step model-copy overhead that made EMA too small to count. It also directly tests an unresolved ambiguity from EXP-000: whether cutout itself was harmful or whether the bundled label smoothing plus slow cosine caused the undertraining.

Observed-horizon cosine is lower effort but less compelling because both prior schedule interventions underperformed or contributed to undertraining. The compact architecture idea has a higher ceiling, but it introduces schedule and throughput confounds at the same time; it is better after cheap augmentation is isolated.

## Chosen Idea
**Selected**: Isolated Cutout on the FP32 Throughput Baseline

**Why this idea**:
It composes with the validated EXP-002 throughput pattern, addresses a still-unresolved regularization component from EXP-000, and has enough plausible effect size to clear the new +0.10 point threshold without a large runtime penalty.

**Hypothesis**:
Adding isolated 16x16 cutout-style masking to the FP32 throughput ResNet-20 recipe will improve generalization enough to reach at least `92.05%` `best_test_acc`, while preserving most of EXP-002's optimizer-step budget.
