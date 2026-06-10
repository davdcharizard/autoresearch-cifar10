# Brainstorm EXP-004
**Created**: 2026-06-08
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

<!-- This file is focused on IDEATION only.
     Goal statement, primary metric, direction, hard constraints, and verification criteria
     live in the goal file (see pointer above). Baseline lives in experiment-indices/maximize-cifar10-best-test-accuracy.tsv.
     Do not duplicate those fields here — always point to the source of truth. -->

## Web Search & Literature Review

- **PyTorch EMA averaging docs** (`knowledge/references/pytorch-ema-averaging.md`; source URL: https://docs.pytorch.org/docs/stable/generated/torch.optim.swa_utils.AveragedModel.html)
  `AveragedModel` supports EMA through `get_ema_multi_avg_fn`, providing a dependency-free way to keep averaged weights and buffers during training.

- **Wide Residual Networks note** (`knowledge/papers/wide-residual-networks.md`)
  Widening residual networks raises CIFAR accuracy ceilings, but a compact variant needs careful runtime and schedule checks under the 300 second budget.

- **Cutout regularization note** (`knowledge/papers/cutout-cifar-regularization.md`)
  Cutout-style masking can improve CIFAR generalization and can be tested inside `train.py`, but prior bundled regularization undertrained this setup.

- **PyTorch throughput tools note** (`knowledge/references/pytorch-throughput-tools.md`)
  The successful baseline already uses precision-preserving `torch.compile` plus channels-last; future ideas should preserve that path unless the change needs otherwise.

## Experimental History Review

- Current best is EXP-002: `best_test_acc=91.95%`, baseline commit `6743174`; with the updated goal threshold, EXP-004 must reach at least `92.05%` to count as an improvement.
- EXP-000 failed at 90.45% when exact cutout, label smoothing, Nesterov, and a slow 64k-step cosine schedule were bundled together.
- EXP-001 showed BF16 compile/channels-last throughput reached 39,558 steps but missed the original baseline at 91.48%, suggesting precision changes can harm this small CNN.
- EXP-002 showed FP32 compile plus channels-last reached 43,398 steps and improved to 91.95%; this is the only validated pattern so far and should be preserved.
- EXP-003 showed the second LR drop at 40k was too early, reaching 91.85% despite 45,279 steps; avoid another early second-drop variant without stronger evidence.
- Gap: no experiment has tested weight averaging, isolated mild augmentation, or higher-capacity architecture on top of the successful FP32 throughput setup.

## Candidate Ideas

### 1. EMA Evaluation Weights on the FP32 Throughput Baseline
**Summary**: Keep the EXP-002 architecture, optimizer, LR milestones, augmentation, and FP32 throughput flags unchanged, but maintain an exponential moving average copy of the model with `torch.optim.swa_utils.AveragedModel`. Update EMA after optimizer steps and run the once-per-epoch evaluation on the EMA weights instead of the instantaneous training weights.

**Reasoning**: EXP-002 already produces a strong, stable training trajectory near the target. EMA targets the evaluated weights directly by smoothing noisy SGD updates without changing data semantics, optimizer updates, precision, or the benchmark harness. The change is dependency-free in local PyTorch and has a plausible chance to add the required +0.10 points because the current best is close to the expected ceiling for ResNet-20 but still exhibits epoch-to-epoch variation.

**Sources**: PyTorch EMA reference `knowledge/references/pytorch-ema-averaging.md`; EXP-002 report `reports/exp-report-002.md`; local support check for `AveragedModel` and `get_ema_multi_avg_fn`.

**Estimated Effort**: low

**Risk Assessment**: EMA can lag behind the online model if decay is too high early in training, and averaging BatchNorm buffers may be imperfect. The failure mode should be clean no-improvement, with little risk of timeout or invalid benchmark changes.

### 2. Mild Cutout-Only on the FP32 Throughput Baseline
**Summary**: Add only a mild tensor-space cutout-style transform, such as `RandomErasing` after `ToTensor()` and before normalization, while preserving the EXP-002 optimizer, LR schedule, model, and FP32 throughput setup.

**Reasoning**: Cutout has direct CIFAR evidence and was not isolated in prior experiments. EXP-000 bundled cutout with label smoothing, Nesterov, and slow cosine, so the failure does not rule out a calibrated cutout-only test. A milder probability or area could improve generalization without reproducing the undertraining pattern.

**Sources**: Cutout note `knowledge/papers/cutout-cifar-regularization.md`; EXP-000 report `reports/exp-report-000.md`; EXP-002 throughput baseline.

**Estimated Effort**: low

**Risk Assessment**: Regularization may still slow convergence under the fixed time budget, and CPU transform overhead may reduce useful steps slightly. A cautious parameterization is needed to avoid repeating EXP-000's undertraining.

### 3. Compact WRN-Style Capacity Increase
**Summary**: Replace ResNet-20 with a compact WRN-style variant, such as a 16-layer width-2 network, while preserving FP32 compile/channels-last. Retune LR milestones to the observed lower step budget if implementation proceeds.

**Reasoning**: WRN-style widening has the highest architecture-level ceiling among the candidates and may be necessary if recipe changes plateau. The H20 has ample VRAM, and the task allows architecture changes in `train.py`.

**Sources**: WRN note `knowledge/papers/wide-residual-networks.md`; TASK.md architecture allowance; EXP-002 FP32 throughput result.

**Estimated Effort**: medium

**Risk Assessment**: More capacity will reduce optimizer steps substantially, making a single 300 second run sensitive to schedule choice. Undertraining or compile overhead could erase the architecture benefit.

## Idea Evaluation

EMA has the best balance for EXP-004. It is supported by an official PyTorch API, requires no dependency or harness changes, preserves the only successful FP32 throughput pattern, and targets the remaining noise between adjacent epoch evaluations. Its expected impact is smaller than an architecture change, but the new +0.10 point threshold is still within the plausible range for averaged weights on a model already peaking near the target.

Mild cutout-only is also attractive because CIFAR regularization evidence is strong, but EXP-000 shows this code path can undertrain if regularization is too strong or combined with a slow schedule. It may be a good next experiment if EMA fails. Compact WRN has the highest theoretical ceiling, but its runtime and schedule uncertainty make it less suitable before exhausting low-overhead improvements on the validated baseline.

## Chosen Idea
**Selected**: EMA Evaluation Weights on the FP32 Throughput Baseline

**Why this idea**:
It composes cleanly with EXP-002, avoids the precision and schedule failure modes seen in EXP-001 and EXP-003, and uses a local PyTorch-supported mechanism that can improve generalization without changing the benchmark target or validation cadence.

**Hypothesis**:
Maintaining and evaluating EMA weights with averaged BatchNorm buffers will smooth late-training SGD noise enough for the same FP32 throughput ResNet-20 recipe to reach at least `92.05%` `best_test_acc`, satisfying the updated improvement threshold.
