# Brainstorm EXP-036
**Created**: 2026-06-09
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

<!-- This file is focused on IDEATION only.
     Goal statement, primary metric, direction, hard constraints, and verification criteria
     live in the goal file (see pointer above). Baseline lives in experiment-indices/maximize-cifar10-best-test-accuracy.tsv.
     Do not duplicate those fields here — always point to the source of truth. -->

## Web Search & Literature Review

- **Wide Residual Networks** (`knowledge/papers/wide-residual-networks.md`)
  Wider CIFAR residual networks can improve accuracy, but local evidence now shows width beyond 28/56/112 loses too much useful budget under this harness.
- **PyTorch EMA averaging** (`knowledge/references/pytorch-ema-averaging.md`)
  Averaging can stabilize evaluated weights, but earlier project evidence showed naive or per-step averaging can add overhead or collapse without careful bounding.
- **PyTorch throughput tools** (`knowledge/references/pytorch-throughput-tools.md`)
  Throughput-preserving changes matter because the fixed 300s budget makes step coverage part of the accuracy mechanism.

## Experimental History Review

- Current baseline is EXP-032 at `best_test_acc=93.70%`; under the +0.10 percentage-point rule, EXP-036 must reach at least `93.80%`.
- The current anchor is `STAGE_WIDTHS = (28, 56, 112)`, reflected `RandomCrop`, `label_smoothing=0.05`, `BATCH_SIZE = 128`, `LR_MILESTONES = [21000, 64000]`, FP32, channels-last, cuDNN benchmark, and `torch.compile`.
- EXP-033 and EXP-034 each reached 93.79% as one-scalar near-misses, but EXP-035 combined them and regressed to 93.63%, so adjacent smoothing/schedule tweaks should not be assumed additive.
- EXP-025 showed batch size 96 was too aggressive: it completed only 32,996 steps, missed its planned second drop, and peaked below threshold. However, a milder batch size 112 remains untested on the newer reflection plus label-smoothing anchor.
- High-importance avoid signals still apply: do not widen beyond 28/56/112 and do not run isolated second-drop retuning. Cutout, projection shortcuts, zero-gamma initialization, higher momentum, lower weight decay, no-decay BN/bias, and padding siblings have also underperformed.
- The remaining plausible low-scope space is a distinct stochasticity adjustment, stronger smoothing, or a carefully bounded late-stability mechanism.

## Candidate Ideas

### 1. Mild Batch Size 112 on the Current Anchor
**Summary**: Preserve the current reflection-padding, label-smoothed 28/56/112 anchor and change only `BATCH_SIZE` from 128 to 112. Keep `LR_MILESTONES = [21000, 64000]` so the first drop remains reachable and the second milestone remains intentionally unreachable unless throughput unexpectedly improves.

**Reasoning**: This tests a distinct stochasticity lever after EXP-035 showed local smoothing/schedule near-misses are not additive. Batch size 112 is a milder version of the failed batch-size-96 idea and should be less damaging to step throughput. If the current anchor benefits from slightly noisier updates or more optimizer steps per data pass, it could improve late peak accuracy without changing architecture, augmentation, optimizer, label smoothing, or evaluation cadence.

**Sources**: `reports/exp-report-025.md`; `reports/exp-report-032.md`; `reports/exp-report-035.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md` § Failed Approaches and Patterns; current `train.py`.

**Estimated Effort**: low

**Risk Assessment**: The throughput penalty may still dominate, especially because smaller batches reduce images processed per step. If the run loses too many steps or epochs, it may underperform like EXP-025. The clean failure mode is a valid no-improvement result.

### 2. Raise Label Smoothing to 0.08
**Summary**: Preserve the current anchor and change only the training loss from `label_smoothing=0.05` to `label_smoothing=0.08`.

**Reasoning**: EXP-032 validated mild label smoothing as a no-throughput regularizer. Stronger smoothing could further reduce overconfident late updates and improve generalization if 0.05 is not yet the optimum. This is simple, isolated, and has no schedule or throughput interaction.

**Sources**: `reports/exp-report-032.md`; `reports/exp-report-033.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md` § Patterns.

**Estimated Effort**: low

**Risk Assessment**: Lower smoothing to 0.03 was closer to the threshold than the 0.05 baseline, so increasing smoothing may suppress top-1 peak accuracy. It also resembles the stronger-regularization direction that failed in the combined EXP-000 recipe.

### 3. Short-Window Late Weight Averaging
**Summary**: Preserve the current anchor and add a tightly bounded late weight averaging mechanism that only averages a short post-drop window, then evaluates the averaged model once per epoch.

**Reasoning**: Weight averaging can stabilize late evaluations, and EXP-032's label-smoothed anchor already has good late behavior. The failure of EXP-021 was specifically naive long equal averaging, so a short window or decay-based variant could avoid snapshot accumulation collapse.

**Sources**: `knowledge/references/pytorch-ema-averaging.md`; `reports/exp-report-021.md`; `reports/exp-report-032.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md` § Failed Approaches.

**Estimated Effort**: medium

**Risk Assessment**: This is more code and overhead than a scalar change. Prior EMA and averaging attempts missed thresholds or collapsed, so the idea needs careful implementation to avoid repeating known failure modes.

## Idea Evaluation

Batch size 112 is the best next probe because it is distinct from the now-unproductive smoothing/schedule micro-space while remaining narrow and easy to verify. EXP-025 rejected aggressive batch size 96, but it did not test a milder value on the stronger reflection plus label-smoothing anchor. The causal mechanism is clear: adjust update noise and data-pass geometry while checking whether the 21k first drop remains reachable.

Raising smoothing to 0.08 is simpler, but evidence is weaker because lower smoothing already produced the best smoothing-only near-miss and stronger smoothing risks suppressing top-1 accuracy. Short-window averaging is plausible but more complex and sits near two failed averaging results; it is better saved until low-risk scalar and stochasticity probes are exhausted.

EXP-036 should therefore test `BATCH_SIZE=112` only, preserving all other anchor choices and explicitly measuring step budget, epoch count, first-drop reachability, and whether the metric reaches 93.80%.

## Chosen Idea
**Selected**: Mild Batch Size 112 on the Current Anchor

**Why this idea**:
It moves to a distinct, still-local lever after EXP-035 showed nearby smoothing/schedule gains do not compose. It is less aggressive than the failed batch-size-96 run and can be implemented as a single scalar change while preserving the current architecture, augmentation, label smoothing, schedule, optimizer, and evaluation harness.

**Hypothesis**:
Changing `BATCH_SIZE` from 128 to 112 will add enough beneficial update stochasticity on the reflection plus label-smoothing anchor to raise `best_test_acc` from 93.70% to at least 93.80%, while still reaching the step-21000 first LR drop within the fixed budget.
