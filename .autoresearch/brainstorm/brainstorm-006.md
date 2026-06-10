# Brainstorm EXP-006
**Created**: 2026-06-08
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

<!-- This file is focused on IDEATION only.
     Goal statement, primary metric, direction, hard constraints, and verification criteria
     live in the goal file (see pointer above). Baseline lives in experiment-indices/maximize-cifar10-best-test-accuracy.tsv.
     Do not duplicate those fields here — always point to the source of truth. -->

## Web Search & Literature Review

- **Wide Residual Networks note** (`knowledge/papers/wide-residual-networks.md`)
  Widening or otherwise increasing compact CIFAR residual capacity can improve accuracy, but fixed-budget runtime needs careful schedule checks.

- **PyTorch throughput tools note** (`knowledge/references/pytorch-throughput-tools.md`)
  The validated baseline depends on precision-preserving `torch.compile` and channels-last; new architecture experiments should preserve this path unless the hypothesis requires changing it.

- **SGDR schedule note** (`knowledge/papers/sgdr-cosine-schedule.md`)
  Schedule shape remains a plausible lever, but prior schedule-only changes have been fragile; schedule changes should be tied to a concrete step-budget reason.

- **EXP-005 cutout report** (`reports/exp-report-005.md`)
  Isolated 16x16 cutout preserved optimizer steps but delayed useful fitting, so the next experiment should avoid stronger regularization and target model capacity or optimization horizon instead.

No new external search was needed; the existing knowledge base already covers the relevant CIFAR architecture, throughput, and schedule options.

## Experimental History Review

- Current baseline remains EXP-002 at `best_test_acc=91.95%`, commit `6743174`; the tightened goal requires `>=92.05%` to count.
- EXP-002 is the only successful post-baseline intervention: FP32 `torch.compile`, channels-last, and cuDNN benchmarking improved step budget and reached 91.95%.
- EXP-003 showed moving the second LR drop from 48k to 40k hurt, so pure milestone retuning is not clearly productive on ResNet-20.
- EXP-004 showed per-step EMA overhead is too expensive for too little gain.
- EXP-005 showed isolated 16x16 cutout is not just a throughput problem: it reached 46,238 steps but only 91.72%, indicating over-regularization/late fitting.
- Gap: no successful experiment has changed the model's capacity. The current ResNet-20 may be near its recipe ceiling under the fixed budget.

## Candidate Ideas

### 1. Schedule-Calibrated ResNet-32 Capacity Increase
**Summary**: Increase CIFAR ResNet depth from ResNet-20 to ResNet-32 by changing `NUM_BLOCKS` from 3 to 5, preserve FP32 compile/channels-last, and adjust LR milestones earlier enough that the larger model still gets a meaningful LR 0.01 refinement phase within the 300 second budget.

**Reasoning**: Prior recipe-only changes are now producing small or negative effects. A modest depth increase raises representational capacity without introducing a new architecture family or dependency. Because the larger model will be slower per step, the schedule must be calibrated to the expected lower step count rather than reusing ResNet-20's `[32000, 48000]` milestones blindly. This is a higher-upside experiment that still keeps changes localized and interpretable.

**Sources**: `knowledge/papers/wide-residual-networks.md`; `reports/exp-report-002.md`; `reports/exp-report-005.md`; current `train.py`.

**Estimated Effort**: medium

**Risk Assessment**: The larger model may fail to reach enough low-LR refinement under the time budget, or the schedule adjustment may confound whether capacity or LR timing caused the result. Failure should still be a clean no-improvement, and the result will reveal whether capacity is worth deeper exploration.

### 2. TF32 Throughput Enablement on the FP32 Baseline
**Summary**: Enable TensorFloat-32 acceleration for FP32 operations with PyTorch backend flags / matmul precision, while preserving model, optimizer, augmentation, LR schedule, batch size, seed, compile, and channels-last.

**Reasoning**: EXP-002 startup warned that TF32 tensor cores were available but not enabled. If enabling TF32 materially increases step count while keeping accuracy stable enough, extra low-LR updates could push the best accuracy over the +0.10 threshold. This is a narrow throughput intervention with low code risk.

**Sources**: `reports/exp-report-002.md`; `knowledge/references/pytorch-throughput-tools.md`; current `train.py`.

**Estimated Effort**: low

**Risk Assessment**: The effect may be too small to clear the tightened threshold, and TF32 arithmetic can slightly change numerics. It is less likely than a capacity change to create a >0.10 point gain.

### 3. Observed-Horizon Cosine Decay Without New Regularization
**Summary**: Replace the current two-step `MultiStepLR` with a cosine decay calibrated to the observed 43k-46k step budget, preserving the EXP-002 model, augmentation, optimizer, FP32 compile, and channels-last path.

**Reasoning**: A smoother schedule could improve anytime accuracy near the end of the fixed time budget. EXP-000's cosine result was confounded by cutout, label smoothing, Nesterov, and a 64k horizon; an observed-horizon version would isolate schedule shape.

**Sources**: `knowledge/papers/sgdr-cosine-schedule.md`; `reports/exp-report-000.md`; `reports/exp-report-002.md`; `reports/exp-report-005.md`.

**Estimated Effort**: low

**Risk Assessment**: Schedule-only interventions have already been fragile, and EXP-003's earlier second drop hurt. Cosine may again keep LR mismatched to the peak-accuracy window.

## Idea Evaluation

The schedule-calibrated ResNet-32 experiment has the best expected impact. The current trajectory suggests the ResNet-20 FP32 recipe is approaching a local ceiling: EMA only produced a tiny noisy gain, cutout hurt despite high step count, and milestone retuning alone reduced accuracy. A modest capacity increase addresses a different bottleneck, and the fixed H20 memory budget gives room for a larger CIFAR model.

TF32 enablement is attractive because it is narrow and was explicitly suggested by the EXP-002 warning, but its upside is likely limited to step count. Since EXP-005 already reached more steps than EXP-002 and still missed badly, step count alone is not enough when the intervention does not improve the model's useful fitting behavior. TF32 remains a good later experiment if capacity changes are too slow.

Observed-horizon cosine is the easiest to implement, but the history argues against making schedule-only changes the lead: EXP-003 hurt, and EXP-000's schedule/regularization bundle was far below baseline. It may be worth revisiting after a capacity experiment establishes a new compute/step budget.

## Chosen Idea
**Selected**: Schedule-Calibrated ResNet-32 Capacity Increase

**Why this idea**:
It targets the likely post-recipe ceiling of the current ResNet-20 while preserving the proven FP32 compile/channels-last throughput path. The change has enough plausible effect size to clear the +0.10 point threshold, and its main risk can be mitigated by moving LR milestones earlier to match the larger model's expected step budget.

**Hypothesis**:
A ResNet-32 variant with calibrated LR milestones will improve CIFAR-10 representational capacity enough to reach at least `92.05%` `best_test_acc`, even with fewer optimizer steps than ResNet-20.
