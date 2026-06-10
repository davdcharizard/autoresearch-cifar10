# Brainstorm EXP-003
**Created**: 2026-06-08
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

<!-- This file is focused on IDEATION only.
     Goal statement, primary metric, direction, hard constraints, and verification criteria
     live in the goal file (see pointer above). Baseline lives in experiment-indices/maximize-cifar10-best-test-accuracy.tsv.
     Do not duplicate those fields here — always point to the source of truth. -->

## Web Search & Literature Review

- **Saved SGDR/cosine schedule note** (`knowledge/papers/sgdr-cosine-schedule.md`)
  Schedule shape matters for fixed-budget CIFAR training. Prior cosine over 64k steps was too slow, but EXP-002 now provides a better observed step budget for retuning step milestones.

- **EXP-002 FP32 throughput report** (`reports/exp-report-002.md`)
  FP32 compile plus channels-last reached 43,398 steps and improved the baseline to 91.95%, but still never reached the original second LR milestone at 48,000.

## Experimental History Review

- Current best is EXP-002: `best_test_acc=91.95%`, baseline commit `6743174`.
- EXP-000 failed because strong regularization plus slow cosine undertrained ResNet-20.
- EXP-001 showed BF16 throughput reached 39,558 steps but missed baseline at 91.48%.
- EXP-002 showed FP32 throughput reached 43,398 steps and improved to 91.95%.
- Gap: the successful EXP-002 run spent late training at LR 0.01 but never entered LR 0.001. A modest second-milestone retune could create low-LR refinement without changing model, augmentation, precision, or evaluation.

## Candidate Ideas

### 1. Earlier Second LR Drop on FP32 Throughput Baseline
**Summary**: Keep the successful EXP-002 FP32 throughput setup unchanged and move only the second LR milestone from 48,000 to around 40,000 steps, leaving the first milestone at 32,000. This lets the 43k-step run spend its last few thousand updates at LR 0.001.

**Reasoning**: EXP-002 improved after reaching the first LR drop but plateaued before the original second drop. Since the final 10% of training produced several evals around 91.5-91.95%, a lower LR phase may refine the model enough to exceed the new 91.95 baseline.

**Sources**: EXP-002 report `reports/exp-report-002.md`; schedule note `knowledge/papers/sgdr-cosine-schedule.md`; local scheduler in `train.py`.

**Estimated Effort**: low

**Risk Assessment**: An earlier second drop may reduce useful learning too soon and lower peak accuracy. The change is narrow and failure is likely a clean no-improvement.

### 2. Enable TF32 Matmul Precision on FP32 Throughput Baseline
**Summary**: Add `torch.set_float32_matmul_precision("high")` before compile to address the Inductor warning from EXP-002 and potentially increase throughput further.

**Reasoning**: EXP-002 emitted a warning that TF32 tensor cores for float32 matmul were available but not enabled. Enabling this may improve speed and possibly steps, while staying closer to FP32 than BF16.

**Sources**: EXP-002 log note in `logs/exp-log-002.md`; PyTorch throughput reference `knowledge/references/pytorch-throughput-tools.md`.

**Estimated Effort**: low

**Risk Assessment**: TF32 changes matmul precision and may affect accuracy. The model is convolution-dominated, so speed benefit may be limited.

### 3. Compact WRN-16-2 With FP32 Throughput
**Summary**: Replace ResNet-20 with a compact WRN-16-2 and keep the successful FP32 throughput setup. Retune milestones to the observed WRN step budget if needed.

**Reasoning**: WRN has a higher ceiling for CIFAR accuracy, and EXP-002 provides a good execution baseline. Architecture changes may be needed to move beyond small recipe gains.

**Sources**: `knowledge/papers/wide-residual-networks.md`; EXP-002 report.

**Estimated Effort**: medium

**Risk Assessment**: More moving parts and lower step count; schedule may need a separate tuning loop. This is better after exploiting the current architecture's schedule.

## Idea Evaluation

Candidate 1 is the best immediate follow-up because it composes directly with the successful EXP-002 code and changes only one scheduler value. It targets a concrete observed gap: the run now reaches 43k steps but the second LR drop remains unreachable. Candidate 2 is also cheap, but TF32 changes precision and may not help this convolution-dominated model. Candidate 3 has higher ceiling but should wait until the current architecture's schedule is tuned.

## Chosen Idea
**Selected**: Earlier Second LR Drop on FP32 Throughput Baseline

**Why this idea**:
It is the narrowest exploitation of the new best result: keep everything that worked in EXP-002 and make the original schedule reachable under the observed 43k-step budget.

**Hypothesis**:
Changing the scheduler milestones from `[32000, 48000]` to `[32000, 40000]` on top of FP32 throughput will add a short LR 0.001 refinement phase and improve `best_test_acc` above the 91.95 baseline.
