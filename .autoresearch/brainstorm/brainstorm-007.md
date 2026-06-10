# Brainstorm EXP-007
**Created**: 2026-06-08
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

<!-- This file is focused on IDEATION only.
     Goal statement, primary metric, direction, hard constraints, and verification criteria
     live in the goal file (see pointer above). Baseline lives in experiment-indices/maximize-cifar10-best-test-accuracy.tsv.
     Do not duplicate those fields here — always point to the source of truth. -->

## Web Search & Literature Review

- **PyTorch throughput tools note** (`knowledge/references/pytorch-throughput-tools.md`)
  Official PyTorch mechanisms such as `torch.compile` and channels-last can improve CNN throughput without changing the benchmark semantics.

- **EXP-002 report** (`reports/exp-report-002.md`)
  The successful FP32 compile/channels-last run emitted a TF32 tensor-core warning and reached 43,398 steps, suggesting another precision-preserving throughput lever remains.

- **EXP-006 report** (`reports/exp-report-006.md`)
  ResNet-32 failed primarily because it reached only 23,642 steps and never hit its first LR drop, making throughput the immediate blocker for capacity experiments.

No new external search was needed; the relevant TF32 signal comes directly from the local PyTorch warning observed in multiple runs and the existing PyTorch throughput knowledge note.

## Experimental History Review

- Current baseline remains EXP-002 at `best_test_acc=91.95%`; the tightened goal requires `>=92.05%`.
- EXP-002 validated FP32 `torch.compile` plus channels-last as the only real improvement so far.
- EXP-003 showed moving the second ResNet-20 LR drop to 40k hurt, so keep the proven `[32000, 48000]` schedule for a narrow throughput test.
- EXP-004/005 show extra per-step mechanisms and regularization do not reliably help.
- EXP-006 shows larger capacity without enough step budget is a poor next move.
- The same non-fatal warning appears in EXP-002/006: TF32 tensor cores are available but not enabled.

## Candidate Ideas

### 1. Enable TF32 on the FP32 Throughput Baseline
**Summary**: Preserve the EXP-002 ResNet-20 recipe and add PyTorch TF32 enablement before model construction/compile. Keep model, optimizer, LR schedule, augmentation, batch size, seed, channels-last, compile, and evaluation cadence unchanged.

**Reasoning**: This is the narrowest untested throughput lever left from the successful baseline. If TF32 improves convolution/matmul throughput enough, the run may get more useful low-LR refinement while preserving the high-level FP32 training recipe and avoiding BF16 autocast's failed behavior. It also directly addresses the warning emitted by `torch.compile`.

**Sources**: `reports/exp-report-002.md`; `reports/exp-report-006.md`; `knowledge/references/pytorch-throughput-tools.md`; local run warnings.

**Estimated Effort**: low

**Risk Assessment**: TF32 changes arithmetic details and may slightly reduce accuracy, so a speedup is not automatically an accuracy gain. The failure mode should be clean no-improvement.

### 2. Time-Budget-Calibrated LR Milestones for ResNet-20
**Summary**: Keep ResNet-20 and all throughput settings unchanged, but choose milestones based on the observed 43k-46k step budget, such as a slightly earlier first drop and no second drop.

**Reasoning**: EXP-006 showed schedule reachability matters, and EXP-002's best came after the first drop. A schedule that spends more of the fixed budget in LR 0.01 could improve peak accuracy.

**Sources**: `reports/exp-report-002.md`; `reports/exp-report-003.md`; `reports/exp-report-006.md`.

**Estimated Effort**: low

**Risk Assessment**: Schedule-only changes are fragile: EXP-003's second-drop retuning already reduced accuracy. This should not lead while TF32 remains untested.

### 3. ResNet-32 with Measured Early LR Drop
**Summary**: Retry ResNet-32 but set the first LR drop around 14k-16k based on EXP-006's observed 23.6k final step budget.

**Reasoning**: EXP-006 did not fully test lower-LR ResNet-32 capacity because it never reached the first drop. A much earlier milestone would test whether the larger model can benefit from refinement.

**Sources**: `reports/exp-report-006.md`; `knowledge/papers/wide-residual-networks.md`.

**Estimated Effort**: low

**Risk Assessment**: This repeats a recently failed direction and introduces a strong schedule confound. It is better after trying TF32 or a measured-step protocol.

## Idea Evaluation

TF32 enablement has the strongest balance of evidence, scope, and safety. It composes directly with the only validated improvement, addresses a repeated local warning, and may improve the fixed-budget step count without adding regularization or changing model capacity. Since the goal now requires +0.10 points, this is not guaranteed, but it is the cleanest remaining throughput experiment.

Schedule retuning is tempting, but EXP-003 already showed schedule changes can hurt even when they reach more LR phases. ResNet-32 with earlier milestones is plausible but would immediately re-enter the capacity path that just missed the LR drop by a wide margin. TF32 should be tested first because it may also make future capacity experiments more feasible.

## Chosen Idea
**Selected**: Enable TF32 on the FP32 Throughput Baseline

**Why this idea**:
It is a narrow, reversible change that directly targets useful step budget while preserving the proven ResNet-20 recipe and avoiding the failed regularization/capacity paths.

**Hypothesis**:
Enabling TF32 tensor-core execution for the FP32 compile/channels-last ResNet-20 baseline will increase useful optimizer steps enough to reach at least `92.05%` `best_test_acc` without the BF16 accuracy loss seen in EXP-001.
