# Brainstorm EXP-002
**Created**: 2026-06-08
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

<!-- This file is focused on IDEATION only.
     Goal statement, primary metric, direction, hard constraints, and verification criteria
     live in the goal file (see pointer above). Baseline lives in experiment-indices/maximize-cifar10-best-test-accuracy.tsv.
     Do not duplicate those fields here — always point to the source of truth. -->

## Web Search & Literature Review

- **PyTorch throughput reference note** (`knowledge/references/pytorch-throughput-tools.md`)
  `torch.compile`, channels-last memory format, and AMP are official PyTorch throughput tools. EXP-001 suggests throughput can increase steps, but BF16 precision may be a confound for accuracy.

- **PyTorch Channels Last Memory Format tutorial** (https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html)
  Channels-last is relevant to convolution throughput without directly changing arithmetic precision or the training target.

- **PyTorch `torch.compile` documentation** (https://docs.pytorch.org/docs/stable/generated/torch.compile.html)
  `torch.compile` can optimize static model execution while preserving FP32 arithmetic if autocast is disabled.

## Experimental History Review

- Baseline remains 91.52% `best_test_acc`.
- EXP-000 combined strong regularization and slow cosine decay, reaching only 90.45%; avoid strong regularization bundles for now.
- EXP-001 increased throughput to 39,558 steps and 102 epochs, versus EXP-000's 35,279 steps and 91 epochs, but finished at 91.48%, just 0.04 points below baseline.
- EXP-001 indicates throughput is promising, but the BF16 autocast component may have introduced a small accuracy penalty. The next test should isolate precision-preserving speedups.

## Candidate Ideas

### 1. FP32 Throughput Without AMP
**Summary**: Repeat the throughput experiment but disable BF16 autocast. Keep cuDNN benchmark, channels-last model/input layout, and `torch.compile` enabled. This preserves FP32 forward/loss arithmetic while testing whether non-AMP speedups alone can add enough optimizer steps to beat the baseline.

**Reasoning**: EXP-001 came within 0.04 points of baseline while changing both throughput and arithmetic precision. Removing BF16 is the cleanest way to test whether the near miss was a precision issue. Even a smaller step-count improvement could win if FP32 restores the last fraction of accuracy.

**Sources**: EXP-001 report `reports/exp-report-001.md`; reference note `knowledge/references/pytorch-throughput-tools.md`; local `train.py`.

**Estimated Effort**: low

**Risk Assessment**: Without BF16, speedup may shrink enough that step count returns near baseline. `torch.compile` can still add startup overhead or runtime compiler failure, but EXP-001 showed compile worked in this code path.

### 2. EXP-001 Plus Earlier Second LR Drop
**Summary**: Keep the EXP-001 throughput bundle and move the second LR milestone earlier, such as `[32000, 38000]`, so the faster run enters the 0.001 LR phase before the time budget ends.

**Reasoning**: EXP-001 completed 39,558 steps, crossing the first baseline milestone but not the second. It peaked just below baseline, so a short final low-LR refinement phase could provide the missing improvement.

**Sources**: EXP-001 report `reports/exp-report-001.md`; saved SGDR/cosine note `knowledge/papers/sgdr-cosine-schedule.md`.

**Estimated Effort**: low

**Risk Assessment**: This changes both throughput and optimizer schedule, making the result harder to attribute. Earlier second drop could also reduce late learning too aggressively.

### 3. Nesterov-Only Baseline
**Summary**: Keep all baseline settings unchanged except enabling `nesterov=True` in SGD. This tests a minimal optimizer tweak without throughput, precision, augmentation, or schedule confounds.

**Reasoning**: Nesterov was bundled into the failed EXP-000 recipe, so its isolated effect remains unknown. It is a one-line low-risk test that may produce a small accuracy gain.

**Sources**: EXP-000 report `reports/exp-report-000.md`; local optimizer in `train.py`.

**Estimated Effort**: low

**Risk Assessment**: Expected impact is small; it may land within run noise and fail to exceed baseline.

## Idea Evaluation

Candidate 1 is the cleanest next experiment. It directly follows the evidence from EXP-001, removes the most likely metric-harming confound, and preserves the useful throughput hypothesis. Because EXP-001 already proved the compile/channels-last path runs cleanly, this experiment has lower execution risk than a brand-new architecture or schedule change.

Candidate 2 may be more aggressive and could exploit the higher step count, but it changes the schedule before isolating whether BF16 was the problem. Candidate 3 is clean but likely too low impact compared with the near miss from EXP-001.

## Chosen Idea
**Selected**: FP32 Throughput Without AMP

**Why this idea**:
EXP-001 missed baseline by only 0.04 points while increasing steps, so the highest-signal next move is to preserve FP32 numerics and keep the non-AMP throughput mechanisms that already proved executable.

**Hypothesis**:
Disabling BF16 autocast while retaining cuDNN benchmarking, channels-last layout, and `torch.compile` will recover the small accuracy gap from EXP-001; if step count remains meaningfully above baseline, `best_test_acc` will exceed 91.52%.
