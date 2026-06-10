# Brainstorm EXP-030
**Created**: 2026-06-08
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

<!-- This file is focused on IDEATION only.
     Goal statement, primary metric, direction, hard constraints, and verification criteria
     live in the goal file (see pointer above). Baseline lives in experiment-indices/maximize-cifar10-best-test-accuracy.tsv.
     Do not duplicate those fields here — always point to the source of truth. -->

## Web Search & Literature Review

- **Torchvision RandomCrop padding reference** (`knowledge/references/torchvision-randomcrop-padding.md`)
  The current anchor now uses validated reflection padding; sibling boundary-fill variants remain possible, but reflection should be preserved unless directly testing a padding-mode replacement.

- **SGDR / cosine schedule note** (`knowledge/papers/sgdr-cosine-schedule.md`)
  Learning-rate schedule shape can materially affect CIFAR optimization, but previous broad schedule replacements were too slow or undercalibrated locally. Narrow step-schedule retuning around a validated anchor is lower risk.

- **PyTorch EMA Weight Averaging** (`knowledge/references/pytorch-ema-averaging.md`)
  Averaging remains a possible late-refinement lever, but local EMA/equal-averaging failures make it riskier than schedule retuning that preserves the training loop.

## Experimental History Review

- Current baseline is EXP-029 at `best_test_acc=93.58%`; under the goal's +0.10 percentage-point rule, EXP-030 must reach at least `93.68%`.
- The current anchor is `STAGE_WIDTHS = (28, 56, 112)`, reflected `RandomCrop`, `BATCH_SIZE = 128`, `LR = 0.1`, `MOMENTUM = 0.9`, `WEIGHT_DECAY = 1e-4`, `LR_MILESTONES = [21000, 64000]`, FP32, channels-last, and `torch.compile`.
- EXP-029 crossed the threshold in the LR 0.01 phase, peaking at epoch 74 and finishing lower. This suggests late refinement may be the next bottleneck after the reflection-padding gain.
- Schedule-only work around the old anchor was mostly bounded: 20k and 23k first drops underperformed 21k, and a 36k second drop reached only 93.13%. However, those failures predate reflection padding, which changed the validation trajectory and new baseline.
- Width expansion beyond 28/56/112 is a high-importance recurring failure. Optimizer and regularization perturbations also have weak local evidence.
- The next experiment should preserve reflection padding and the 28/56/112 architecture, then test a minimal coupled schedule adjustment that becomes plausible only because EXP-029 changed the late-accuracy trajectory.

## Candidate Ideas

### 1. Reflection Anchor With 32k Second LR Drop
**Summary**: Change `LR_MILESTONES` from `[21000, 64000]` to `[21000, 32000]` while preserving reflection padding and every other anchor setting.

**Reasoning**: EXP-029 peaked at 93.58% around the middle of the LR 0.01 phase and then oscillated, finishing at 93.35%. A second drop at 32k should be reachable within the ~43k-step budget and gives roughly 11k lower-LR refinement steps without changing capacity, augmentation, optimizer, or validation cadence. This is not a repeat of EXP-024 because the baseline augmentation has changed and the peak now occurs earlier/higher.

**Sources**: EXP-029 report; EXP-024; `goal-learnings/maximize-cifar10-best-test-accuracy.md`; `train.py` `LR_MILESTONES`.

**Estimated Effort**: low

**Risk Assessment**: Prior second-drop tuning failed on the old anchor, so this may simply reduce exploration or overfit a late plateau. The failure mode is a valid no-improvement run with preserved throughput.

### 2. Symmetric Padding for RandomCrop
**Summary**: Replace `padding_mode="reflect"` with `padding_mode="symmetric"` while keeping the rest of the EXP-029 anchor unchanged.

**Reasoning**: EXP-029 validated crop-boundary statistics as a meaningful lever. Symmetric padding is a direct sibling experiment that may preserve edge continuity differently from reflection without adding overhead.

**Sources**: `knowledge/references/torchvision-randomcrop-padding.md`; EXP-029 report.

**Estimated Effort**: low

**Risk Assessment**: Reflection is already validated, and symmetric padding may be worse or equivalent. Since the new threshold is 93.68%, a sibling padding-mode replacement may have too small an expected gain.

### 3. Low-Frequency Late EMA on Reflection Anchor
**Summary**: Add an EMA model only after the first LR drop and update it sparsely, evaluating the EMA once per epoch.

**Reasoning**: EXP-029's late oscillation suggests weight averaging could smooth the final solution. Starting after the first LR drop and updating sparsely addresses the overhead and collapse modes seen in EXP-004 and EXP-021.

**Sources**: `knowledge/references/pytorch-ema-averaging.md`; EXP-004; EXP-021; EXP-029 report.

**Estimated Effort**: medium

**Risk Assessment**: EMA adds implementation complexity and can still reduce step count or mishandle BatchNorm buffers. It is a plausible later test but less isolated than a schedule change.

## Idea Evaluation

The second LR drop has the strongest fit to the immediate evidence from EXP-029. It preserves the newly validated reflection-padding anchor and targets the observed behavior: peak accuracy occurs well before the end of the run, while later epochs oscillate around but below the best. A reachable second drop at 32k is more aggressive than EXP-024's old-anchor 36k drop, but it is motivated by a new trajectory rather than retrying the old schedule in isolation.

Symmetric padding is attractive because EXP-029 proved padding mode matters, but replacing a validated improvement has a smaller expected gain and a higher chance of just undoing part of the new baseline. Late EMA could exploit the same oscillation signal, but previous averaging attempts make it more operationally risky. EXP-030 should use the simpler schedule-retuning probe first.

## Chosen Idea
**Selected**: Reflection Anchor With 32k Second LR Drop

**Why this idea**:
It is a narrowly scoped, one-line schedule change that preserves the successful reflection-padding anchor while directly targeting the new late-refinement bottleneck seen in EXP-029.

**Hypothesis**:
Adding a reachable second LR drop at step 32000 will convert EXP-029's late LR 0.01 oscillation into steadier low-LR refinement, improving `best_test_acc` from 93.58% to at least `93.68%`.
