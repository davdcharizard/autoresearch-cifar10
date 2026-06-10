# Brainstorm EXP-027
**Created**: 2026-06-08
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

<!-- This file is focused on IDEATION only.
     Goal statement, primary metric, direction, hard constraints, and verification criteria
     live in the goal file (see pointer above). Baseline lives in experiment-indices/maximize-cifar10-best-test-accuracy.tsv.
     Do not duplicate those fields here — always point to the source of truth. -->

## Web Search & Literature Review

- **Wide Residual Networks** (`knowledge/papers/wide-residual-networks.md`)
  CIFAR capacity scaling remains useful background, but local evidence now treats width beyond 28/56/112 as a recurring high-importance failure.

- **SGDR / cosine scheduling** (`knowledge/papers/sgdr-cosine-schedule.md`)
  Smooth schedules remain possible later, but local step-schedule variants around the current anchor have repeatedly missed the 93.33% threshold.

- **PyTorch throughput tools** (`knowledge/references/pytorch-throughput-tools.md`)
  The validated FP32 compile/channels-last path should be preserved; recent failures that preserved throughput still clarify which non-throughput levers are weak.

No new external search was needed. A candidate standard CIFAR channel-std normalization change was considered, but `prepare.py` fixes evaluator preprocessing to mean-only normalization with `std=(1, 1, 1)`, so changing only train-time std would create a train/eval distribution mismatch.

## Experimental History Review

- Current baseline is EXP-016 at `best_test_acc=93.23%`; with the goal's +0.10 percentage-point rule, EXP-027 must reach at least `93.33%`.
- The current best recipe is `STAGE_WIDTHS = (28, 56, 112)`, `BATCH_SIZE = 128`, `LR = 0.1`, `MOMENTUM = 0.9`, `WEIGHT_DECAY = 1e-4`, and `LR_MILESTONES = [21000, 64000]`.
- Width above 28/56/112 is a recurring high-importance failure; further capacity increases should be deprioritized.
- Schedule-only changes around the current anchor are bounded: first-drop 20k and 23k underperform 21k, and the reachable 36k second drop remains below threshold.
- Aggressive smaller batch size reduced useful step budget, and momentum 0.95 worsened post-drop refinement to 92.90%.
- Lower global weight decay (`5e-5`) hurt the anchor, but a targeted optimizer param-group change that keeps `1e-4` on conv/linear weights while excluding BatchNorm and bias parameters remains untested.
- Cutout-style masking is a recurring regularization failure, but not all regularization-adjacent changes are equivalent; optimizer weight-decay targeting is mechanistically different from adding augmentation or reducing global decay.

## Candidate Ideas

### 1. Exclude BatchNorm and Bias from Weight Decay
**Summary**: Keep `WEIGHT_DECAY = 1e-4` for convolution and linear weights, but pass BatchNorm parameters and bias parameters to SGD with `weight_decay=0.0`.

**Reasoning**: The current optimizer decays every parameter, including BatchNorm scale/shift and the final classifier bias. A common modern CNN practice is to decay only weight tensors that benefit from L2 regularization while leaving normalization and bias terms unregularized. This differs from EXP-023 because it does not lower regularization on the main conv/linear weights; it only removes decay from parameters where weight decay can distort normalization statistics or bias calibration.

**Sources**: `train.py` optimizer setup; `experiment-indices/maximize-cifar10-best-test-accuracy.tsv` EXP-016, EXP-023, EXP-026; `goal-learnings/maximize-cifar10-best-test-accuracy.md`.

**Estimated Effort**: medium

**Risk Assessment**: The effect may be small, and removing BN/bias decay could behave like weaker regularization if the anchor needs those constraints. Implementation must avoid changing architecture, schedule, or validation cadence.

### 2. Zero-Initialize Residual Branch Last BatchNorm
**Summary**: Initialize each residual block's second BatchNorm scale (`bn2.weight`) to zero so residual branches start as identity perturbations.

**Reasoning**: Zero-initialized residual branches can stabilize deep residual optimization by making each block initially close to identity. This changes initialization rather than architecture or schedule and has no expected throughput penalty. It is distinct from projection shortcuts, width scaling, and momentum changes.

**Sources**: `train.py` `BasicBlock` and initialization path; local architecture failures in the experiment index.

**Estimated Effort**: medium

**Risk Assessment**: Under a strict 300s budget, starting residual branches at zero may slow early feature learning and reduce peak accuracy. It may require longer training than this harness allows.

### 3. Reflection Padding for RandomCrop
**Summary**: Change training augmentation from zero-padded `transforms.RandomCrop(32, padding=4)` to reflection-padded random crop.

**Reasoning**: Reflection padding can reduce artificial black-border artifacts while preserving the same crop/flip augmentation class and evaluation harness. It is a lightweight augmentation-quality change rather than a stronger regularizer like cutout.

**Sources**: `train.py` transform definition; cutout failures in `goal-learnings/maximize-cifar10-best-test-accuracy.md`.

**Estimated Effort**: low

**Risk Assessment**: The expected effect is likely small and could be negative if the baseline's zero padding is part of the tuned recipe. It also changes data augmentation, a class where prior stronger changes have mostly hurt.

## Idea Evaluation

Excluding BatchNorm and bias from weight decay is the strongest next experiment because it targets a precise optimizer regularization mechanism while preserving the successful anchor, throughput path, batch size, schedule, and main conv/linear decay strength. It also directly addresses a gap left by EXP-023: global lower weight decay failed, but targeted no-decay groups have not been tested.

Zero-initialized residual branches are plausible but risk slowing learning under the fixed time budget. Reflection padding is simple but has weaker expected impact and sits closer to the augmentation space where cutout variants failed. The train-time channel-std normalization idea is not selected because the evaluator's fixed mean-only preprocessing would make the distribution mismatch hard to interpret.

## Chosen Idea
**Selected**: Exclude BatchNorm and Bias from Weight Decay

**Why this idea**:
It is a targeted optimizer regularization change that keeps the validated architecture, schedule, and `1e-4` decay on main weights intact. It is meaningfully different from lowering global weight decay and avoids the throughput penalties seen in capacity, batch-size, and momentum experiments.

**Hypothesis**:
Creating SGD parameter groups with `weight_decay=1e-4` for convolution/linear weights and `weight_decay=0.0` for BatchNorm and bias parameters will improve calibration/generalization enough to raise `best_test_acc` to at least 93.33% while respecting all hard constraints.
