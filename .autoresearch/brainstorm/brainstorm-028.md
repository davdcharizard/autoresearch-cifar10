# Brainstorm EXP-028
**Created**: 2026-06-08
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

<!-- This file is focused on IDEATION only.
     Goal statement, primary metric, direction, hard constraints, and verification criteria
     live in the goal file (see pointer above). Baseline lives in experiment-indices/maximize-cifar10-best-test-accuracy.tsv.
     Do not duplicate those fields here — always point to the source of truth. -->

## Web Search & Literature Review

- **Torchvision ResNet source** (https://docs.pytorch.org/vision/master/_modules/torchvision/models/resnet.html)
  Torchvision exposes a `zero_init_residual` option that initializes the final BatchNorm in each residual branch to zero, making residual blocks start closer to identity; the source notes a reported small accuracy gain.

- **Identity Mappings in Deep Residual Networks** (https://arxiv.org/abs/1603.05027)
  He et al. argue that residual networks train and generalize better when shortcut and residual-unit design preserve direct signal propagation, motivating identity-preserving residual initialization changes.

- **Batch Normalization Biases Residual Blocks Towards the Identity Function in Deep Networks** (https://arxiv.org/abs/2002.10444)
  De and Smith analyze why residual branches near identity at initialization improve trainability, giving a mechanism for adjusting residual branch scale without changing architecture.

## Experimental History Review

- Current baseline is EXP-016 at `best_test_acc=93.23%`; under the goal's +0.10 percentage-point rule, EXP-028 must reach at least `93.33%`.
- The current anchor is `STAGE_WIDTHS = (28, 56, 112)`, `BATCH_SIZE = 128`, `LR = 0.1`, `MOMENTUM = 0.9`, `WEIGHT_DECAY = 1e-4`, and `LR_MILESTONES = [21000, 64000]`.
- Local evidence says further widening beyond 28/56/112 is a high-importance recurring failure; proportional, minimal, and final-stage-only width increases have all missed threshold.
- Schedule-only brackets around the anchor are likely exhausted: 20k and 23k first drops underperform 21k, and a reachable second drop at 36k reached only 93.13%.
- Regularization and optimizer perturbations have mostly hurt: cutout variants, lower global weight decay, no-decay BatchNorm/bias groups, Nesterov, and momentum 0.95 all missed baseline.
- The validated FP32 compile/channels-last path should be preserved; BF16 and TF32 variants underperformed.
- A distinct initialization change has not yet been tested. It can preserve architecture, parameter count, schedule, batch size, optimizer, preprocessing, and validation cadence while changing early residual dynamics.

## Candidate Ideas

### 1. Zero-Initialize Residual Branch Last BatchNorm
**Summary**: Initialize every `BasicBlock.bn2.weight` to zero after the existing Kaiming initialization so each residual branch initially contributes near zero and each block starts closer to an identity mapping.

**Reasoning**: This directly targets residual optimization rather than capacity, schedule, augmentation, or optimizer regularization. Torchvision's ResNet implementation includes this exact option for `BasicBlock` residual branches, and the residual-identity literature gives a clear mechanism: early training can benefit when shortcut paths carry stable signals while residual branches learn perturbations. The change has no expected throughput penalty and keeps the successful 28/56/112 anchor intact.

**Sources**: Torchvision ResNet source; arXiv:1603.05027; arXiv:2002.10444; `train.py` `BasicBlock` and `ResNet._weights_init`; EXP-016, EXP-027.

**Estimated Effort**: low

**Risk Assessment**: Zero residual branches may slow early feature learning under the 300s time budget. The worst case is a valid no-improvement run with healthy throughput but lower early accuracy.

### 2. Reflection Padding for RandomCrop
**Summary**: Change `transforms.RandomCrop(32, padding=4)` to use reflection padding while preserving crop size, padding amount, flip augmentation, normalization, and evaluation harness.

**Reasoning**: Reflection padding could remove artificial zero-border artifacts without adding a stronger regularizer like cutout. It is a small augmentation-quality change that preserves throughput and does not touch `prepare.py`.

**Sources**: `train.py` transform definition; `goal-learnings/maximize-cifar10-best-test-accuracy.md` cutout failures.

**Estimated Effort**: low

**Risk Assessment**: The expected gain is likely smaller than the +0.10 point threshold, and augmentation changes have a weak local track record. It may also perturb a tuned baseline in the wrong direction.

### 3. Low-Frequency Late EMA
**Summary**: Maintain an EMA model only after the first LR drop and update it at a low frequency, then evaluate that EMA once per epoch instead of paying per-step overhead throughout training.

**Reasoning**: EXP-004 showed that per-step EMA reached a tiny gain but lost too many steps. A late, sparse EMA targets the same averaging benefit while reducing overhead and avoiding the long equal-average collapse seen in EXP-021.

**Sources**: `knowledge/references/pytorch-ema-averaging.md`; EXP-004; EXP-021; `goal-learnings/maximize-cifar10-best-test-accuracy.md`.

**Estimated Effort**: medium

**Risk Assessment**: The implementation is more complex, can still cost throughput, and risks stale BatchNorm statistics. It should be tried only after lower-risk initialization or augmentation probes.

## Idea Evaluation

Zero-initializing `bn2.weight` has the strongest combination of external evidence, mechanism clarity, and local fit. It is a standard residual-network option, has a direct identity-preserving rationale, and avoids the classes that have repeatedly failed locally: extra capacity, first-drop schedule brackets, stronger augmentation, and optimizer retuning. Its failure mode is also clean: if the fixed 300s budget cannot exploit the initialization, the result should simply classify as no-improvement.

Reflection padding is easy but has weaker expected impact and lives in augmentation space, where prior cutout experiments suggest the current recipe does not need more or different regularization. Low-frequency late EMA is mechanistically plausible, but it is more invasive and still inherits risks from EXP-004 and EXP-021. Given the current threshold, the next experiment should prefer the no-throughput-cost initialization lever.

## Chosen Idea
**Selected**: Zero-Initialize Residual Branch Last BatchNorm

**Why this idea**:
It is a narrowly scoped initialization change with published and reference-implementation support, no expected parameter-count or throughput penalty, and a mechanism distinct from the failed schedule, width, regularization, and optimizer variants.

**Hypothesis**:
Setting each residual block's final BatchNorm scale to zero will make the 28/56/112 ResNet-20 start closer to a stack of identity mappings, improving early optimization stability and post-drop refinement enough to reach at least `93.33%` best test accuracy.
