# Brainstorm EXP-000
**Created**: 2026-06-08
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

<!-- This file is focused on IDEATION only.
     Goal statement, primary metric, direction, hard constraints, and verification criteria
     live in the goal file (see pointer above). Baseline lives in experiment-indices/maximize-cifar10-best-test-accuracy.tsv.
     Do not duplicate those fields here — always point to the source of truth. -->

## Web Search & Literature Review

- **Improved Regularization of Convolutional Neural Networks with Cutout** (https://arxiv.org/abs/1708.04552)
  Cutout randomly masks square regions during training, is easy to implement alongside standard crop/flip augmentation, and was evaluated directly on CIFAR-10/CIFAR-100/SVHN with improved generalization. This is highly relevant because the current baseline uses only crop and horizontal flip.

- **Wide Residual Networks** (https://arxiv.org/abs/1605.07146)
  WRNs trade excessive depth for more channels and report better accuracy/efficiency than very deep thin ResNets on CIFAR-style tasks. This suggests widening or replacing the current ResNet-20 could improve `best_test_acc`, but it carries runtime and implementation risk under the fixed 300 second budget.

- **SGDR: Stochastic Gradient Descent with Warm Restarts** (https://arxiv.org/abs/1608.03983)
  SGDR uses cosine annealing with restarts to improve anytime performance for SGD and reports CIFAR-10/CIFAR-100 results. This is relevant because the current step-based schedule is keyed to step counts, while this benchmark is wall-clock limited and evaluated once per epoch.

## Experimental History Review

- Current best is the baseline row only: `best_test_acc=91.52%` at commit `869b7a7`.
- First experiment under this restored goal; there are no prior failed approaches or validated project-specific recipes in `.autoresearch/goal-learnings/maximize-cifar10-best-test-accuracy.md`.
- Codebase gap: `train.py` uses only RandomCrop/RandomHorizontalFlip, no label smoothing, no cutout/random erasing, no mixup, no EMA, and a fixed MultiStepLR schedule.
- Codebase gap: the model is small enough that regularization changes are cheaper and safer than architectural expansion for the first loop.

## Candidate Ideas

### 1. Cheap CIFAR Regularization and Cosine Schedule
**Summary**: Keep the ResNet-20 architecture but strengthen the training recipe: add tensor-space cutout via `transforms.RandomErasing`, add modest label smoothing to cross entropy, switch the step LR schedule to cosine annealing over the expected step budget, and optionally enable Nesterov momentum. This tests whether the baseline is primarily under-regularized and schedule-limited rather than architecture-limited.

**Reasoning**: Cutout is directly supported by CIFAR evidence and is almost free relative to convolution cost. Label smoothing is built into PyTorch cross entropy and can reduce overconfidence without changing the evaluation harness. A cosine schedule is a low-risk optimizer change supported by SGDR-style CIFAR results and better matches a fixed-time run than hard drops at fixed steps. The current baseline has simple augmentation and an abrupt step schedule, so this idea targets clear omissions without slowing training much.

**Sources**: Cutout paper https://arxiv.org/abs/1708.04552; SGDR paper https://arxiv.org/abs/1608.03983; local code `train.py`.

**Estimated Effort**: low

**Risk Assessment**: Cutout or label smoothing can over-regularize a small ResNet-20 under a 300 second budget, reducing early accuracy. Cosine scheduling needs a sensible step horizon; if the actual step count differs substantially from `MAX_STEPS`, the LR may decay too slowly or too quickly. Worst case is a clean no-improvement run.

### 2. Small Wide Residual Network Under the Same Budget
**Summary**: Replace the current ResNet-20 with a compact WRN-style CIFAR model, such as a 16-layer or 22-layer network with width factor 2 and optional dropout. Keep the same data pipeline and optimizer family initially, adjusting batch size only if needed to fit the single-GPU budget.

**Reasoning**: WRN evidence suggests wider, shallower residual networks can improve CIFAR accuracy and training efficiency versus thin deep variants. The H20 has ample VRAM, so a modest width increase is plausible. This directly attacks model capacity and representation quality, which may matter if recipe-level regularization only yields small gains.

**Sources**: Wide Residual Networks paper https://arxiv.org/abs/1605.07146; local model implementation in `train.py`; task note that VRAM increases are acceptable for meaningful accuracy gains.

**Estimated Effort**: medium

**Risk Assessment**: More parameters and FLOPs may reduce epoch count within 300 seconds, which could offset accuracy gains. A new architecture also has more surface area for implementation bugs. If too wide, the run may be slower, less optimized, or overfit without tuned regularization.

### 3. Mixup-Style Target Regularization
**Summary**: Add minibatch mixup in the training loop with soft targets and a moderate alpha value, keeping the current ResNet-20 architecture and SGD optimizer. This tests whether interpolated examples improve generalization more than image-space masking alone.

**Reasoning**: Mixup-style regularization is a strong generalization technique for image classifiers and can be implemented entirely in `train.py` without new dependencies. It is more invasive than label smoothing because it changes the loss computation and target representation, but it may improve robustness under the same evaluation harness.

**Sources**: local code `train.py`; general CIFAR training practice; no prior project experiments.

**Estimated Effort**: medium

**Risk Assessment**: Mixup can underfit small models if alpha or LR are not tuned, and it complicates loss reporting. It is also less directly grounded by the consulted sources than cutout/SGDR for this first loop. Worst case is a no-improvement run with lower peak accuracy.

## Idea Evaluation

Candidate 1 has the strongest risk-adjusted evidence for the first loop. It combines two directly relevant, low-cost interventions: cutout-style augmentation from CIFAR-focused regularization evidence, and cosine scheduling from CIFAR SGD scheduling evidence. Its mechanism is clear: increase effective data diversity, reduce overconfident fits, and avoid abrupt LR transitions while preserving architecture speed and the once-per-epoch evaluation cadence.

Candidate 2 may have the highest eventual ceiling because architecture capacity can move CIFAR accuracy materially, and WRN evidence is strong. However, it is a larger implementation and runtime bet. Under a strict 300 second budget, the first experiment should establish whether the current thin ResNet can be improved cheaply before replacing the model.

Candidate 3 is plausible but less attractive as the first test. Mixup changes target semantics and loss handling, and its benefit on this short fixed-budget ResNet-20 run is less certain than cutout plus schedule improvements.

## Chosen Idea
**Selected**: Cheap CIFAR Regularization and Cosine Schedule

**Why this idea**:
This is the most defensible first experiment because it targets obvious gaps in the current recipe while keeping compute, architecture, and implementation risk low. It is also easy to analyze: if it improves, later loops can compose it with a wider model; if it fails, the report will clarify whether over-regularization or LR timing was the likely cause.

**Hypothesis**:
Adding cutout-style random erasing, modest label smoothing, and a cosine LR schedule to the existing ResNet-20 training loop will improve `best_test_acc` above the 91.52% baseline within the same 300 second budget by improving generalization without materially reducing training throughput.
