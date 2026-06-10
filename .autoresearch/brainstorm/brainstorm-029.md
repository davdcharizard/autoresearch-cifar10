# Brainstorm EXP-029
**Created**: 2026-06-08
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

<!-- This file is focused on IDEATION only.
     Goal statement, primary metric, direction, hard constraints, and verification criteria
     live in the goal file (see pointer above). Baseline lives in experiment-indices/maximize-cifar10-best-test-accuracy.tsv.
     Do not duplicate those fields here — always point to the source of truth. -->

## Web Search & Literature Review

- **Torchvision RandomCrop documentation** (https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.RandomCrop.html)
  `RandomCrop` supports `padding_mode` with non-constant options including `reflect`, so the crop-padding boundary behavior can be changed without changing crop size, padding amount, or adding dependencies.

- **Improved Regularization of Convolutional Neural Networks with Cutout** (`knowledge/papers/cutout-cifar-regularization.md`)
  CIFAR accuracy can be sensitive to train-time image augmentation, but the local cutout failures show erased-patch regularization is too strong for this fixed-budget anchor.

- **PyTorch EMA Weight Averaging** (`knowledge/references/pytorch-ema-averaging.md`)
  EMA remains a plausible accuracy lever, but prior local EMA and equal-averaging attempts make it a higher-risk follow-up unless implemented with very low overhead.

## Experimental History Review

- Current baseline is EXP-016 at `best_test_acc=93.23%`; under the goal's +0.10 percentage-point rule, EXP-029 must reach at least `93.33%`.
- The current anchor is `STAGE_WIDTHS = (28, 56, 112)`, `BATCH_SIZE = 128`, `LR = 0.1`, `MOMENTUM = 0.9`, `WEIGHT_DECAY = 1e-4`, `LR_MILESTONES = [21000, 64000]`, FP32, channels-last, and `torch.compile`.
- Width scaling has a high-importance recurring failure beyond 28/56/112: proportional widening, minimal widening, and final-stage-only widening all missed the threshold.
- Schedule-only variants around the anchor appear locally bounded: first-drop brackets at 20k and 23k underperformed the 21k anchor, and a reachable 36k second drop reached only 93.13%.
- Optimizer and regularization perturbations have mostly hurt: Nesterov, momentum 0.95, lower weight decay, no-decay BatchNorm/bias groups, cutout, and zero-initialized residual branches all underperformed.
- A lighter augmentation-boundary change is still untested. It does not erase pixels, alter model capacity, change optimizer dynamics, or add validation calls, and should preserve the existing throughput path.

## Candidate Ideas

### 1. Reflection Padding for RandomCrop
**Summary**: Change the training transform from `transforms.RandomCrop(32, padding=4)` to `transforms.RandomCrop(32, padding=4, padding_mode="reflect")`, preserving every other augmentation, normalization, model, optimizer, schedule, and evaluation setting.

**Reasoning**: The current random crop pads CIFAR images with constant zeros before cropping. Reflection padding can reduce artificial border artifacts while keeping the same crop distribution and no extra model or training-loop cost. This is meaningfully different from the failed cutout variants because it does not remove image content or increase regularization strength; it changes only how synthetic crop margins are filled.

**Sources**: Torchvision RandomCrop documentation; `train.py` transform definition; EXP-005 and EXP-009 cutout failures; EXP-016 anchor; EXP-028 no-improvement.

**Estimated Effort**: low

**Risk Assessment**: The expected effect may be small, and the current zero-padding behavior may already be part of the tuned recipe. Worst case is a valid no-improvement run with unchanged throughput and lower or similar accuracy.

### 2. Low-Frequency Late EMA
**Summary**: Maintain a second EMA model only after the first LR drop and update it sparsely, then evaluate the EMA weights once per epoch instead of evaluating the raw model.

**Reasoning**: EXP-004 showed EMA narrowly helped but not enough after per-step overhead, while EXP-021 showed naive long equal averaging collapses. A late sparse EMA directly targets those failure modes by starting only in the refinement phase and reducing update cost.

**Sources**: `knowledge/references/pytorch-ema-averaging.md`; EXP-004; EXP-021; `goal-learnings/maximize-cifar10-best-test-accuracy.md`.

**Estimated Effort**: medium

**Risk Assessment**: This is more invasive than an augmentation-boundary change, may still reduce steps, and can mishandle BatchNorm statistics if buffers lag. It should remain a backup until cheaper no-overhead probes are exhausted.

### 3. Batch Size 112 With Reachable First Drop
**Summary**: Try a milder batch-size reduction than EXP-025, retuning milestones to the measured reachable step budget instead of attempting a second LR drop that may be missed.

**Reasoning**: Batch size 96 was too slow, but a smaller move from 128 to 112 might increase gradient noise enough to improve generalization while preserving more throughput. It is a distinct batch-size probe rather than a repeat of the aggressive 96 setting.

**Sources**: EXP-025; EXP-016; `goal-learnings/maximize-cifar10-best-test-accuracy.md`.

**Estimated Effort**: medium

**Risk Assessment**: Step loss remains the main risk, and schedule calibration would be speculative before a full run. This has lower priority than a one-line augmentation change with no expected training-loop overhead.

## Idea Evaluation

Reflection padding has the best risk-adjusted profile for EXP-029. Its external support is straightforward API support rather than a promise of accuracy, but the mechanism is clear: change crop-border statistics without adding a stronger regularizer or reducing the step budget. It is also orthogonal to the failed width, schedule, optimizer, weight decay, and residual-initialization changes.

Low-frequency late EMA has a plausible path to a larger gain, but two local averaging experiments already exposed overhead and stability risks. Batch size 112 could still be useful, but EXP-025 makes any smaller-batch experiment step-budget risky. Given the current +0.10 point threshold, the next experiment should first test the lowest-overhead untried perturbation that preserves the anchor recipe.

## Chosen Idea
**Selected**: Reflection Padding for RandomCrop

**Why this idea**:
It is a narrowly scoped, one-line augmentation-boundary change with official torchvision support, no new dependency, no extra validation, no model-capacity increase, and no expected throughput penalty. It explores an untested part of the recipe while avoiding the recurring local failure modes.

**Hypothesis**:
Using reflection padding for random crop margins will reduce artificial zero-border artifacts in training views and improve the 28/56/112 ResNet-20's best test accuracy enough to reach at least `93.33%`.
