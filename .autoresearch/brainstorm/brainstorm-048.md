# Brainstorm EXP-048
**Created**: 2026-06-09
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

## Web Search & Literature Review

- **PyTorch throughput tools** (`knowledge/references/pytorch-throughput-tools.md`)
  The current anchor already uses the validated FP32 compile/channels-last path. Further ideas should avoid reduced precision and avoid extra runtime overhead unless the expected accuracy gain is large.
- **Wide Residual Networks** (`knowledge/papers/wide-residual-networks.md`)
  Wider residual networks can improve CIFAR accuracy, but local experiments show this repo's 28/56/112 width is already the best calibrated capacity point under the current fixed budget.

No new external search was needed for EXP-048. The local experiment history is more specific than generic CIFAR guidance at this point, and recent infrastructure contention makes low-overhead local changes the priority.

## Experimental History Review

- Current best remains EXP-038 at `best_test_acc=93.97%`; the active threshold is 94.07% because improvements must clear +0.10 percentage points.
- The validated anchor is `STAGE_WIDTHS=(28, 56, 112)`, batch size 128, LR 0.1, momentum 0.9, weight decay 2e-4, first LR drop at step 21000, reflection crop padding, label smoothing 0.05, FP32 compile, and channels-last.
- EXP-047 adds a protocol warning: GPU contention can make step-schedule experiments miss the 21k first drop. Candidate ideas should avoid extra compute or CPU/GPU overhead and should include a clean milestone check.
- Strong negative evidence rules out several broad categories: further width increases, isolated scalar LR changes, isolated nearby weight-decay retunes, smaller batch sizes, cutout masking, EMA/SWA-style averaging, and isolated second-drop schedule tuning.
- Photometric augmentation is not cleanly exhausted because EXP-047 missed the first LR drop, but another augmentation retry should wait for a cleaner GPU window or a stronger low-overhead reason.
- The remaining promising space is low-cost training dynamics that preserve the anchor while affecting generalization, especially mechanisms not yet bracketed by prior experiments.

## Candidate Ideas

### 1. Lower BatchNorm Momentum to 0.05
**Summary**: Set all BatchNorm layers to use `momentum=0.05` instead of the PyTorch default `0.1`, while preserving every other anchor setting. This changes only the running-statistics update rate used for evaluation, not model size, optimizer, schedule, augmentation, or loss.

**Reasoning**: CIFAR evaluation uses BatchNorm running statistics through `model.eval()`. The current model sees many augmented mini-batches, and a slower running-stat update may smooth noisy per-batch estimates without adding runtime overhead. This is distinct from failed weight averaging and failed smoothing retunes: it targets normalization-state stability rather than weights, labels, or learning rate.

**Sources**: `train.py` BatchNorm defaults; `prepare.py` evaluation uses `model.eval()`; goal-learnings patterns for preserving the 21k anchor; EXP-047 protocol finding on avoiding overhead.

**Estimated Effort**: low

**Risk Assessment**: Running stats may lag too much and slightly hurt calibration, but the failure mode should be a clean no-improvement. The change has negligible throughput impact and should still reach the first LR drop if the GPU is not severely contended.

### 2. Partial Residual-Branch BN Scale Initialization
**Summary**: Initialize each residual block's final BatchNorm scale to a small positive value such as `0.1` instead of the default random-initialized scale or the failed zero-gamma value.

**Reasoning**: EXP-028 showed full zero-gamma identity bias undertrained badly, but a partial scale could keep some early residual learning while improving optimization smoothness. It is a no-parameter, low-overhead architecture-initialization tweak.

**Sources**: EXP-028 report; `knowledge/references/resnet-zero-init-residual.md`; current `BasicBlock.bn2` structure in `train.py`.

**Estimated Effort**: low

**Risk Assessment**: EXP-028 is strong negative evidence for this family. Even partial scaling may still slow early representation learning enough to miss the threshold, so this is lower priority than a normalization-statistics tweak.

### 3. Decoupled SGD Weight Decay at 2e-4
**Summary**: Keep SGD momentum and LR schedule, but implement decoupled weight decay manually instead of optimizer-coupled L2 decay.

**Reasoning**: EXP-038 validated stronger shrinkage, while EXP-039/041 bracketed scalar coupled decay around `2e-4`. Decoupling decay from gradient and momentum dynamics might preserve the useful regularization magnitude while avoiding momentum-buffer interactions.

**Sources**: EXP-038/039/041 reports; current SGD optimizer path in `train.py`.

**Estimated Effort**: medium

**Risk Assessment**: Implementation is more invasive and easy to miscalibrate because classical SGD with coupled L2 is part of the validated anchor. It also makes attribution less clean than a simple BN-statistics parameter.

## Idea Evaluation

Lower BatchNorm momentum has the best risk-adjusted value for EXP-048. It is a genuinely new axis, it modifies only normalization state dynamics, and it should add no meaningful overhead under the fixed time budget. It also directly interacts with the evaluation pathway because `Eval.evaluate()` calls `model.eval()`, so running-stat quality can affect `best_test_acc`.

Partial residual BN scaling is plausible but carries the negative prior from EXP-028. It should remain a later option only if a partial value is carefully justified, because full zero-gamma already showed severe fixed-budget undertraining.

Decoupled weight decay is mechanistically interesting but less conservative. The current best recipe already depends on coupled `WEIGHT_DECAY=2e-4`, and nearby scalar brackets failed. Changing decay semantics is better saved for a more deliberate optimizer-dynamics loop.

## Chosen Idea
**Selected**: Lower BatchNorm Momentum to 0.05

**Why this idea**:
It is the cleanest low-overhead training-dynamics test left after LR, weight decay, smoothing, batch size, and augmentation brackets. It preserves the validated step-schedule and regularization anchor while testing whether smoother BatchNorm running statistics improve evaluation accuracy.

**Hypothesis**:
Using `momentum=0.05` for all BatchNorm layers will smooth noisy running-stat estimates from augmented CIFAR mini-batches and improve `best_test_acc` to at least 94.07% without reducing step coverage or violating the fixed evaluation harness.
