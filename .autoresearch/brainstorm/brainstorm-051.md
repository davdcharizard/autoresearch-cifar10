# Brainstorm EXP-051
**Created**: 2026-06-09
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

## Web Search & Literature Review

- **ResNet zero-init residual context** (`knowledge/references/resnet-zero-init-residual.md`)
  Zeroing the final BatchNorm scale in residual branches is a known identity-preserving initialization lever, but the local fixed-budget risk is slower early representation learning.
- **Wide residual network context** (`knowledge/papers/wide-residual-networks.md`)
  Capacity scaling has already validated the current `28/56/112` anchor, so EXP-051 should avoid more width and focus on a no-parameter change.
- **PyTorch throughput context** (`knowledge/references/pytorch-throughput-tools.md`)
  The current FP32 compile/channels-last path should remain intact; the next experiment should not trade away step budget unless the mechanism is very strong.

No new external search was needed. The next useful space is now dominated by local evidence: isolated augmentation, scalar LR, scalar decay, BN momentum, EMA, and cosine variants have all failed to clear the `94.07%` threshold.

## Experimental History Review

- Current best remains EXP-038 at `best_test_acc=93.97%`; a new improvement must reach at least `94.07%`.
- The validated anchor is `STAGE_WIDTHS=(28, 56, 112)`, batch size 128, LR 0.1, momentum 0.9, weight decay 2e-4, first LR drop at 21000, reflection crop padding, label smoothing 0.05, FP32 compile, and channels-last.
- EXP-050 closed the ColorJitter attribution gap: clean mild ColorJitter reached the LR drop but peaked at 93.49%, so isolated photometric jitter should be deprioritized.
- EXP-049 showed decoupled weight decay weakens the anchor, and EXP-039/041 bracketed nearby scalar weight decay values as worse.
- EXP-048 showed lower BN momentum is not useful, so BatchNorm state dynamics should not be changed directly.
- EXP-028 fully zero-initialized residual branch `bn2.weight` and undertrained badly at 91.74%, but its report explicitly leaves a partial scale such as 0.1 as an unexplored avenue.
- The next candidate should preserve the exact schedule, optimizer, augmentation, and throughput path, while testing a distinct mechanism not already cleanly failed.

## Candidate Ideas

### 1. Partial Residual-Branch BN Scale Initialization
**Summary**: Initialize every `BasicBlock.bn2.weight` to `0.1` after normal module initialization, rather than default 1.0 or full zero-gamma. This keeps residual branches active but damped at startup.

**Reasoning**: EXP-028 suggests full identity bias is too aggressive for the 300s budget, but a small nonzero residual scale may improve early stability without fully suppressing residual learning. It changes no parameter count, data pipeline, optimizer, schedule, validation cadence, or runtime-heavy component.

**Sources**: `knowledge/references/resnet-zero-init-residual.md`; `reports/exp-report-028.md`; `train.py` `BasicBlock.bn2`.

**Estimated Effort**: low

**Risk Assessment**: The local prior is negative because full zero-gamma undertrained. The intended difference is the nonzero scale; if 0.1 still slows learning too much, the run should fail cleanly as no-improvement.

### 2. Hybrid Post-Drop Cosine Tail
**Summary**: Preserve the 21k high-LR window and first step drop to 0.01, then replace the flat 0.01 tail with a smooth cosine decay over the remaining time-budgeted steps.

**Reasoning**: Full time-budget cosine underperformed in EXP-046, but that changed the whole schedule. A tail-only cosine would preserve the validated first drop while trying to reduce late plateau drift.

**Sources**: `reports/exp-report-046.md`; goal-learnings schedule findings; `knowledge/papers/sgdr-cosine-schedule.md`.

**Estimated Effort**: medium

**Risk Assessment**: Schedule-only changes have repeatedly failed, including second drops and full cosine. This idea has a plausible distinction but fights a strong negative local pattern.

### 3. Very Mild Residual Dropout
**Summary**: Add a tiny dropout probability inside the residual branch before the second convolution or before the residual add, active only during training.

**Reasoning**: A small stochastic regularizer could improve generalization without changing evaluation, but unlike cutout it acts inside the model rather than the image pipeline.

**Sources**: local failed-approach history on external regularizers; `train.py` `BasicBlock`.

**Estimated Effort**: medium

**Risk Assessment**: This introduces runtime overhead and can destabilize residual learning. It is less directly supported by local evidence than partial BN scaling and could reduce useful step throughput.

## Idea Evaluation

Partial residual-branch BN scaling is the best next experiment despite the negative EXP-028 prior because it directly tests the one reasonable variant that full zero-gamma left open. The mechanism is clear and isolated: reduce initial residual-branch scale without fully zeroing it. It also has almost no runtime cost and keeps the validated anchor intact.

The hybrid post-drop cosine tail is conceptually interesting, but schedule-only work is now a high-importance failed pattern. Preserving the first drop distinguishes it from EXP-046, yet it still risks spending another loop on a schedule family with repeated local failures.

Very mild residual dropout is more speculative and adds training-loop overhead or stochasticity. It may be worth considering later, but it has weaker evidence and a worse failure mode than an initialization-only test.

## Chosen Idea
**Selected**: Partial Residual-Branch BN Scale Initialization

**Why this idea**:
It is a tightly scoped, no-overhead way to revisit a documented ResNet initialization mechanism while explicitly avoiding EXP-028's all-zero residual branch. The partial scale tests whether a weaker identity bias can improve stability without sacrificing the fixed-budget learning speed.

**Hypothesis**:
Initializing each residual block's final BatchNorm scale to `0.1` will mildly stabilize residual updates while preserving enough branch signal for fast learning, improving `best_test_acc` to at least `94.07%`.
