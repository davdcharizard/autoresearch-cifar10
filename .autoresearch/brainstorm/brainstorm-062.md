# Brainstorm EXP-062
**Created**: 2026-06-09
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

## Web Search & Literature Review

- **Existing knowledge base** (`knowledge/README.md`)
  The saved knowledge covers the relevant background for this loop: throughput tools, crop padding, width scaling, cutout/mixup/RandAugment, cosine schedules, stochastic depth, SE attention, residual initialization, and shortcut/downsampling tweaks. No new external search was needed because the next high-value gap is directly identified by the current experimental map.
- **Wide Residual Networks** (`knowledge/papers/wide-residual-networks.md`)
  Wider, shallower residual networks can improve CIFAR accuracy by trading depth for width. This project's prior width-scaling wins stopped at 28/56/112 with three blocks per stage; EXP-062 can test whether a shallower two-block-per-stage model reopens the capacity frontier.
- **Current anchor implementation** (`train.py`)
  The model is controlled by `NUM_BLOCKS` and `STAGE_WIDTHS`, making a compact-depth/width test a small, `train.py`-only structural intervention.

## Experimental History Review

- Current best remains EXP-038 at `best_test_acc=93.97%`; EXP-062 must reach at least `94.07%` to count as an improvement under the explicit +0.10 percentage-point threshold.
- The validated anchor is `NUM_BLOCKS=3`, `STAGE_WIDTHS=(28, 56, 112)`, `BATCH_SIZE=128`, `LR=0.1`, `MOMENTUM=0.9`, `WEIGHT_DECAY=2e-4`, `LR_MILESTONES=[21000, 64000]`, reflection crop padding, full-run label smoothing 0.05, FP32 compile, channels-last, and once-per-epoch validation.
- EXP-061 closed the smallest remaining head-only regularization lever: final classifier dropout preserved throughput and LR timing but peaked at 93.54%.
- Recurring failed families now include isolated schedule changes, weight averaging, batch-size deviations, label-smoothing deviations, cosine schedules, mild mixup, BN/bias no-decay, scalar LR deviations, residual branch scaling, isolated augmentation, broad SE, shortcut smoothing, and classifier-head dropout.
- The major untested frontier is now structural: change the depth/width tradeoff rather than adding another scalar regularizer. Because widening beyond 28/56/112 failed at three blocks per stage, a shallower model is the cleanest way to test whether more channels can be afforded without the same throughput loss.

## Candidate Ideas

### 1. Compact ResNet-14 with Moderate Width Increase
**Summary**: Change the CIFAR ResNet from three residual blocks per stage to two (`NUM_BLOCKS = 2`, ResNet-14) while moderately increasing stage widths to `STAGE_WIDTHS = (32, 64, 128)`. Keep the current optimizer, schedule, augmentation, batch size, label smoothing, weight decay, compile path, and evaluation loop unchanged.

**Reasoning**: Prior capacity gains came from width scaling, but additional width at full depth lost the fixed-budget tradeoff. A two-block-per-stage model reduces depth and residual-block count, which may recover enough throughput to afford a modest width increase. The mechanism is different from EXP-017/019/020 because it changes depth and width together rather than adding channels on top of the same depth. It directly tests the compact-WRN hypothesis from the knowledge base while remaining a small `train.py` edit.

**Sources**: `knowledge/papers/wide-residual-networks.md`; experiment index EXP-011 through EXP-020; `goal-learnings/maximize-cifar10-best-test-accuracy.md` width-scaling pattern and width-beyond-anchor failure.

**Estimated Effort**: low

**Risk Assessment**: Reducing depth may remove useful representational capacity and the wider channels may still be slower than expected. The likely failure mode is a valid no-improvement, but the result will clarify whether the remaining search should keep structural architecture changes alive.

### 2. Stage-3-Only SE Gate
**Summary**: Add SE channel gates only to the final stage (`layer3`) instead of every residual block. Keep stages 1 and 2 unchanged and preserve the optimizer, schedule, augmentation, loss, batch size, and evaluation loop.

**Reasoning**: EXP-058 showed all-block SE underperformed, but that does not fully rule out a late-only gate. Stage-3 channels are closest to classification semantics, so a narrow recalibration might help without the overhead and early-feature perturbation of all-block SE.

**Sources**: `knowledge/papers/squeeze-and-excitation-networks.md`; `reports/exp-report-058.md`; goal-learning entry "SE channel attention underperforms the current block."

**Estimated Effort**: medium

**Risk Assessment**: This is adjacent to a recent no-improvement, so evidence strength is weak. Implementation is more invasive than a depth/width constant change and may still add enough overhead or gating noise to miss the anchor.

### 3. Delayed Late-Only Regularizer
**Summary**: Add a regularizer that activates only after the first LR drop, such as late classifier dropout or late weak stochastic depth. Preserve the high-LR representation-learning phase exactly and regularize only the LR 0.01 refinement phase.

**Reasoning**: Several static regularizers underperform, but a late-only version could target post-drop overfit without weakening pre-drop learning. This would test whether timing, rather than regularizer type, caused some previous misses.

**Sources**: `reports/exp-report-054.md`; `reports/exp-report-061.md`; goal-learnings on failed static residual and classifier regularizers.

**Estimated Effort**: medium

**Risk Assessment**: The evidence is weak because both candidate regularizer families already failed in static form. Scheduler-conditioned behavior also increases implementation complexity and attribution risk.

## Idea Evaluation

Compact ResNet-14 with moderate width increase has the strongest distinct mechanism. The current search has exhausted many scalar and regularization levers, while the architecture frontier has not tested depth reduction as the way to afford wider representations. Prior width-scaling wins provide positive evidence that additional channel capacity can help, and the failures beyond 28/56/112 may be throughput/depth tradeoff failures rather than absolute capacity failures.

Stage-3-only SE is plausible but close to EXP-058's all-block SE miss. It remains worth trying eventually, but after broad SE reached only 93.71%, a narrower SE variant has weaker evidence than a structural depth/width frontier test.

Delayed late-only regularization is an interesting salvage path for failed regularizers, but EXP-054 and EXP-061 both suggest the anchor is not obviously regularization-limited. It is also more stateful than the compact architecture change and could produce a result that is harder to interpret.

The lead candidate is therefore Compact ResNet-14 with Moderate Width Increase. It makes a small, explicit architecture tradeoff, avoids repeating closed scalar families, and directly probes whether the remaining path to 94.07% requires a different capacity/throughput balance.

## Chosen Idea
**Selected**: Compact ResNet-14 with Moderate Width Increase

**Why this idea**:
It is the most distinct remaining lever after EXP-061 closed head-only dropout. It tests a plausible compact-WRN-style tradeoff using only two constants in `train.py`, while preserving all optimizer, schedule, augmentation, and evaluation anchors for clean attribution.

**Hypothesis**:
Changing to `NUM_BLOCKS = 2` and `STAGE_WIDTHS = (32, 64, 128)` will preserve or improve step throughput relative to full-depth widening, retain enough representational capacity, reach the step-21000 LR drop, and raise `best_test_acc` to at least `94.07%`.
