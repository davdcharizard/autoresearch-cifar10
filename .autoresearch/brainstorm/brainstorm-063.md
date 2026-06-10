# Brainstorm EXP-063
**Created**: 2026-06-09
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

## Web Search & Literature Review

- **Squeeze-and-Excitation Networks** (`knowledge/papers/squeeze-and-excitation-networks.md`)
  SE blocks use global feature statistics to recalibrate channel responses, can be inserted into existing residual blocks, and are reported to improve classification accuracy with modest compute. The saved implementation note recommends inserting the gate after the residual branch's final normalization and before shortcut addition.
- **Current anchor implementation** (`train.py`)
  `BasicBlock` has a clean insertion point after `bn2(conv2)` and before shortcut addition. The `ResNet` constructor builds `layer1`, `layer2`, and `layer3` separately, so a stage-limited gate can be implemented without changing the optimizer, data pipeline, schedule, or evaluation loop.
- **EXP-058 all-block SE report** (`reports/exp-report-058.md`)
  All-block SE reached the first LR drop and completed validly but peaked at 93.71%, suggesting broad per-block gating is too expensive or poorly matched. Its unexplored avenue explicitly identifies cheaper final-stage-only SE as a narrower test.
- **EXP-062 compact ResNet-14 report** (`reports/exp-report-062.md`)
  Removing depth and widening to `(32,64,128)` improved step count but fell to 93.51%, so the next architecture probe should preserve ResNet-20 depth rather than trading away residual blocks.

## Experimental History Review

- Current best remains EXP-038 at `best_test_acc=93.97%`; EXP-063 must reach at least `94.07%` to count as an improvement under the explicit +0.10 percentage-point threshold.
- The validated anchor is ResNet-20 with `STAGE_WIDTHS=(28,56,112)`, `BATCH_SIZE=128`, `LR=0.1`, `MOMENTUM=0.9`, `WEIGHT_DECAY=2e-4`, `LR_MILESTONES=[21000,64000]`, reflection crop padding, full-run label smoothing 0.05, FP32 compile, channels-last, and once-per-epoch validation.
- Many scalar and regularization families are now closed or low priority: isolated LR/weight-decay brackets, batch-size deviations, label-smoothing changes, cosine tails, mixup variants, Cutout, mild RandAugment/ColorJitter, EMA/SWA-style averaging, residual BN scale initialization, stochastic depth, classifier-head dropout, and shortcut smoothing.
- Architecture capacity evidence is mixed but now more constrained. Width scaling to 28/56/112 was a major path to the current anchor, but widening beyond that and compact shallow-wide depth removal both underperform.
- EXP-058 shows all-block SE is not enough, but it does not fully rule out a final-stage-only gate because the overhead and early-feature perturbation would be smaller. The remaining plausible architecture gap is localized, depth-preserving modification rather than broad capacity scaling.

## Candidate Ideas

### 1. Final-Stage-Only SE Gate
**Summary**: Add Squeeze-and-Excitation channel recalibration only inside `layer3` residual blocks. Keep stages 1 and 2 as plain `BasicBlock`s, preserve `NUM_BLOCKS=3`, `STAGE_WIDTHS=(28,56,112)`, and leave all optimizer, schedule, augmentation, loss, compile, and evaluation settings unchanged.

**Reasoning**: The final stage contains the highest-level class-semantic channels, so channel recalibration there may improve classification features while avoiding all-block SE's early-feature disruption and most of its repeated overhead. This directly tests the still-open variant named in EXP-058 and obeys EXP-062's lesson to preserve ResNet-20 depth.

**Sources**: `knowledge/papers/squeeze-and-excitation-networks.md`; `reports/exp-report-058.md`; `reports/exp-report-062.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`.

**Estimated Effort**: medium

**Risk Assessment**: The all-block SE miss is negative prior evidence, and even two final-stage gates may add enough overhead or optimization noise to miss 94.07%. The failure mode should be a valid no-improvement with useful attribution because the change is localized and easy to verify at startup.

### 2. Stage-3 Width Redistribution Without Extra Total Depth
**Summary**: Preserve ResNet-20 depth but slightly redistribute width toward the final stage, for example `(26, 52, 120)` or a similarly small late-stage-biased shape, while keeping total model size near the current anchor and retaining the fixed schedule.

**Reasoning**: EXP-062 suggests not removing residual depth, while prior final-stage-only widening to 128 failed when applied on top of the current widths. A smaller redistribution could increase high-level capacity with less overall cost than broad widening.

**Sources**: experiment index EXP-017, EXP-019, EXP-020; `reports/exp-report-062.md`; goal-learning width-scaling findings.

**Estimated Effort**: low

**Risk Assessment**: This is close to the recurring width-beyond-anchor failure and may repeat a known bad family unless the exact redistribution materially reduces early-stage cost. Evidence is weaker than the SE-localization gap.

### 3. Late-Only Tiny Stochastic Depth
**Summary**: Activate very small stochastic depth only after the first LR drop and only in the final stage, preserving the high-LR representation-learning phase exactly.

**Reasoning**: Static stochastic depth failed in EXP-054, but a late-only final-stage variant would test whether timing caused the failure. It targets late overfit without weakening early feature acquisition.

**Sources**: `knowledge/papers/stochastic-depth-resnets.md`; `reports/exp-report-054.md`; `reports/exp-report-061.md`; goal-learning residual regularizer entries.

**Estimated Effort**: medium

**Risk Assessment**: Evidence is weak because static stochastic depth and classifier dropout both underperformed. Scheduler-conditioned model behavior also increases complexity and attribution risk compared with a local architecture gate.

## Idea Evaluation

Final-stage-only SE has the clearest remaining mechanism and the strongest direct support from the current experiment map. It is not a retry of all-block SE: it preserves early residual blocks and applies channel recalibration only where class-level channels should matter most. It also responds directly to EXP-062 by preserving depth instead of chasing throughput through a shallower network.

Stage-3 width redistribution is simple, but it sits near the high-importance width-beyond-anchor failure. It may still be worth trying later, but the current evidence says extra channels above 28/56/112 are a poor bet unless paired with a stronger architectural rationale.

Late-only tiny stochastic depth is conceptually distinct from static regularization, but it is more stateful and has weaker evidence. Static residual regularization and classifier-head dropout both underperformed, so this should wait until the more localized architecture gap is closed.

The lead candidate is therefore Final-Stage-Only SE Gate. It is a focused, depth-preserving architecture test with a known implementation path and a clear scientific question left open by EXP-058.

## Chosen Idea
**Selected**: Final-Stage-Only SE Gate

**Why this idea**:
It is the most targeted untested architecture variant after EXP-062 showed depth removal fails and EXP-058 showed broad SE is too weak. A final-stage-only gate limits overhead and perturbation while testing whether semantic channel recalibration can lift the current ResNet-20 anchor.

**Hypothesis**:
Adding SE gates only to `layer3` will preserve the first LR drop and most of the anchor's step budget while improving late-stage feature calibration enough to reach at least `94.07%` `best_test_acc`.
