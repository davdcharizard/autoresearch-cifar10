# Brainstorm EXP-043
**Created**: 2026-06-09
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

## Web Search & Literature Review

- **SGDR / cosine scheduling knowledge entry** (`.autoresearch/knowledge/papers/sgdr-cosine-schedule.md`)
  Smoother LR decay is a known CIFAR-relevant scheduling lever, but this repo's current step schedule is already strongly validated; cosine remains a plausible later alternative if simple optimizer probes stall.
- **PyTorch EMA averaging knowledge entry** (`.autoresearch/knowledge/references/pytorch-ema-averaging.md`)
  AveragedModel/EMA can improve evaluated weights without changing the optimizer path, but prior local EMA/averaging variants in this repo had overhead or collapse risks.
- **Mixup knowledge entry** (`.autoresearch/knowledge/papers/mixup-beyond-erm.md`)
  Mixup is a CIFAR-relevant regularizer, but EXP-042 crashed before final metrics and introduced run-control risk; it should not be retried immediately.

## Experimental History Review

- Current baseline is `best_test_acc=93.97%` from EXP-038 / commit `755be2c`; the active goal requires `best_test_acc >= 94.07%`.
- The current anchor is `STAGE_WIDTHS=(28,56,112)`, reflected `RandomCrop`, `label_smoothing=0.05`, `WEIGHT_DECAY=2e-4`, `LR=0.1`, `MOMENTUM=0.9`, and `LR_MILESTONES=[21000,64000]`.
- EXP-038 validated stronger shrinkage: `WEIGHT_DECAY=2e-4` improved the label-smoothed reflection anchor to 93.97%.
- EXP-039 and EXP-041 bracketed weight decay around `2e-4`; both `3e-4` and `1.5e-4` were worse, so isolated weight-decay retuning is low priority.
- EXP-040 showed `LR=0.12` weakened the `2e-4` anchor to 93.70%, indicating the high-LR side is too noisy or unstable.
- The lower initial LR side remains untested on the `2e-4` anchor. It is a clean scalar probe with no throughput or infrastructure risk.
- Width increases beyond 28/56/112, schedule-only second drops, smaller batches, cutout, smoothing deviations, projection shortcuts, and zero-init residual branches are all documented failed or lower-priority spaces.
- EXP-042's mixup crash does not disprove mixup, but it argues for choosing the next experiment with a highly reliable local execution path.

## Candidate Ideas

### 1. Initial LR 0.08 on the 2e-4 Anchor
**Summary**: Change only `LR` from `0.1` to `0.08`, preserving `LR_MILESTONES=[21000,64000]`, `WEIGHT_DECAY=2e-4`, label smoothing, reflection padding, architecture, batch size, and the full training loop.

**Reasoning**: EXP-040 established that increasing initial LR to `0.12` hurts the current anchor. Testing the lower side asks whether less high-LR noise improves the post-drop plateau. This is not a schedule-only second-drop probe and not a weight-decay retune; it is the clean missing side of the initial-LR bracket around the successful `2e-4` anchor. It has almost no implementation risk and should preserve step budget and verification geometry.

**Sources**: EXP-038 report; EXP-040 report; `.autoresearch/goal-learnings/maximize-cifar10-best-test-accuracy.md` failed LR and validated weight-decay entries.

**Estimated Effort**: low

**Risk Assessment**: Lower LR may undertrain pre-drop features or reduce useful exploration, producing a valid no-improvement. Worst case is a clear metric regression without destabilizing the harness.

### 2. No-Restart Cosine Annealing over Expected Step Horizon
**Summary**: Replace `MultiStepLR` with `CosineAnnealingLR` over the expected reachable step budget, likely around 41000 steps, while preserving all other anchor settings.

**Reasoning**: Cosine schedules are CIFAR-relevant and can smooth transitions compared with abrupt drops. However, this repo has already found that the current 21k first drop is a strong calibrated schedule anchor, and earlier cosine/strong-regularization bundling in EXP-000 undertrained badly. A clean cosine-only test may still be informative, but it changes the whole LR trajectory and is less local than the LR 0.08 bracket.

**Sources**: `.autoresearch/knowledge/papers/sgdr-cosine-schedule.md`; EXP-000 index row; schedule failed-approach entries.

**Estimated Effort**: medium

**Risk Assessment**: A full-trajectory schedule swap may spend too long at intermediate/low LR or miss the known useful post-drop regime. Failure mode is likely valid no-improvement, but interpretation may be less clean.

### 3. Bounded Late EMA Evaluation
**Summary**: Add a low-frequency EMA copy that starts only after the first LR drop, updates every few steps or epochs, and evaluates the EMA model once per epoch instead of the raw model.

**Reasoning**: Late validation often drifts below the peak, so an averaged model could stabilize the evaluated weights. The PyTorch EMA reference supports this without changing optimizer or data flow. But EXP-004 showed per-step EMA overhead and EXP-021 showed naive post-drop equal averaging collapse, so this must be carefully bounded and is more implementation-sensitive than an LR scalar probe.

**Sources**: `.autoresearch/knowledge/references/pytorch-ema-averaging.md`; EXP-004 and EXP-021 failed-approach entries; EXP-042 report warning against risky next experiments.

**Estimated Effort**: medium

**Risk Assessment**: EMA may slow throughput, mishandle BatchNorm statistics, or repeat prior averaging failures. It is worth revisiting later with a tight implementation, but not immediately after an execution crash.

## Idea Evaluation

`LR=0.08` has the strongest near-term evidence-to-risk ratio. It directly brackets the successful anchor after the high-LR side failed, changes only one scalar, and should produce a valid benchmark run with the same local execution path as prior successful scalar probes. Its expected upside is modest, but the +0.10 rule means a small stabilizing effect could still count if the lower LR improves post-drop accuracy from 93.97 to 94.07 or better.

Cosine scheduling has good general literature support, but it competes with a schedule that has been carefully calibrated through many experiments. Because schedule-only second drops have recurring failures and EXP-000's cosine bundle undertrained, cosine is better held as a later, clean schedule-only experiment if scalar optimizer probes are exhausted.

Bounded late EMA targets a real late-plateau mechanism, but the local history around averaging is fragile. It is more complex to implement, easier to slow down, and more likely to introduce an invalid or crash outcome than the LR probe. After EXP-042's interrupted run, the next experiment should favor a reliable, interpretable path.

## Chosen Idea
**Selected**: Initial LR 0.08 on the 2e-4 Anchor

**Why this idea**:
It is the clean missing side of the initial-LR bracket around the current best anchor. EXP-040 showed `LR=0.12` is too high; testing `LR=0.08` is the least risky way to learn whether the current recipe benefits from less high-LR noise while preserving every validated regularization and architecture choice.

**Hypothesis**:
Lowering `LR` from `0.1` to `0.08` while preserving the `2e-4` label-smoothed reflection anchor will reduce early high-LR noise and improve the late post-drop plateau enough to reach `best_test_acc >= 94.07%`.
