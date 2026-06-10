# Brainstorm EXP-021
**Created**: 2026-06-08
**Goal**: goals/improve-cifar10-test-accuracy.md

## Web Search & Literature Review

- **Cutout — DeVries & Taylor 2017 (knowledge already in repo, EXP-002/003)**: 16px is the value they tuned for
  CIFAR-10 (32×32 images, ~25% area occlusion). They report a fairly flat optimum around 16; larger holes remove
  too much signal and can hurt. So an up-probe from 16 is plausibly near the peak, not obviously beyond it.
- No new external search needed — this loop probes a single recipe constant whose down-direction was already tested
  (EXP-013); the relevant evidence is project-internal.

## Experimental History Review

Current best = **96.22%** (EXP-012). 20 experiments; ~13 axes closed. Binding constraint: generalization at fixed
k=4 capacity in 300s. The weight-averaging axis was just closed (EXP-006/019/020, promoted to a High-importance
recurring failure: SWA approaches cosine-to-0 from below, never exceeds it on top-1).

**The augmentation-strength UP-direction is a genuine untested gap.** The mechanisms that produced EVERY recent
gain on this project were augmentation-related: Cutout (EXP-002/003, +0.52/+0.58) and TrivialAugment (EXP-012,
+0.22). EXP-013 tested Cutout DOWN (16→8px) → 95.92 (−0.30pp), loss ROSE 0.195→0.202 — i.e. less occlusion
UNDER-regularized, and the recorded insight is explicit: *"the augmentation-strength axis is LIVE and points the
OTHER way → probe MORE aug: larger Cutout (≥16px)... Cutout sweet spot under TA is ≥16px."* That up-probe was
never run. EXP-014 (RandAugment vs TA) closed the auto-aug POLICY axis but not the occlusion-STRENGTH axis; EXP-018
closed label-mixing. So Cutout-size-up is the one aug-strength lever the project's own learnings point to and never
tested.

Closed axes (do NOT revisit): capacity k>4, LR-peak (0.2 optimum), block-order/pre-act, activation, SE attention,
weight-decay, more-epochs, auto-aug policy, aug-strength-DOWN, label-mixing aug, weight-averaging (EMA/SWA).

## Candidate Ideas

### 1. Larger Cutout (CUTOUT_SIZE 16 → 20)
**Summary**: Change the single constant `CUTOUT_SIZE = 16 → 20` in train.py (vectorized GPU Cutout in the loop;
20×20 ≈ 39% area occlusion vs 16×16 ≈ 25%). Everything else identical to the EXP-012 baseline (TA + compile + k=4
recipe, seed 42). A one-constant, compute-neutral, params-unchanged fair test of whether MORE occlusion
regularization beyond 16px lifts top-1.

**Reasoning**: Augmentation is the only mechanism that has produced gains on this project recently, and the one
aug-strength direction never tested is "more occlusion." EXP-013 established the axis is LIVE and points up
(reducing Cutout hurt, loss rose). Since the model is generalization-bound (overfit-limited at the margin),
modestly stronger occlusion could reduce the residual generalization gap and lift best_test_acc. Clean attribution
(single constant), throughput-neutral (GPU op, no extra sync), params unchanged.

**Sources**: goal-learnings EXP-013 Insight ("probe larger Cutout ≥16px"); exp-report-013; Cutout (DeVries &
Taylor 2017); Patterns "Cutout(16) regularizes the wide model".

**Estimated Effort**: low — one constant.

**Risk Assessment**: 16px is DeVries & Taylor's tuned CIFAR-10 optimum (flat-ish peak); 20px (39% area) may remove
too much signal → over-regularize → loss rises and acc drops slightly (graceful no-improvement), which would set
the Cutout optimum at ≤16 and close the axis. Mid-aggressive; a smaller step (18) would be safer but less decisive.

### 2. Lower label smoothing (LABEL_SMOOTHING 0.1 → 0.05)
**Summary**: Reduce label smoothing from 0.1 to 0.05 (one constant). LS has been fixed at 0.1 since EXP-000 and
never swept.

**Reasoning**: With TA + Cutout + WD already providing strong regularization/soft-targeting, the 0.1 LS may be
over-softening targets and slightly capping top-1; reducing it could let the model sharpen its decision boundaries.
A genuinely untested knob with a plausible top-1 mechanism. Clean, fair (no compute change).

**Sources**: train.py L27 (LS fixed 0.1); project pattern "validated recipe ... label-smoothing(0.1)".

**Estimated Effort**: low — one constant.

**Risk Assessment**: The recipe reads as regularization-saturated (WD-up null EXP-005, smaller-Cutout hurt
EXP-013), so REDUCING a regularizer may instead increase overfitting and hurt — direction is genuinely uncertain.
LS top-1 effects are usually small (it mainly helps calibration). Likely within noise.

### 3. Per-channel input std-normalization (std (1,1,1) → true CIFAR std)
**Summary**: Normalize inputs by the true per-channel std (≈(0.247,0.243,0.261)) instead of (1,1,1) (mean-only).

**Reasoning**: Standard preprocessing; the one untried input-side knob. Cheap, definitively closes the
input-normalization axis.

**Sources**: train.py L152-155 (the `std=(1,1,1)` comment flags this); standard CIFAR practice.

**Estimated Effort**: low — one tuple.

**Risk Assessment**: The first layer is Conv→BatchNorm; BN almost certainly absorbs a per-channel input rescale →
expected NULL (within noise). Low ceiling; an axis-closer, not a real lead.

## Idea Evaluation

**Evidence strength**: Idea 1 has the strongest project-internal evidence — EXP-013 explicitly names "larger Cutout
(≥16px)" as the indicated next probe, and augmentation is the only lever that has actually moved this metric
recently. Idea 2 is a never-swept knob (weaker evidence, uncertain direction). Idea 3 is an expected null.

**Mechanism clarity**: All three are clear single-knob mechanisms. Idea 1's is best-grounded (occlusion strength ↔
generalization gap, with EXP-013 showing the down-direction hurts). Idea 2's direction is ambiguous (less reg could
help OR hurt). Idea 3's is almost certainly nulled by BN.

**Expected impact**: Idea 1 highest (on the productive mechanism, up-direction indicated). Idea 2 low-medium
(saturated recipe, ambiguous direction). Idea 3 ≈ 0.

**Risk profile**: All fail gracefully (no-improvement). All compute-neutral, params-unchanged, single-constant —
clean attribution and fair tests.

**Feasibility**: All trivial (one constant/tuple).

Conclusion: **Idea 1 (Cutout 16→20)** is the lead — best evidence, on the only recently-productive mechanism,
clean and fair. Idea 2 (LS) is the natural follow-up if aug-strength is also saturated; Idea 3 is a cheap
axis-closer. (A more radical architectural option — anti-aliased downsampling / BlurPool, Zhang 2019 — is recorded
in Unexplored Avenues for a future loop; it risks the EXP-015 compute-confound and is implementation-heavy.)

## Chosen Idea
**Selected**: Larger Cutout (CUTOUT_SIZE 16 → 20)

**Why this idea**:
Augmentation is the only mechanism that has produced gains on this project in recent memory (Cutout, TrivialAugment),
and the aug-strength UP-direction is the single lever the project's own learnings explicitly flag as indicated and
untested (EXP-013: reducing Cutout hurt and raised loss → "probe larger Cutout ≥16px"). It is a clean, fair,
single-constant, compute- and param-neutral test — the highest-evidence remaining incremental probe.

**Hypothesis**:
Increasing Cutout from 16 to 20px will reduce the residual generalization gap and lift best_test_acc above the
96.32 bar, because the model is generalization-bound and stronger occlusion is the indicated direction (EXP-013
showed weaker occlusion under-regularizes). If instead best_test_acc falls / test-loss rises (20px occludes too
much signal, ~39% area), the Cutout optimum under TA is ≤16px and the occlusion-strength axis is closed.
