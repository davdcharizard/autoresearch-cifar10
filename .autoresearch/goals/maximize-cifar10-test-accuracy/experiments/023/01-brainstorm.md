# Brainstorm EXP-023
**Created**: 2026-07-26

## Web Search & Literature Review

- **Wide Residual Networks** (`knowledge/papers/wide-residual-networks.md`): width can allocate CIFAR compute more effectively than extreme depth; local EXP-010 confirmed a small positive signal from selective low-resolution width.
- **RandAugment** (`knowledge/papers/randaugment.md`): input diversity can improve CIFAR generalization, but current local evidence and the accepted recipe make late or additive regularization poorly aligned with the clean-refinement requirement.
- **mixup** (`knowledge/papers/mixup.md`): retain the accepted batch-shared alpha-0.2 treatment and 65% cutoff while probing orthogonal model or systems changes.
- **EXP-017/019 local attention evidence**: two feature-driven stage-3 gates reached 94.16, while final-only and static approximations failed; conditional interaction rather than mean attenuation supplied the signal.

No network source was consulted; this offline loop uses persistent peer-reviewed distillations and completed local artifacts.

## Experimental History Review

- The accepted WRN-16-2 plus early mixup scores 94.07% in 141.9 passes. Twenty subsequent trials have not cleared 94.17%, so small standalone calibration changes have repeatedly fallen inside the rejection band.
- The two strongest architecture results are selective 160-channel stage-3 width at 94.11%/132.16 passes and two exact-neutral stage-3 SE gates at 94.16%/133.64 passes. They target related but non-identical mechanisms: representational capacity versus input-conditional channel selection.
- Exact `[2,2,3]` depth reached 94.15%, but efficient low-rank and fixed-MAC redistribution failed. Added capacity is useful only when it preserves early blocks and dense low-resolution representation.
- Stage-3 attention simplification is a high-importance recurring failure: final-only and static first-gate approximations lost the signal. Any cheap gate must preserve both placements and per-example conditioning rather than remove a gate or replace it with a constant.
- Mixup duration, late SAM, EMA, weight-decay removal, cosine-to-zero, dropout, CutMix, BF16, and residual endpoint initialization are closed. Lower loss and extra exposure alone have not guaranteed higher top-1.
- The limiting gap is a small generalization/representation boundary under a hard fixed-time budget. Candidates need enough upside to exceed 0.10 points while preserving the accepted early/high-resolution path and terminal hard-label refinement.

## Collected Ideas

## Combinations

## Candidate Ideas

### Two Diagonal Conditional Stage-3 Gates
**Summary**: Add exact-neutral gates to both stage-3 residual branches, but compute each channel scale as `2 * sigmoid(weight[c] * pooled_feature[c] + bias[c])`. Initialize weight and bias to zero so the accepted model is exact at construction; preserve per-example, per-channel conditioning with only 512 learned scalars and elementwise work.

**What it targets**: EXP-017's positive conditional-attention signal while eliminating the two small global MLPs and their measured 4.6% step overhead. Both gate placements remain intact, directly avoiding the final-only/static failures.

**Reasoning**: EXP-019 showed static channel attenuation is insufficient, not that every cheaper conditional mapping fails. A diagonal pooled-feature response restores input dependence at both blocks and can learn channel-specific sign/strength with negligible matrix compute. If global cross-channel mixing is not essential, exposure should recover while retaining the useful mechanism.

**Sources**: EXP-017, EXP-018, and EXP-019 reports; `03-experiment-learnings.md` recurring attention failure.

**Estimated Effort**: medium

**Risk Assessment**: Removing cross-channel interaction may be another destructive simplification despite retaining conditionality. Zero weight/bias gives exact unit scales and should still produce direct gradients proportional to the pooled feature, but a preflight must prove both parameter gradients open before scoring.

### Compose Selective Width with Full Two-Gate SE
**Summary**: Use explicit stage widths `[32, 64, 160]` and place exact-neutral ratio-16 squeeze-and-excitation on both 160-channel stage-3 residual branches. Initialize accepted common parameters identically and create both new gates from preregistered seed 23017 in a restored RNG fork; retain the full accepted training recipe and add no runtime gate diagnostics.

**What it targets**: The remaining low-resolution representation/selection gap by combining more dense 8x8 channels with input-conditional global channel interaction. Width supplies features that the gates can selectively route rather than asking either mechanism to clear the margin alone.

**Reasoning**: EXP-010 and EXP-017 are the only model changes with positive deltas (+0.04 and +0.09). Their nominal composition reaches beyond the +0.10 bar, and the gate may make added channels more useful instead of merely increasing redundant capacity. Both interventions preserve early blocks and the hard-label tail, unlike failed redistribution or terminal optimization changes.

**Sources**: `knowledge/papers/wide-residual-networks.md`; EXP-010 and EXP-017 reports; EXP-018/019 conditional-interaction failures.

**Estimated Effort**: medium

**Risk Assessment**: The mechanisms may be redundant rather than additive, combined overhead may reduce exposure, and composing two result-selected near misses increases benchmark-selection risk. Require exact semantic isolation and a preregistered >=125-pass timing gate; any score below 94.17 is sub-additive failure with no post-hoc rescue.

### FP32 Fused SGD for More Accepted Updates
**Summary**: Enable PyTorch's existing fused FP32 implementation on the unchanged two-group Nesterov SGD optimizer, with no model, data, schedule, seed, or evaluator changes. Score only if warm matched timing shows materially higher production-path exposure and optimizer-state semantics remain aligned.

**What it targets**: Fixed-budget update density without BF16's arithmetic change. The accepted model remains structurally identical, so any speed gain buys more low-floor clean refinement steps.

**Reasoning**: EXP-009 showed that extra exposure paired with BF16 numerics is harmful, not that exact FP32 throughput is useless. The H20 is underutilized by this small model and fused multi-tensor optimizer work may reduce launch overhead. This is a low-complexity way to test the systems headroom suggested after EXP-010.

**Sources**: EXP-009 and EXP-010 reports; `project-insights.md` H20 headroom and tensor-shape timing observations.

**Estimated Effort**: low

**Risk Assessment**: Optimizer work is a small share of the 10-12 ms step, so speedup may be negligible; fused accumulation order can alter the fixed-seed trajectory. Abort before scoring unless measured end-to-end retention improves by at least 2% and state/group semantics pass.

## Review

The blind reviewer selected the width-plus-full-SE composition at 6/10 evidence and 7/10 impact. I adopt its constraints: no training-time diagnostics, fixed new-gate seed 23017 in a restored RNG fork, matched production timing, and a fail-closed 125-pass projection. The hypothesis is super-additivity from routing newly supplied features, not numerical addition of result-selected deltas. The diagonal gate remains informative but likely weakens the only positive attention mechanism; fused SGD lacks an accuracy mechanism. Full review: `01-idea-review.md`.

## Idea Evaluation

Adopt the verdict in `01-idea-review.md`. The composition has the only credible ceiling above the acceptance margin and preserves the full conditional interaction demanded by prior failures. A result below 94.17 will be interpreted as redundancy/sub-additivity and close this exact composition.

## Chosen Idea
**Selected**: Selective 160-Channel Stage 3 with Full Two-Gate SE

**Why this idea**:
The two strongest local architecture signals supply complementary ingredients: dense low-resolution features and input-conditional channel routing. Keeping both gate placements and full cross-channel interaction avoids the recurring attention-simplification failure, while the reviewed 127-pass gate prevents an exposure-collapsed score.

**Hypothesis**:
Diagnostic-free exact-neutral ratio-16 SE on both residual branches of a 160-channel final stage, initialized from fixed seed 23017 without disturbing accepted common state/RNG, will project at least 127 passes and raise fixed-seed `best_test_acc` from 94.07% to at least 94.17% because conditional routing makes the added low-resolution channels more useful.
