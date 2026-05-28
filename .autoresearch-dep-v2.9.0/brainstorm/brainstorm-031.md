# Brainstorm EXP-031
**Created**: 2026-05-28
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

No new search needed.

## Experimental History Review

- **Current best**: 96.46% (EXP-020). Seven consecutive failures (EXP-024-030).
- **Near-miss results**: EXP-026 Nesterov +0.06pp (optimizer), EXP-022 reflect padding + Cutout +0.07pp (data quality). Both individually below 0.1pp threshold.
- **Failed combinations**: EXP-027 (Nesterov + warmup) combined two changes on the SAME axis (optimizer/schedule) and hurt. The lesson: combine changes on DIFFERENT axes.
- **Remaining viable strategy**: Stack two orthogonal near-miss changes on different axes — optimizer quality (Nesterov) + data quality (reflect padding).

## Candidate Ideas

### 1. Nesterov + Reflect Padding in RandomCrop
**Summary**: Combine `nesterov=True` in SGD with `padding_mode='reflect'` in `transforms.RandomCrop`. Two zero-cost changes targeting different axes: optimizer gradient quality and training data quality at crop borders.

**Reasoning**: Nesterov (+0.06pp in EXP-026) improves gradient estimates via look-ahead. Reflect padding replaces zero-filled borders in crops with mirrored pixels, providing more natural image context. These target different mechanisms — the optimizer never sees the padding mode, and the padding mode doesn't affect gradient computation. EXP-027 failed because it combined Nesterov with warmup shortening (same optimization axis). This combination is on orthogonal axes.

**Sources**: EXP-026 (Nesterov +0.06pp), EXP-022 (reflect padding was part of +0.07pp, bundled with Cutout swap)

**Estimated Effort**: low — two parameter changes

**Risk Assessment**: Very safe. Both changes are individually proven harmless. The isolated effect of reflect padding is unknown (was bundled with Cutout swap in EXP-022), adding uncertainty. If effects are additive: ~+0.10pp (threshold). If not: likely in the 96.5% noise band.

## Chosen Idea
**Selected**: Nesterov + Reflect Padding in RandomCrop

**Why this idea**: Only remaining viable strategy after seven failures — stack two individually-proven near-miss changes on orthogonal axes.

**Hypothesis**: Combining Nesterov (+0.06pp) with reflect padding will improve best_test_acc to 96.56-96.60% by stacking better gradient estimates with higher-quality crop border data. Both are zero throughput cost.
