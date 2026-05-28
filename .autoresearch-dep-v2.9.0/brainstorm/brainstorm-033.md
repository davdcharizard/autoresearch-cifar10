# Brainstorm EXP-033
**Created**: 2026-05-28
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review
No new search.

## Experimental History Review
- **Baseline**: 96.56% (EXP-031, Nesterov+reflect)
- **EXP-032**: Alternating flip alone gave 96.64% (+0.08pp) — just 0.02pp below 96.66% threshold
- **Strategy**: Add one more zero-cost change on a 4th axis to compound past threshold

## Candidate Ideas

### 1. Alternating Flip + WD 4e-4 (reduced from 5e-4)
**Summary**: Combine alternating flip augmentation with slightly reduced weight decay (5e-4 → 4e-4). This stacks four zero-cost changes on four axes: optimizer (Nesterov), data quality (reflect padding), augmentation pattern (alternating flip), and regularization strength (WD reduction).

**Reasoning**: Alternating flip added +0.08pp in isolation (EXP-032). We need +0.02pp more. Slight WD reduction allows larger weights in the final convergence phase, potentially improving discriminative capacity for hard examples. The reduction is conservative (20%) to avoid the regularization-collapse risk seen with larger changes.

**Estimated Effort**: low — two changes (alternating flip + WD constant)
**Risk Assessment**: Low. 20% WD reduction is conservative. Alternating flip already proven at +0.08pp.

## Chosen Idea
**Selected**: Alternating Flip + WD 4e-4

**Hypothesis**: Combining alternating flip (+0.08pp) with slight WD reduction will compound to >= +0.10pp (to 96.66%+) by adding deterministic balanced orientation training plus slightly relaxed weight regularization.
