# Brainstorm EXP-003
**Created**: 2026-05-28
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

No new sources — building on EXP-001 and EXP-002 findings.

## Experimental History Review

**Current state** (3 experiments):
- BASE: 91.81%
- EXP-000: 92.10% (+0.29%), recipe improvements
- EXP-001: 94.03% (+1.93%), k=2 + AMP + compile (T_max=55, actual 78 epochs)
- EXP-002: 94.09% (no-improvement), k=3 + broken T_max calibration + CutMix

**Critical data from EXP-002**: k=3 achieves 62 epochs in 300s with AMP+compile. The T_max calibration was the sole failure cause — epoch 1 torch.compile JIT overhead inflated timing ~4.5x. The model itself had 2.4M params and ran fine.

**Key patterns**:
- Width is primary lever (Medium, confirmed across EXP-000/001)
- T_max must match actual epochs (High, reinforced by EXP-002)
- k=3 gets 62 epochs → correct T_max = 62 - 5 (warmup) = 57

## Candidate Ideas

### 1. k=3 + T_max=57 + CutMix (retry with fix)

**Summary**: Exact same approach as EXP-002 but with correct static T_max=57 (derived from EXP-002's actual 62 epochs). k=3 ({48,96,192}, 2.4M params) with CutMix(alpha=1.0, p=0.5), AMP, torch.compile, Nesterov, projection shortcuts, label smoothing 0.1.

**Reasoning**: EXP-002 failed purely due to T_max calibration. The k=3 model ran 62 epochs and achieved 94.09% best even with T_max=10 (cosine finished at epoch ~15). With proper T_max=57, the cosine schedule will span the full training, and the model should significantly exceed 94.03%.

**Sources**: EXP-002 data (62 epochs measured), EXP-001 analysis, CutMix paper

**Estimated Effort**: low (single config fix from EXP-002)

**Risk Assessment**: Very low. The only change from EXP-002 is T_max=57 instead of dynamic calibration. Architecture is identical and ran successfully.

### 2. k=2 + T_max=73 (fix only)

**Summary**: Keep EXP-001's k=2 model but fix T_max from 55 to 73 (78 actual - 5 warmup). No other changes.

**Reasoning**: EXP-001's 2.1% best/final gap suggests just fixing T_max on k=2 could push from 94.03% to higher. Minimal risk.

**Sources**: EXP-001 analysis

**Estimated Effort**: low

**Risk Assessment**: Very low but lower ceiling than k=3.

## Idea Evaluation

Idea 1 is clearly superior: it fixes the known T_max issue AND leverages more capacity (k=3, 2.4M vs 1.08M params). The risk is minimal since k=3 architecture is proven to run successfully from EXP-002. Idea 2 is lower risk but also lower reward.

## Chosen Idea

**Selected**: k=3 + T_max=57 + CutMix (retry with fix)

**Why this idea**: Direct fix of EXP-002's sole failure cause. The architecture and augmentation are proven to run; only T_max was wrong. We have exact epoch data (62) from EXP-002 to set T_max correctly.

**Hypothesis**: k=3 with correct T_max=57 and CutMix will improve best_test_acc from 94.03% to approximately 95-96%, driven by 2.25x capacity over k=2 with proper LR scheduling.
