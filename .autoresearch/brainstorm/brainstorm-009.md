# Brainstorm EXP-009
**Created**: 2026-05-28
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Experimental History Review

Baseline: 95.73% (EXP-007). Three consecutive failures (k=6, TrivialAugment, stochastic depth). Need a clean, single-variable improvement.

Pre-activation was tested in EXP-005 but confounded with k=6 (which failed due to insufficient epochs). Pre-activation itself was never tested at k=4 where epoch count is safe.

## Candidate Ideas

### 1. Pre-activation Blocks at k=4

**Summary**: Convert BasicBlock from post-activation (Conv→BN→ReLU→Conv→BN→+→ReLU) to pre-activation (BN→ReLU→Conv→BN→ReLU→Conv→+). Add bn_final before pooling. Keep all other settings from EXP-007 (k=4, EMA, WD=5e-4, CutMix, T_max=49).

**Reasoning**: Pre-activation improves gradient flow through identity shortcuts. He et al. 2016 showed consistent improvement on CIFAR-10. This is the only well-evidenced architectural change we haven't tested at the right scale. Single-variable change from proven baseline.

**Sources**: He et al. 2016 (Identity Mappings in Deep Residual Networks)

**Estimated Effort**: medium (restructure BasicBlock)

**Risk Assessment**: Low. Single architectural change, well-evidenced. Same param count and similar compute.

## Chosen Idea

**Selected**: Pre-activation Blocks at k=4

**Hypothesis**: Pre-activation blocks will improve from 95.73% to ~95.9-96.1% through better gradient flow.
