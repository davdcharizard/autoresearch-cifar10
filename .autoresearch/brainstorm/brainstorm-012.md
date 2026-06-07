# Brainstorm EXP-012
**Created**: 2026-05-28
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Chosen Idea
**Selected**: NUM_BLOCKS=4 (ResNet-26) at k=4 + T_max=42

Add depth through more blocks. 12 total blocks instead of 9. ~5.8M params, est 45-50 epochs. T_max=42.

**Hypothesis**: More depth at k=4 will improve from 95.73% to ~95.9-96.1%.
