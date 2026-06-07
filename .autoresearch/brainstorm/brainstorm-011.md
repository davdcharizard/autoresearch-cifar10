# Brainstorm EXP-011
**Created**: 2026-05-28
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Chosen Idea
**Selected**: k=4 + Dropout(0.3) before FC

Simplest possible regularization addition. Dropout before the final classifier is standard for wide models. Single line of code. Zero compute cost. EXP-007 config unchanged otherwise.

**Hypothesis**: Dropout(0.3) before FC will improve from 95.73% to ~95.8-96.0%.
