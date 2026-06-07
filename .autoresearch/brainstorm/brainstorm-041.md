# Brainstorm EXP-041
**Created**: 2026-06-04
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Chosen Idea
**Selected**: torch.compile mode='max-autotune' + T_max=43

The default torch.compile mode may not generate the fastest kernels. mode='max-autotune' benchmarks multiple kernel implementations and selects the fastest. This trades compilation time (excluded from training budget) for faster per-step training speed, potentially yielding 2-3 extra epochs in the 300s budget.
