# Brainstorm EXP-042
**Created**: 2026-06-04
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Chosen Idea
**Selected**: torch.compile mode='reduce-overhead' + T_max=43

mode='reduce-overhead' uses CUDA graphs to minimize Python dispatch overhead between GPU kernels. For our small model where per-step time is dominated by kernel launch overhead (not kernel compute), this could provide meaningful speedup.

Unlike max-autotune (which was slower), reduce-overhead targets the DISPATCH overhead which is proportionally larger for small models.
