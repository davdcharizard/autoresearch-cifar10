# Brainstorm EXP-035
**Created**: 2026-06-04
**Goal**: goals/maximize-cifar10-test-accuracy.md

## CRITICAL DISCOVERY

Running the exact baseline code (zero modifications) on the current system gives:
- 48 epochs (not 54-58 like the original baseline system)  
- 93.99% accuracy (not 96.39%)

The system is ~15% slower. T_max=49 was tuned for ~54 epochs (54-5=49 cosine epochs). With only 48 epochs, the cosine gets 43 steps (48-5), leaving LR at 0.004 instead of reaching 0.

**ROOT CAUSE**: ALL experiments since EXP-030 failed because of T_max misalignment, NOT because of the individual code changes tested. The T_max must be reduced to 43 to match the current system.

## Chosen Idea
**Selected**: T_max=43 (aligning cosine schedule to current system's 48 epochs)

**Hypothesis**: With T_max=43, the cosine LR schedule will complete at epoch 48 (5 warmup + 43 cosine), restoring the proper decay profile. This should recover the model's convergence quality and achieve accuracy comparable to the original 96.39% baseline (adjusted for system speed).
