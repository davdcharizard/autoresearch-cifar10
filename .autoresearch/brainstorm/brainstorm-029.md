# Brainstorm EXP-029
**Created**: 2026-05-29
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Experimental History Review
- 30 experiments, 96.39% baseline, 13 consecutive failures
- EXP-018: channels_last gave 59 epochs (9% speedup) but T_max=55 hurt (slower decay → 96.11%)
- EXP-019: channels_last + T_max=49 gave 64 epochs but CosineAnnealingLR restart degraded model (96.28% best, 95.38% final, 0.9% gap)
- ROOT CAUSE IDENTIFIED: CosineAnnealingLR is periodic. After T_max steps, LR rises again.
- SOLUTION: Keep channels_last + T_max=49 + clamp LR to minimum after cosine completes. Extra epochs at near-zero LR give free refinement without restart degradation.

## Chosen Idea
**Selected**: Channels_last + LR clamp after cosine completion

**Why this idea**: Addresses the ROOT CAUSE of EXP-018/019 failures. Channels_last speedup is real (9%). T_max=49 decay rate is optimal. The clamp prevents the periodic restart that degraded EXP-019.

**Hypothesis**: Channels_last with T_max=49 and LR clamped to minimum after epoch 54 will yield ~59-64 epochs, with 5-10 extra refinement epochs at near-zero LR. This will improve best_test_acc to ≥96.49%.
