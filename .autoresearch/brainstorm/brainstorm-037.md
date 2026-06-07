# Brainstorm EXP-037
**Created**: 2026-06-04
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Key Insight
torch.cuda.synchronize() is called EVERY training step (~19K times per run). This forces CPU-GPU synchronization, preventing the GPU from pipelining work. Removing it and tracking time at epoch boundaries allows the GPU to run at full efficiency.

## Chosen Idea
**Selected**: Epoch-level time tracking + channels_last + T_max=49 + LR clamp

**Changes**: Replace per-step synchronize with epoch-level wall-clock tracking. Add channels_last for additional speedup. T_max=49 + LR clamp for proper schedule.

**Hypothesis**: Removing per-step sync barrier + channels_last will give 60+ epochs, reaching the original baseline's epoch count and accuracy ≥96.49%.
