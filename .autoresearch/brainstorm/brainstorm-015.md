# Brainstorm EXP-015
**Created**: 2026-05-29
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Chosen Idea
**Selected**: Mixup (alpha=0.2) replacing CutMix + EMA decay 0.9995

Two targeted changes on the proven k=4+EMA baseline:
1. Replace CutMix with Mixup(alpha=0.2) — smoother augmentation (global blend vs hard rectangular cut). On 32x32 images, CutMix's rectangular cuts cover large portions and create unrealistic features.
2. EMA decay 0.999 → 0.9995 — doubles the averaging half-life from ~1.8 to ~3.6 epochs, giving a smoother final model.

**Hypothesis**: Mixup + slower EMA will improve from 95.73% to ~96.0%.
