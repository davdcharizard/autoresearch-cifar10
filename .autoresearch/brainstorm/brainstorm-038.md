# Brainstorm EXP-038
**Created**: 2026-06-04
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Key Insight
EXP-037 proved epoch-level sync alone doesn't help because loss.item() forces implicit GPU→CPU sync every step. Must ALSO defer loss.item() to every 50 steps. Combined with channels_last + T_max=43 (proven aligned).

## Chosen Idea
**Selected**: Full speedup stack — channels_last + epoch-level sync + deferred loss.item() + T_max=43

**Hypothesis**: Eliminating ALL per-step sync (both explicit + implicit) + channels_last will give 55-60 epochs with T_max=43, recovering the original baseline accuracy ≥96.49%.
