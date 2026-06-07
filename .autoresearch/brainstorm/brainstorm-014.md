# Brainstorm EXP-014
**Created**: 2026-05-29
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Experimental History Review

Baseline: 95.73% (EXP-007, k=4 + SGD + EMA + WD=5e-4). Custom VGG-style architecture (EXP-013) failed to beat it.

**Untried dimension**: AdamW optimizer. Every experiment so far used SGD. AdamW has:
- Per-parameter adaptive learning rates → better convergence in limited epochs
- Decoupled weight decay → cleaner regularization
- Standard settings (lr=1e-3, wd=0.05) are well-established
- Could enable k=5 or higher to converge where SGD fails

## Chosen Idea

**Selected**: k=4 ResNet + AdamW (lr=1e-3, wd=0.05) + EMA

Replace SGD with AdamW. Remove the Nesterov/momentum params. Use lr=1e-3, weight_decay=0.05 (standard AdamW settings, note WD is much higher for AdamW than SGD since it's decoupled). Keep everything else from EXP-007: k=4, EMA, CutMix, label smoothing, AMP, compile, T_max=49.

**Hypothesis**: AdamW's adaptive per-parameter LR will lead to faster and better convergence than SGD, improving from 95.73% to ~96.0-96.3%.
