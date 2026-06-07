# Brainstorm EXP-034
**Created**: 2026-06-04
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Experimental History Review
- 35 experiments, baseline 96.39%, 18 consecutive failures
- Recent experiments getting fewer epochs (47-48 vs baseline 54-58), likely from data/ re-download
- DataLoader currently does NOT use persistent_workers — workers respawn every epoch
- persistent_workers=True keeps workers alive, eliminating respawn overhead → more epochs → better convergence

## Chosen Idea
**Selected**: persistent_workers=True in DataLoader

**Why**: Free speedup. Workers persist between epochs, eliminating process spawn overhead (~0.5s per epoch × ~54 epochs = ~27s saved = ~5 more epochs). No training changes, zero risk.

**Hypothesis**: persistent_workers=True will give ~58-60 epochs (up from ~54), improving convergence and pushing best_test_acc above 96.49%.
