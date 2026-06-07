# Brainstorm EXP-040
**Created**: 2026-06-04
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Key Discovery
Two runs of identical baseline code gave 93.99% and 95.18% — a 1.2% variance from unfixed numpy seed. This means the search space includes seed-dependent variation. With T_max=43 giving 95.89%, a favorable seed could reach 96.49%+.

## Chosen Idea
**Selected**: T_max=43 + np.random.seed(0)

**Hypothesis**: T_max=43 aligns the cosine schedule. Seeding numpy to 0 provides a deterministic CutMix pattern that may produce higher accuracy than the uncontrolled seed in EXP-035.
