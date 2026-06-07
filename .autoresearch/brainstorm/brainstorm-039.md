# Brainstorm EXP-039
**Created**: 2026-06-04
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Key Insight
The EMA model is evaluated, not the training model. Improving EMA quality is as valuable as improving training accuracy. The fixed EMA_DECAY=0.999 is a compromise — too slow for early training (model changes fast, EMA lags behind) and too fast for late training (model is near convergence, should average more).

## Chosen Idea
**Selected**: Progressive EMA decay (0.99 → 0.9999) + T_max=43

**Implementation**: Linear interpolation from 0.99 to 0.9999 based on training progress (epoch / total_expected_epochs). Early: fast EMA tracking (0.99). Late: heavy smoothing (0.9999).

**Hypothesis**: Progressive EMA improves the EMA model by: (1) tracking faster early → EMA stays relevant, (2) averaging more late → EMA is smoother near convergence. This should improve accuracy without needing extra epochs.
