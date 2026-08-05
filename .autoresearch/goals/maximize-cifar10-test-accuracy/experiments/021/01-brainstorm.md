# Brainstorm EXP-021
**Created**: 2026-07-26

## Web Search & Literature Review

- **Mixup** (`knowledge/papers/mixup.md`): beta alpha controls interpolation strength; local evidence validates batch-shared sampling and alpha 0.2.
- **Time Matters in Regularizing Deep Networks** (`knowledge/papers/time-matters-regularization.md`): retain the now-bracketed 65% cutoff and isolate strength rather than changing duration again.
- **EXP-017 to EXP-020 reports**: attention interaction is costly and fragile, while the mixup duration optimum is bracketed around 65%; a one-parameter strength probe is the remaining clean local gap.

No network source was consulted; this loop uses offline persistent knowledge and completed experiment artifacts.

## Experimental History Review

- The accepted 94.07% recipe uses batch-shared Beta(0.2,0.2) mixup until 65%. The duration is now bracketed: 50% scored 93.91 and 75% scored 93.82 at normal exposure.
- Stronger alpha 0.4 scored 93.57, and per-example alpha-0.2 coefficients scored 93.79. These close stronger softness and coefficient decorrelation, but the weaker batch-shared side remains untested.
- Stage-3 attention produced the closest 94.16 result only with two conditional gates; final-only/static approximations regressed. Removing diagnostics from the full gates is a narrow unresolved efficiency question, not a new mechanism.
- The current limiter is precise regularization calibration rather than fit or exposure. Alpha 0.1, full two-gate SE without observation, and short-window SAM are the remaining bounded options, each with modest or speculative upside.

## Collected Ideas

## Combinations

## Candidate Ideas

### Ten-Percent Early-Window SAM
**Summary**: Apply non-adaptive SAM with rho 0.05 only for the first 10% counted time, restoring BatchNorm buffers between the two passes, then use exact accepted SGD for the remaining 90%.

**What it targets**: Optimization geometry rather than regularization strength, capacity, or exposure.

**Reasoning**: This is the main qualitatively distinct idea left and confines double-pass cost to 30 seconds. It may find a flatter basin before accepted training takes over.

**Sources**: `knowledge/papers/time-matters-regularization.md`; EXP-013/017/020 history; prior unscored proposal in `experiments/015/proposals/idea-03.md`.

**Estimated Effort**: high

**Risk Assessment**: Rho/window are uncalibrated, BatchNorm restoration is complex, and lost early updates near peak LR may dominate. There is no local sharpness diagnostic supporting the premise.

### Two-Gate SE Without Runtime Diagnostics
**Summary**: Recreate EXP-017's exact two stage-3 SE gates with its preregistered seed and remove all per-forward diagnostic accumulation, changing fixed-time exposure but not gate semantics.

**What it targets**: The 94.16 near miss and measured observation overhead while preserving the only positive conditional interaction.

**Reasoning**: EXP-018/019 show both conditional gates matter. A direct timing preflight could establish whether removing observation recovers meaningful passes before scoring.

**Sources**: EXP-017/018/019 reports and high-importance attention learning.

**Estimated Effort**: medium

**Risk Assessment**: Diagnostic reductions may be cheap, making this a noise-level replay; reusing a known near-positive initialization also creates seed-selection concerns.

### Weaker Alpha-0.1 Mixup
**Summary**: Change only `MIXUP_ALPHA` from 0.2 to 0.1 while retaining the accepted 65% cutoff and batch-shared coefficient/permutation behavior.

**What it targets**: Mixup strength at the now-fixed duration, reducing the frequency of strongly interpolated inputs/targets while preserving stochastic batch-level regularization.

**Reasoning**: Alpha 0.4 was too strong; the lower side is the only untested strength direction. A one-line change preserves exposure and cleanly closes whether accepted mixup is overly soft even though the under-regularization prior is substantial.

**Sources**: EXP-002/005/015/020; `knowledge/papers/mixup.md`; `03-experiment-learnings.md`.

**Estimated Effort**: low

**Risk Assessment**: Beta(0.1,0.1) concentrates near endpoints and may under-regularize, consistent with the failed 50% window. Alpha response need not be monotonic around 0.2.

## Review

The blind review selected SAM at 6/10 evidence and 7/10 impact but found the proposed early window mechanistically inconsistent because 90% later SGD could erase it. I adopt a final 10% counted-time window at fixed rho 0.05, when flatness is relevant to convergence. The plan must verify exact weight restoration and byte-restored BatchNorm buffers after the perturbation pass. Alpha 0.1 fights under-regularization evidence, and two-gate SE without diagnostics is a seed-selected noise-scale replay. Full review: `01-idea-review.md`.

## Idea Evaluation

Adopt the verdict in `01-idea-review.md` with the recommended late-window refinement. SAM is the only remaining candidate with a mechanism and effect ceiling distinct enough to clear the margin; the other candidates mainly close low-upside gaps.

## Chosen Idea
**Selected**: Final-Ten-Percent SAM

**Why this idea**:
Apply standard non-adaptive SAM only after 90% counted progress, preserving the accepted trajectory through its entire high-LR and most low-LR training. This targets final solution geometry while bounding double-pass cost to 30 seconds and preventing subsequent plain SGD from erasing the intervention.

**Hypothesis**:
Using rho 0.05 SAM only during the final 10% counted time, with exactly one persistent BatchNorm update and exact perturbation restoration per step, will retain at least 90% effective fixed-time exposure and raise fixed-seed `best_test_acc` from 94.07% to at least 94.17% by improving terminal basin flatness.
