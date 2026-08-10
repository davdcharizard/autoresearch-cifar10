# EXP-018 Focused Plan Re-Review

## Verdict: APPROVED

No blocking plan issue remains. Claude confirmed that all prior blockers are concretely resolved and found no new contradiction, false PASS/FAIL path, infeasible threshold, timer/evaluator reward hack, or scope violation.

## Confirmed Resolutions

1. `[86%,98%)` provides about eight projected endpoints, with preflight requiring eight and production flooring at seven.
2. Controllers must call importable production `train.py` helpers rather than reimplement SWA/BN logic.
3. Quantitative endpoint-spread floors replace a bare nonzero check.
4. Merge requires final SWA accuracy itself to clear 94.25 and the pre-SWA online best.
5. BN refresh uses cumulative momentum with original-momentum restoration.
6. Persisted `install_step == num_steps` proves no post-install optimizer step.
7. One joint conservative projection gates snapshot count, exposure, refresh, evaluations, and wall time.
8. Refresh explicitly recreates iterators across persistent weak-loader exhaustion.

## Non-Blocking Watch Items

- Measure warmed weak-epoch timing faithfully because snapshot-count margin remains about one epoch.
- Treat the spread floor mainly as a frozen-shadow implementation guard; the final-SWA-vs-online-best gate carries attribution integrity.
- The 26,091 actual-step floor is deliberately stricter than the goal and protected by the 26,200 preflight projection.
- Implement an explicit one-shot `swa_finalized` guard even though the refreshed counter should naturally terminate the outer loop.

## Provenance

- Reviewer: external Claude CLI, mandatory no-fallback path
- Command outcome: exit code 0
- Completed: 2026-08-06

