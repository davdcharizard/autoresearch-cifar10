# EXP-018 Corrected-Source Re-Review

## Verdict: APPROVED

Claude confirmed that the timing gates now depend on measured synchronized wall-step and evaluator times, projected steps use the correct measured units/direction, the 4.5-second refresh-budget assertion is present, and the arithmetic tolerance is scale-aware. It found no blocking correctness, scope, formula, or reward-hacking issue.

## Confirmed Fixes

- Projected snapshots and evaluations follow the production 240/258/294-second structure.
- Projected steps use `floor(1000 * available_seconds / conservative_step_ms)`.
- Projected wall uses fresh evaluator timing rather than a literal.
- Production cannot enter BN refresh with less than 4.5 counted seconds.
- FP32 recursive-mean validation uses relative tolerance plus an absolute floor.

## Non-Blocking Watch Items

- The evaluation upper-bound projection uses the slow conservative epoch, but 20 evaluations would require implausibly faster production on the same H20; runtime verification remains authoritative.
- Snapshot and exposure margins remain about one epoch and must be reported explicitly before launch.
- Synthetic one-step timing-child spread is not representative; production's own spread check carries mechanism integrity.

## Provenance

- Reviewer: external Claude CLI, mandatory no-fallback path
- Command outcome: exit code 0
- Completed: 2026-08-06

