# EXP-018 Implementation Addendum Review

## Verdict: APPROVED

Claude found no blocking production correctness, scope, metric-integrity, or reward-hacking issue. It traced and approved uniform-mean recursion, detached storage, ordered installation, state/RNG nonmutation, timer charging, snapshot/finalization evaluation ordering, install-step enforcement, cumulative BN refresh, hard targets, iterator recreation, and tracked scope.

## Prioritized Findings

### 1. Timing controller had non-functional feasibility gates

`projected_evaluations` and `projected_wall_seconds` were fixed literals, so their gates could not fail. `projected_steps` scaled historical EXP-010 steps by budget fraction while ignoring the five newly measured step times. Fix by deriving steps from synchronized measured wall-step time, deriving evaluations from measured weak-epoch duration, and measuring evaluator wall time.

### 2. Snapshot margin remains about one epoch

Nominal production projects eight endpoints with a floor of seven. The preflight's eight-snapshot requirement is the protection; require warmed conservative weak-epoch timing and report the margin explicitly before launch.

### 3. Refresh budget invariant was implicit

The 98% break currently leaves about six seconds, but the production code did not assert that the remaining budget can cover the measured 390-batch refresh. Add an explicit minimum refresh-budget assertion.

### 4. FP64-reference tolerance could false-fail near rounding margin

The fixed `atol=2e-6` is probably sufficient but unnecessarily tight for iterative FP32 lerp at values around 4.5. Use a scale-aware relative tolerance with a small absolute floor.

## Provenance

- Reviewer: external Claude CLI, mandatory no-fallback path
- Command outcome: exit code 0
- Completed: 2026-08-06

