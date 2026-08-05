# EXP-013: Late Whole-State EMA

## Execution
- **Created**: 2026-07-24
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-test-accuracy-013
- **Commit**: (pending)
- **PR**: N/A - local/offline
- **Outcome**: failed - valid result below margin

## Implementation Notes
### Summary
Added external whole-state EMA at 65%/0.999 with in-place exception-safe evaluation swap. Live training objects and optimizer path are unchanged.
### Surprises & Discoveries
Partial-copy failure testing required a non-broadcastable shadow shape; scalar shape broadcast silently. EMA retained 99.05% throughput.
### Decisions
Shadows are an unregistered plain dict. Floating state is interpolated, integral counters copied, and init direct-copy counts as update one.

## Experimental Adjustments
None.

## Run Log
### Run 1
Metadata:
- **Job ID**: local exec session 93030
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.3-gpt-5-6-sol/run.log`
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-07-24 17:09 UTC
- **Ended**: 2026-07-24 17:15 UTC

Description:
- One fixed-seed accepted WRN run with whole-state EMA initialized at the 65% hard-label boundary and decay 0.999. Preflight projects 140.55 passes. Success requires 94.17%; no policy fallback.

Observations:
- Semantic preflight passed complete state coverage, external ownership, live/optimizer equality, normal and partial-failure restoration, finite FP32 state, and exact count. Timing aggregates accepted 10.296559 / candidate 10.395423 ms gave retention 0.990490 and 140.550475 projected passes; max CV 0.007904. (source: preflight stdout)
- Run exited 0 after 300.0 counted / 340.5 total seconds. EMA initialized once at step 17,738 / 195.0 seconds and made exactly 9,795 updates, equal to 27,533-17,738. Best EMA accuracy was 94.10%; terminal was 93.79% / 0.2596 loss. (source: `run.log`)

Key Metrics:
- best/final: 94.10% / 93.79%; delta +0.03 vs baseline, 0.07 below 94.17%; final loss 0.2596.
- exposure: 27,533 steps = 140.96896 passes; 142 epochs; 29 evaluations; 1,094.0 MiB peak; 691,674 parameters.

## Verification Results
### Conditions Checked
- **Protocol**: PASS - one H20, one run, exact architecture, 300.0/340.5 seconds, 140.97 passes, one transition/init, exact update count, finite state, accepted cadence.
- **Metric**: FAIL - 94.10% is below required 94.17%; no rerun.
### Informational Metrics

## Errors & Dead Ends
### 2026-07-24 - Weak partial-swap fault injection
- Error: `partial swap failure not injected`
- Root cause: one-element shadow broadcast during `copy_` instead of failing.
- Source: first semantic preflight
- Do NOT retry: use broadcastable shapes for partial-copy exception tests.

## Human Notes
> Autopilot; no intervention.
