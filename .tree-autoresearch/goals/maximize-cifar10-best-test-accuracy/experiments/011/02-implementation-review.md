# Adversarial implementation review: EXP-011

**Reviewer**: Claude Opus via Claude Code 2.1.220
**Reviewed**: 2026-08-06, after correctness and GPU preflight, before the metric run

## Verdict

No blocking correctness, protocol-integrity, RNG, or timing concern was found. Claude explicitly confirmed that the EMA sample is post-optimizer and post-SAM-restore but pre-synchronization; the timestamp and progress share the same step-entry charged-time value; cadence-31 alternates ordinary/SAM parity; evaluation calls exactly one source; the fresh-state restoration proof is genuine; and the parent online path is value-identical.

## Prioritized concerns

1. A final audit exception occurred before the metric summary and would discard valuable diagnostics on an invalid run. Accepted: final evaluation-count, parity, and audit exceptions are now captured, an `ema_audit_failed` line and complete summary are printed, and the process then exits nonzero. Successful behavior is unchanged.
2. EMA-only tail evaluation cannot distinguish EMA harm from ordinary fixed-seed tail noise. Accepted as an interpretation limit imposed by the preregistered one-source, once-per-epoch protocol; analysis must not overclaim causality.
3. The projected 25,570 steps leave only 1.5% headroom over the 25,200 dose gate. Accepted under the preregistered first-valid preflight rule; a realized shortfall remains a no-improvement classification, not a rerun trigger.
4. Durable preflight evidence should state the weighting because transient harnesses are deleted. Addressed: the benchmark used `0.875 * ordinary_median + 0.125 * SAM_median`, matching the production schedule's 75% ordinary prefix plus a 50/50 ordinary/SAM tail.
5. Remaining notes were cosmetic or weak-check observations and did not identify behavior defects.

## Reviewer correction

The review's final note said `peak_vram_mb` divides by `1e6`; current and parent code actually divide by `1024 * 1024`, so both the run and preflight report MiB consistently. This does not affect the review verdict.
