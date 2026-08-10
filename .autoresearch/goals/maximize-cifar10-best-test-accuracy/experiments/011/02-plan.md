# Plan EXP-011: Increase CutMix Probability to 0.75
- **Created**: 2026-08-06

## Milestones

### Milestone 1: Implement and verify the isolated scalar
- [x] Change only `CUTMIX_PROBABILITY` from 0.5 to 0.75 in `train.py`.
- [x] Pass compilation, Ruff, formatting, pre-commit, exact one-line diff, and constant-value checks.
- [x] Time 1,000 real N1/M7 strong-loader batches in a fresh process; require >=120 batches/s, 70-80% probability targets, valid shapes/row sums, and eight clean worker exits.

### Milestone 2: Execute one fixed-protocol run
- [x] Confirm one idle 97,871 MiB H20 and no stale log; run once under 600 seconds with all output redirected to `run.log`.
- [x] Monitor errors and bounded checkpoints without rerunning or changing probability.

### Milestone 3: Verify metric and mechanism
- [x] Verify exit 0, complete summary, 300-second counted budget, sub-600 total, one switch, eight workers stopped, exact target provenance, unique evaluator epochs, and 1,073,962 parameters.
- [x] Require `best_test_acc >=94.25%` for improvement and preserve a valid lower result without retry.
- [x] Compare the 80% checkpoint with EXP-010's 89.73% and the 87.08% underfit marker, then compare first weak, final NLL, and step retention.

## Code Changes
- **`train.py`**: Change the single probability literal from `0.5` to `0.75`. All CutMix implementation, alpha, RNG isolation, loader lifecycle, provenance, architecture, optimization, schedule, timing, seed, and evaluation code remain byte-identical.

## Configuration Changes
- `CUTMIX_PROBABILITY`: `0.5 -> 0.75` (strengthen the mechanism that improved EXP-010 while retaining its hard-tail recovery stage).
- Expected mixed share: 75% of strong batches, approximately 60% of total optimizer steps.
- Unchanged: alpha 1.0, width 2, N1/M7 through 80%, hard weak tail, all-parameter decay `1e-4`, LR/momentum, batch 128, seed 42, and evaluator cadence.

## Adversarial Review Response
- Mandatory Claude plan review completed with exit code 0; no fallback reviewer was used.
- Concern 1 is constrained by the user-defined goal: the necessary condition is +0.10 and seed rerolls are prohibited. Adding post-hoc trajectory gates would silently change that contract. A bare 94.25-94.35 pass will be merged as protocol improvement but explicitly reported as weak causal evidence and baseline-inflation risk.
- Accepted concern 2: measure p=0.75 real-loader throughput during the already-planned 1,000-batch preflight and retain the established >=120 batches/s gate.
- Accepted concern 3: treat counted time as satisfying the fixed budget when it reaches at least 300.0 seconds within one synchronized-step overshoot, rather than requiring a brittle literal; provenance bands are sanity checks for the declared stochastic method, not exact equalities.

## Execution Environment
- Method: local `timeout 600s uv run train.py > run.log 2>&1`.
- Resources: exactly one idle NVIDIA H20, approximately 97,871 MiB; existing eight workers.
- Estimated runtime: approximately 330-345 seconds total; 300 counted seconds; around 26.9k steps based on identical hard/soft cost.
- Log output: full stdout/stderr only to project-root `run.log`; targeted bounded monitoring, never `tee`.
- Tool skill: none.

## Abort Criteria
- Before launch, stop for any diff beyond the literal, malformed probability target, realized 1,000-batch fraction outside 70-80%, throughput below 120 batches/s, or failed worker shutdown.
- During execution, stop for wrong/busy GPU, traceback, OOM, non-finite loss, target assertion, worker-lifecycle failure, or total runtime beyond 600 seconds.
- Do not abort merely for a switch checkpoint below 87.08%: it diagnoses excessive regularization after 80% of the budget, while the fixed hard tail must still test recovery.
- One valid fixed-seed run only; no seed reroll, probability/alpha tuning, or lifecycle adjustment after observing accuracy.

## Verification Protocol

### Verification Procedure
1. Query the index and require baseline 94.15 at `7c1e7d8`, resolving the success threshold as 94.25%.
2. Run compile, Ruff, format, pre-commit, and diff checks. Require only `train.py` and exactly `CUTMIX_PROBABILITY = 0.5 -> 0.75`.
3. In a fresh guarded process, time 1,000 actual strong-loader batches after iterator startup. Require >=120 batches/s, mixed float targets `[128,10]` with finite nonnegative values and row sums one, hard integer targets `[128]`, 700-800 mixed batches, and all eight workers stopped. This measures the only changed host cost; EXP-010 already established unchanged hard/soft GPU parity and lifecycle feasibility.
4. Confirm exactly one idle H20 and no `run.log` variant, then execute once with `timeout 600s uv run train.py > run.log 2>&1`. Require exit 0.
5. Parse ten finite summary fields; require counted time to reach 300.0 seconds with at most one normal synchronized-step overshoot, total <600, 1,073,962 parameters, one `randaugment+cutmix->base` switch near 80.0%, eight stopped workers, and final strong provenance in the broad 70-80% sanity band.
6. Require one evaluation per unique epoch. Parse `best_test_acc`; >=94.25% is improvement, otherwise valid no-improvement with no rerun. Treat a 94.25-94.35 bare pass as protocol success but weak causal evidence.
7. Compare actual steps with EXP-010's 26,898 and require no unexplained material loss; compare switch accuracy with 89.73% and 87.08%, first weak with 93.16%, final NLL with 0.1934, and endpoint slope/best-final gap. Diagnostics cannot override the primary gate.

### Informational Metrics (Optional)
- Final summary and trajectory values from targeted `run.log` lines.
- Realized mixed count/fraction from the switch log.
- Preflight 1,000-batch target-format counts and worker exit count from `03-execute.md`.
