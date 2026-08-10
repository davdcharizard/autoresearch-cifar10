# EXP-016: 106-State Trailing Uniform Clean-Tail SWA

## Execution

Overall Status & Info:
- **Created**: 2026-08-06
- **Autonomy**: autopilot
- **Experiment Branch**: tree-autoresearch/maximize-cifar10-best-test-accuracy-exp-016
- **Base Node**: 004
- **Commit**: e080f7f
- **Outcome**: failed

## Implementation Notes

### Summary

Implemented a `TrailingUniformSWA` state manager entirely in `train.py`. It allocates a 106-slot GPU ring for every parameter and persistent floating buffer before charged timing, samples post-optimizer/post-SAM states every 31 clean-tail steps, and directly reduces the full ring only after it fills. Evaluation remains live until sample 106 and then uses exactly one full-window averaged source with exception-safe full-state restoration. Final diagnostics make cadence, kernel age/ESS, source-at-best, state coverage, parity, distances, BN state, integer buffers, RNG, and restoration reconstructable.

### Surprises & Discoveries

The plan critic identified that evaluating cumulative warm-up averages would allow the rejected full-quarter estimator to produce the max metric. The implementation therefore suppresses averaged evaluation until the exact 106th sample. The first transient CPU smoke also showed that a `/tmp` script launched through `uv` does not automatically place the repository on `sys.path`; adding the current working directory fixed the harness without changing production code.

### Decisions

Used one leading-dimension ring allocation per floating source tensor rather than 106 independent state copies, limiting allocation count while preserving direct slot access. Removed the incremental running sum and materialize the mean via direct reduction across all 106 slots after readiness, eliminating subtract/add cancellation drift. Ring allocation happens before `t_start_training`; copies and reductions happen inside the charged step. Positive BN variance is treated as an integrity invariant while BN ratios/distances remain diagnostics.

## Experimental Adjustments

- **Suppress averaged evaluation until the 106th sample**: prevents a growing cumulative mean from entering `best_test_acc`; adopted from Claude Opus plan review before implementation. (ref: `02-plan-review.md` concern 1)
- **Use direct full-ring reduction and tighter dose gates**: removes running-sum drift and limits throughput confounding. (ref: `02-plan-review.md` concerns 2 and 10)

## Run Log

### Run 1

Metadata:
- **Job ID**: N/A (local preflight)
- **Log file(s)**: `/tmp/exp016_preflight.log`; no metric `run.log` was launched
- **WandB**: N/A
- **Status**: failed
- **Started**: 2026-08-06 12:03 UTC
- **Ended**: 2026-08-06 12:04 UTC

Description:
- The execution first runs deterministic CPU and accuracy-blind paired GPU checks. If the decisive numeric GPU gates pass, it launches exactly one seed-42, 300-charged-second CIFAR-10 metric run on physical GPU 0. The expected result is a fully dosed 106-state boxcar with at least 25,400 optimizer steps and `best_test_acc >=95.50%` for local improvement.

Observations:
- Static checks passed: `train.py` compiles, `git diff --check` is clean, and the only tracked change is `train.py`.
- CPU smoke passed with `window=106 evictions=106 wraps=1 restore_checks=1 rng_failures=0` after the transient import-path repair. (source: `/tmp/exp016_swa_smoke.py` stdout)
- The first complete numeric GPU preflight used physical GPU 0 (`NVIDIA H20`, UUID `GPU-b1bc897d-2183-dad2-8302-8800bc02a633`), guarded the evaluator, read zero test batches, preserved exact online parent/candidate state, and exercised two production-frequency full-window updates per 248-step timed trace. (source: `/tmp/exp016_preflight.log` JSON)
- The preflight was decisively `FAIL` because paired-ratio MAD/median was `0.0053073426`, above the preregistered `0.005` ceiling. No accuracy-bearing run was launched and no gate was changed. (source: `/tmp/exp016_preflight.log` JSON)

Key Metrics:
- Parent round seconds: `[3.147468, 3.138780, 3.165140, 3.140729, 3.163192]`; parent drift `0.00837499` passed the `0.03` gate. (source: `/tmp/exp016_preflight.log` JSON)
- Candidate round seconds: `[3.145776, 3.155481, 3.146648, 3.145871, 3.143007]`. (source: `/tmp/exp016_preflight.log` JSON)
- Paired ratios: `[0.99946230, 1.00532079, 0.99415781, 1.00163726, 0.99361867]`; median `0.99946230` passed, maximum `1.00532079` passed, MAD/median `0.00530734` failed. (source: `/tmp/exp016_preflight.log` JSON)
- Projected optimizer steps: `25,573.751`; projected total runtime `457.488s`; candidate peak allocation `1,738.40 MiB`. (source: `/tmp/exp016_preflight.log` JSON)
- Primary metric: not measured; `best_test_acc` unavailable because preflight failed before the sole metric launch.

## Verification Results

### Conditions Checked

- Execution failed at the preregistered pre-metric dispersion gate; metric verification was skipped as required by the execution workflow.

### Informational Metrics

- Accuracy and full-run informational metrics were not collected because no metric process was launched.

## Errors & Dead Ends

### 2026-08-06 — Decisive paired preflight dispersion exceeded ceiling
- Error: `ratio_mad_over_median=0.0053073426 > 0.005`
- Root cause: five alternating-order paired ratios had slightly more dispersion than the fixed execution-soundness ceiling even though median overhead was `0.999462x` and all individual ratios were at most `1.005321x`.
- Source: `/tmp/exp016_preflight.log` complete numeric JSON.
- Do NOT retry: do not rerun this numeric gate, relax the dispersion threshold, or launch the EXP016 metric run; record a failed pre-metric leaf.

### 2026-08-06 — Preflight used exact nvidia-smi memory label for PyTorch-visible memory
- Error: `assert round(properties.total_memory / 1024 / 1024) == 97871`
- Root cause: `nvidia-smi` reports 97,871 MiB physical memory while PyTorch exposes 97,508.75 MiB usable memory; the plan requires an approximate H20 identity check.
- Source: `/tmp/exp016_preflight.log` before any numeric timing gate.
- Do NOT retry: do not require byte-exact equality between physical and framework-visible GPU capacity; use the preregistered approximate tolerance while matching name and UUID exactly.

### 2026-08-06 — Transient smoke could not import repository prepare module
- Error: `ModuleNotFoundError: No module named 'prepare'`
- Root cause: `uv run python /tmp/exp016_swa_smoke.py` placed `/tmp`, not the repository root, on the script import path.
- Source: first CPU smoke stderr before any production or metric run.
- Do NOT retry: do not run repository-importing `/tmp` harnesses without explicitly prepending the current working directory to `sys.path`.

## Human Notes

> Autopilot execution; no human intervention during this phase.
