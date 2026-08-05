# EXP-025: Diagnostic-Free Full Two-Gate SE Closure

## Execution

Overall Status & Info:
- **Created**: 2026-07-26
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-test-accuracy-025
- **Commit**: (pending - committed on loop success)
- **PR**: N/A - offline/local-only session
- **Outcome**: failed

## Implementation Notes

### Summary

Added the exact opt-in EXP-017 ratio-16 SE path to both stage-3 residual blocks after accepted model initialization. The production implementation contains only the two projections and residual scaling; all EXP-017 diagnostic buffers, reductions, methods, and reporting are absent. An ignored preflight exercises the actual constructor path for semantic identity, gradient opening, optimizer grouping, and matched throughput.

### Surprises & Discoveries

The accepted `WideResNet` constructor was already the natural verification seam: an opt-in `stage3_attention` argument can attach gates after whole-model initialization while `main()` selects the candidate. No separate model builder or duplicated scored construction is needed.

### Decisions

The timing harness maintains a private CUDA RNG state for each path and excludes RNG state transfer from the timed region. The scored run is additionally required to realize at least 137 passes, addressing the plan review's concern that a short timing projection alone may overstate full-run exposure.

## Experimental Adjustments

- None. Seed, gate ratio, placement, initialization, training recipe, and thresholds match the preregistered plan.

## Run Log

### Run 1 - Preflight only

Metadata:
- **Job ID**: N/A - scored run not launched
- **Log file(s)**: N/A - preflight command output recorded inline below; no `run.log` was created
- **WandB**: N/A
- **Status**: failed preflight
- **Started**: 2026-07-26 20:15 UTC
- **Ended**: 2026-07-26 20:16 UTC

Description:
- One offline local H20 run of the accepted WRN recipe with two diagnostic-free full conditional SE gates. The run tests whether restoring exposure relative to EXP-017 converts its 94.16% near miss to the required 94.17%, without changing seed or training semantics. It will launch only after exact semantic and >=137 projected-pass preflight gates pass.

Observations:
- Semantic checks passed on one NVIDIA H20: two gates, 696,042 parameters, seed 17017, accepted common state/RNG/logit identity, exact-neutral scale, residual-only placement, optimizer grouping, and two-step opening all passed (source: `uv run python .../experiments/025/preflight.py --semantics` output, 2026-07-26).
- Matched timing was stable but missed the binding exposure floor: mixup accepted/candidate medians were 13.234459/13.659803 ms and hard-label medians were 12.833601/13.409848 ms. Weighted retention was 0.964769, projecting 136.900785 passes versus the required 137.0 (source: `uv run python .../experiments/025/preflight.py --throughput` output, 2026-07-26).
- The scored command was not launched, the threshold was not lowered, and no `run.log` was created.

Key Metrics:
- semantic preflight: PASS; parameters: 696,042 (source: preflight output, 2026-07-26)
- mixup timing CV: accepted 0.001307, candidate 0.001608; hard timing CV: accepted 0.000466, candidate 0.004804 (source: preflight output, 2026-07-26)
- weighted accepted/candidate step time: 13.094159/13.572319 ms; retention: 0.964769; projected passes: 136.900785 (source: preflight output, 2026-07-26)

## Verification Results

### Conditions Checked

- Baseline: PASS - 94.07% at `eb08811`; required score threshold 94.17% (source: `exp-index.sh baseline`, 2026-07-26).
- Scope/device/compile: PASS - one NVIDIA H20, `git diff --check` clean, and both production/preflight modules compiled (source: preflight command output, 2026-07-26).
- Semantic identity: PASS - all exact state, RNG, initialization, placement, gradient, optimizer, and diagnostic-absence checks passed (source: semantic preflight output, 2026-07-26).
- Projected exposure: FAIL - 136.900785 passes is below the preregistered 137.0 minimum (source: throughput preflight output, 2026-07-26).
- Scored run and metric condition: skipped - aborted immediately after the failed projected-exposure necessary condition.

### Informational Metrics

- No scored informational metrics; the production run was not launched.

## Errors & Dead Ends

### 2026-07-26 - Diagnostic-free full SE missed exposure gate
- Error: `AssertionError: 136.9007850514866` after `projected_passes=136.900785 < 137.0`
- Root cause: Even without EXP-017's observation reductions, two dense conditional gates retained only 96.4769% of accepted throughput, just below the preregistered exposure floor.
- Source: EXP-025 throughput preflight output; values preserved in Run 1 above.
- Do NOT retry: Do not lower the 137-pass gate, rerun timing for a favorable sample, change seed/ratio/placement, or launch this scored treatment.

## Human Notes

> Autopilot run; no intervention requested.
