# EXP-046: CIFAR-Mean Crop Fill

## Execution

Overall Status & Info:
- **Created**: 2026-07-27
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-test-accuracy-046
- **Commit**: (pending - committed on loop success)
- **PR**: N/A (offline local run)
- **Outcome**: failed - pre-score loader stability gate

## Implementation Notes

### Summary

Changed exactly the accepted `RandomCrop` constructor to use constant RGB fill `(125,123,114)`. Crop geometry, padding mode, transform order, worker-safe RandAugment, normalization, model, training loop, and all state remain unchanged. Added an ignored evaluator-blocked preflight with independent NumPy pixel bytes, transform/RNG/state controls, same-pool active-to-inactive worker replay, production-payload loader timing, and counterbalanced complete-step H20 timing.

### Surprises & Discoveries

The installed torchvision constant PIL path uses `ImageOps.expand` for both scalar and tuple fill and samples crop coordinates only afterward, directly supporting the preregistered RNG claim. Plan review clarified that trace payloads themselves distort loader performance, so semantic tracing and production-payload timing are separate modes.

### Decisions

The pixel oracle preallocates a NumPy RGB array rather than reusing PIL/torchvision padding. Loader timing records both delay-free service and production-like 11ms overlap windows; worker transition tracing never contributes to timing gates. A structurally valid low-exposure score remains a final single result rather than grounds for rerun.

## Experimental Adjustments

- None.

## Run Log

### Run 1

Metadata:
- **Job ID**: N/A - score not launched
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.3-gpt-5-6-sol/run.log`
- **WandB**: N/A
- **Status**: failed preflight
- **Started**: 2026-07-27 (local qualification)
- **Ended**: 2026-07-27 (before score)

Description:
- One fixed-seed local H20 score of quantized CIFAR-mean constant crop padding will run only after pixel/RNG/worker and timing gates pass. It tests whether removing the frequent high-magnitude black border improves the accepted invariance learner while preserving at least127 passes. The valid score will not be rerun regardless of outcome.

Observations:
- Production diff is exactly one `RandomCrop` fill argument; syntax and diff checks pass before semantic qualification (source: local static audit, 2026-07-27).
- Semantic preflight passed directly: all 162 forced pixel cases matched the independent NumPy oracle; 160 touching cases changed only the mask; candidate normalized fill was `[-0.0012039,0.00015295,0.00055882]`; sampled contact/synthetic shares were `0.987480/0.133672`; 64 calls covered21 RandAugment decisions with exact states; and complete active/inactive worker epochs each traced49,920 exact nonpixel samples (source: semantic preflight stdout, 2026-07-27).
- Model/state qualification remained exact at1,003,482 parameters,52 parameter tensors, and97 state entries (source: semantic preflight stdout, 2026-07-27).
- Production-payload overlap windows were stable (`CV 0.17-2.11%`), candidate/accepted weighted overlap was `2.70705/2.70499s`, and projected wall was only`344.169s`. Delay-free service failed the preregistered stability gate: accepted-active CV`13.55%`, candidate-active CV`8.21%`, and candidate-inactive CV`5.76%`; candidate inactive maximum`1.18743s` also exceeded1.10x the accepted`1.04878s` median. Qualification stopped before H20 timing or scoring (source: timing preflight stdout, 2026-07-27).

Key Metrics:
- `best_test_acc`: unavailable - sole score not launched.
- Semantic treatment incidence: contact `98.7480%`, synthetic area `13.3672%` (source: semantic preflight stdout).
- Loader weighted overlap accepted/candidate: `2.70499/2.70705s`; projected wall `344.169s` (source: timing preflight stdout).

## Verification Results

### Conditions Checked

- **Semantic/source/state contract - PASS**: exact one-line source scope, independent pixel/RNG/state/worker controls all passed.
- **Loader feasibility - FAIL**: three delay-free service CVs exceeded5%, and one candidate inactive service epoch exceeded the1.10x maximum gate.
- **H20 exposure timing - skipped**: aborted after loader gate failure.
- **Completion and primary metric - skipped**: scored run was not launched.

### Informational Metrics

- Not collected because a necessary pre-score feasibility condition failed.

## Errors & Dead Ends

### 2026-07-27 - Delay-free loader service instability
- Error: accepted-active/candidate-active/candidate-inactive service CVs were `13.55%/8.21%/5.76%`, above the preregistered5% ceiling; candidate inactive max also exceeded1.10x accepted median.
- Root cause: Delay-free persistent-worker service was host-variable across the two preregistered streams even though production-like11ms overlap remained stable. The experiment contract required both modes.
- Source: timing preflight stdout before H20 timing or any score.
- Do NOT retry: Do not rerun, relax the service gate, drop the accepted variability control, or rescue with another fill/mode/worker timing protocol after observing these windows.

## Human Notes

> Autopilot requested; user asleep. Offline/local only, no GitHub, remote, or network.
