# EXP-031: Scale-Controlled Max-Residual Global Pooling

## Execution

Overall Status & Info:
- **Created**: 2026-08-06
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-best-test-accuracy-031
- **Commit**: (pending — committed on loop success)
- **Outcome**: failed — exact-corpus safety veto before timing/production

## Implementation Notes

### Summary

Calibrated the bounded pooling coefficient on the preregistered first 1,024 CIFAR-10 training examples from the exact seed-42 CPU initialization, then implemented the frozen descriptor `avg + 0.10*s*(max-avg)` with `s=0.68908708`. Added sparse counted training-ratio safety checks and eval-forward piggyback telemetry without changing evaluator calls or logits. Initial compile, Ruff, format, pre-commit, whitespace, and tracked scope checks passed.

### Surprises & Discoveries

The first ignored calibration invocation reproduced the known path-launched-controller `ModuleNotFoundError: prepare`; adding the documented project-root `sys.path` bootstrap fixed only the controller. The identical calibration then passed twice with source hash `29c15232...`, state hash `ef6b4f64...`, input hash `9e33bca1...`, `rms_avg=4.7559553078`, and `rms_residual=6.9018204627`.

### Decisions

Following plan review, absolute production step floors were removed to avoid node-speed reroll bias. The fresh paired candidate/control ratio is the sole exposure gate. Initialization scale is not treated as a lifetime guarantee: fixed training-only 1,000-step monitors hard-veto an effective perturbation ratio above 0.25, while test-evaluation ratios remain non-blocking telemetry and can never tune the coefficient.

## Experimental Adjustments

- **Controller path bootstrap**: Prepended the project root to `sys.path` after the first calibration-only import failure; calibration rule, source, corpus, state, and candidate code remained unchanged. (ref: `calibration.log` and implementation notes)

## Run Log

### Run 1

Metadata:
- **Job ID**: N/A — production not authorized
- **Log file(s)**: `experiments/031/calibration.log`, `experiments/031/preflight.log`, `experiments/031/preflight-report.json`
- **WandB**: N/A
- **Status**: failed preflight; timing and production skipped
- **Started**: 2026-08-06
- **Ended**: 2026-08-06

Description:
- Conditional single seed-42 H20 run of bounded max-residual global pooling on the accepted recipe. Production is forbidden until semantic, exact-corpus, perturbation-drift, and five-pair timing gates pass. Formal success requires at least 94.25% without coefficient adjustment or reroll.

Observations:
- Calibration passed reproducibly with frozen scale 0.68908708 and effective coefficient 0.068908708; initialization added-residual/average RMS ratio was 0.1000000000. (source: `experiments/031/calibration.log`)
- Exact-corpus replay used the registered 200 strong and 64 weak batches with matching hashes. The candidate produced candidate-only >95% class concentration at steps 3, 18, and 19. (source: `experiments/031/preflight-report.json`)
- Dynamic scale and optimization gates failed: aggregate added-residual/average RMS reached 0.409621, maximum per-example ratio 4.341249, update p95/max 1.338413/1.583063, and strong/weak terminal loss-EMA ratios 1.130417/1.304067. Classifier-gradient p95 was 1.126985 and did pass, but cannot waive the other vetoes. (source: `experiments/031/preflight-report.json`)
- Per the no-rescue rule, coefficient 0.68908708 and all gates were left unchanged. Paired timing and the scored seed-42 run were not executed.

Key Metrics:
- Primary metric: NaN; no production accuracy run was authorized.
- Safety: candidate-only concentration steps `[3,18,19]`; update p95/max `1.338413/1.583063`; strong/weak loss-EMA ratios `1.130417/1.304067`; max aggregate/per-example perturbation `0.409621/4.341249`.

## Verification Results

### Conditions Checked

- **Calibration/static semantics — pass:** reproducible frozen scale, exact corpus/source/state hashes, syntax/quality/scope, formula, parameters, and finite execution passed.
- **Exact-corpus dynamic safety — fail:** class concentration, update, loss, aggregate perturbation, and per-example perturbation gates failed.
- **Timing — skipped:** safety veto blocked GPU timing.
- **Production/primary metric — skipped:** no scored run; metric is NaN and verdict must be invalid.

### Informational Metrics

- Frozen scale 0.68908708; coefficient 0.068908708; calibration RMS avg/residual 4.7559553078/6.9018204627.

## Errors & Dead Ends

### 2026-08-06 — Bounded-init max residual became unbounded on the training trajectory
- Error: `logit_cosine=0.979292`, candidate-only concentration at steps 3/18/19, `update_ratio_p95=1.338413`, `max_aggregate_ratio=0.409621`, and `max_per_example_ratio=4.341249` violated preregistered gates.
- Root cause: Initialization aggregate RMS calibration did not control sparse per-example max features or their evolving training-time ratio; the shared classifier still amplified hard-max geometry.
- Source: `experiments/031/preflight-report.json` serialized before assertions and second `preflight_pool.py` run.
- Do NOT retry: Do not lower/tune the coefficient or corpus within EXP031; any revisit needs an intrinsically bounded per-example smooth aggregation, not initialization-only global RMS scaling.

### 2026-08-06 — Disposable pooling oracle used an invalid FP64 distributivity tolerance
- Error: `torch.allclose(actual, reference, rtol=1e-12, atol=1e-12)` failed before corpus replay.
- Root cause: Production forms `max-avg` in FP32 before promotion, while the reference reassociated the expression in FP64; FP32 subtraction rounding makes a 1e-12 equality invalid.
- Source: first `preflight_pool.py` invocation, line 109; no corpus update or production run occurred.
- Do NOT retry: Compare the two algebraically equivalent paths at FP32-appropriate `rtol=1e-6, atol=1e-7`; do not alter coefficient, formula, safety gates, or candidate code.

### 2026-08-06 — Calibration controller lacked project-root import path
- Error: `ModuleNotFoundError: No module named 'prepare'`
- Root cause: Path-launched ignored controller did not inherit the project root on `sys.path`.
- Source: first `calibrate_pool.py` invocation before any tracked candidate edit.
- Do NOT retry: Bootstrap `ROOT` into `sys.path` before importing root modules; the corrected identical calibration passed.

## Human Notes

> Autopilot requested; no execution-phase intervention.
