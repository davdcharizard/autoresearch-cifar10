# EXP-020: Isolated PyTorch Nesterov Momentum

## Execution

Overall Status & Info:
- **Created**: 2026-08-06
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-best-test-accuracy-020
- **Commit**: (pending - committed on loop success)
- **Outcome**: failed

## Implementation Notes

### Summary

Created EXP020 from the 94.15 frontier commit and modified only tracked `train.py`. The complete production semantic diff is the literal `nesterov=True` argument on the accepted single PyTorch SGD constructor; model, data, schedule, loss, timer, evaluator, lifecycle, seed, logging, and summary are unchanged.

### Surprises & Discoveries

The initial patch expanded the constructor formatting. It was corrected before any verification so the final word diff contains only `nesterov=True`. External Claude independently confirmed installed PyTorch 2.9.1 first-buffer, 1.9x transient, and shared steady-scale mechanics and conditionally approved the execution plan.

The direct recurrence controller needed deterministic diagnostic mode plus `CUBLAS_WORKSPACE_CONFIG=:4096:8` before two identical CUDA model backwards produced bitwise-equal gradients. Under normal production-like CUDA settings, the paired safety models again had different first raw gradients/buffers despite identical reset state; this is ordinary backward nondeterminism, not an optimizer-state defect. Independently, Nesterov crossed the registered class-concentration veto at strong step 11, which is sufficient to block production.

### Decisions

Production evaluation count is capped at 19 to prevent faster epochs from creating more max-metric opportunities. The first-direction gate reads the explicit pre-storage update tensor rather than rounded parameter differences. Any persisted safety corpus is described as production-distribution evidence, not the scored run's exact batches.

The candidate-only concentration event was not relaxed as "transient" after observation. Although external plan review called this apparatus advisory/over-scoped, the final approved plan retained the threshold, so it remains binding and no timing or production run is authorized.

## Experimental Adjustments

- **Closed plan-review fairness/controller gaps**: Added a 19-evaluation cap, explicit pre-storage ratio measurement, and planning materialization that observed 100/200 CutMix batches with eight workers stopped. (ref: `02-plan-review.md`)
- **Enabled deterministic diagnostic mode for raw-gradient alignment**: The first semantics attempt exposed ordinary CUDA backward non-bitwise behavior; the unchanged recurrence passed with deterministic algorithms and required cuBLAS workspace configuration. (ref: Errors & Dead Ends)
- **Blocked production after the registered concentration veto**: Nesterov reached 124/128 predictions in one class at strong step 11 while control's top class was 83/128. (ref: `/tmp/exp020_safety_result.json`)

## Run Log

### Run 1

Metadata:
- **Job ID**: N/A - production blocked by preflight
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.5-gpt-5-6-sol-adversarial/run.log`
- **WandB**: N/A
- **Status**: failed preflight; production not launched
- **Started**: 2026-08-06
- **Ended**: 2026-08-06

Description:
- Sole seed-42 production run of isolated PyTorch Nesterov on the accepted width-2 RandAugment/CutMix recipe. It is authorized only after recurrence, replayable production-distribution safety, and five-pair timing/exposure gates pass. Expected result is 94.30% best accuracy, with improvement requiring at least 94.25%, 26,629 steps, no more than 19 evaluations, and every integrity condition.

Observations:
- Static implementation passed `py_compile`, Ruff 0.15.6, `git diff --check`, and scope inspection; the word diff adds only `nesterov=True`, with only tracked `train.py` modified and `data/` preserved. (source: preflight command output)
- Installed/manual recurrence passed four FP32 steps with changing gradients, coupled decay, and an LR change: pre-storage first-direction ratio 1.899999990, exact first buffers, hard/soft alignment, 1,073,962 parameters, and optimizer RNG neutrality. (source: `/tmp/exp020_semantics.py`, `SEMANTICS_PASS`)
- Immutable production-distribution corpus passed at SHA-256 `49b367ebf14f4ab9d7dc78e49407e532fe821d127e0d6ecbe15fcab5e5f06647`: 200 strong batches, exactly 100 CutMix, 64 hard weak batches, and both sets of eight workers stopped. (source: `/tmp/exp020_materialize.py`, `CORPUS_PASS`)
- Paired safety stayed finite through 264 steps; representative recurrence errors were zero, first replay-loss ratio 1.104710, maximum update spike 1.736714x, strong final/max loss-EMA ratios 0.960829/0.976076, weak final/max ratios 0.964297/0.983864, and optimizer RNG remained neutral. (source: `/tmp/exp020_safety_result.json`)
- The safety gate failed at strong step 11: candidate histogram `[0,0,124,0,1,3,0,0,0,0]` versus control `[0,0,0,83,0,0,0,0,41,4]`, a candidate-only 96.875% concentration. Timing and production were skipped; no `run.log` was created. (source: `/tmp/exp020_safety_result.json`)

Key Metrics:
- Static/recurrence/corpus gates: **passed**.
- Paired safety: **failed** - candidate-only greater-than-95% concentration at strong step 11.
- Production accuracy: **not measured** - no scored run authorized.

## Verification Results

### Conditions Checked

- Baseline/scope/one-keyword source: **passed** - 94.15 at `7c1e7d8`; only `nesterov=True` added.
- Installed/manual recurrence and RNG: **passed** - `SEMANTICS_PASS`, first ratio 1.899999990.
- Persisted corpus/lifecycle/targets: **passed** - exact digest, 100/200 CutMix, 64 hard weak, workers stopped.
- Paired production-distribution safety: **failed** - candidate-only 96.875% one-class concentration at step 11.
- Timing/exposure: **skipped - aborted after safety failure**.
- Production completion/metric: **skipped - production not launched**.

### Informational Metrics

- No production metrics; `run.log` was never created.

## Errors & Dead Ends

### 2026-08-06 - CUDA gradient comparison was non-bitwise without deterministic diagnostics
- Error: `AssertionError` comparing raw gradients from aligned sequential CUDA backwards.
- Root cause: ordinary CUDA convolution/GEMM backward did not produce bitwise-identical gradients across separate model executions; neither optimizer had stepped.
- Source: `/tmp/exp020_semantics.py` attempt 1.
- Do NOT retry: use deterministic algorithms for bitwise diagnostic comparisons; do not attribute pre-step CUDA rounding differences to Nesterov.

### 2026-08-06 - Deterministic cuBLAS required process-start workspace configuration
- Error: `RuntimeError: Deterministic behavior ... uses CuBLAS ... set CUBLAS_WORKSPACE_CONFIG`.
- Root cause: deterministic algorithms were enabled after process start without the required cuBLAS workspace environment.
- Source: `/tmp/exp020_semantics.py` attempt 2.
- Do NOT retry: launch deterministic CUDA controllers with `CUBLAS_WORKSPACE_CONFIG=:4096:8` already set.

### 2026-08-06 - Nesterov triggered the candidate-only concentration veto
- Error: `AssertionError: candidate_only_concentration_events=1`.
- Root cause: on the immutable seed-42 production-distribution corpus, Nesterov concentrated 124/128 step-11 predictions in class 2 while control remained at 83/128 in its top class.
- Source: `/tmp/exp020_safety_result.json`, SHA-256 `49b367ebf14f4ab9d7dc78e49407e532fe821d127e0d6ecbe15fcab5e5f06647`.
- Do NOT retry: do not rerun, warm up, clip, lower LR, reset momentum, or relax the registered 95% threshold inside EXP020.

## Human Notes

> Autopilot; external Claude idea and plan reviews completed successfully with no fallback.
