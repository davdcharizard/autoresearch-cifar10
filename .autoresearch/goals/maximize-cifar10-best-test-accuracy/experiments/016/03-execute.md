# EXP-016: BF16-Funded Width-3 Postactivation ResNet-20

## Execution

Overall Status & Info:
- **Created**: 2026-08-06
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-best-test-accuracy-016
- **Commit**: (pending - committed on loop success)
- **Outcome**: failed

## Implementation Notes

### Summary

The production diff changes only `train.py`: width 2 becomes width 3, CUDA BF16 capability is fail-closed and reported with unchanged TF32 provenance, and only training forward plus cross-entropy run inside BF16 autocast. Model/master parameters, backward, SGD state, BatchNorm persistent state, evaluation, data, schedule, timer, seed, and worker lifecycle remain accepted. Ignored controllers implement production-distribution numerical, three-arm timing, loader/lifecycle, and wall gates.

### Surprises & Discoveries

The environment's actual defaults are cuDNN TF32 enabled, matmul TF32 disabled, `highest` matmul precision, and cuDNN benchmark disabled. External Claude's first implementation review found that mere cross-arm backend equality could still falsely authorize BF16 if TF32 were consistently off; the final timing controller now asserts the exact reviewed defaults. Claude also identified an undocumented BatchNorm dtype assumption and inverted wall-exposure bound, both corrected before any controller ran.

### Decisions

All controller sources and the production diff received mandatory external Claude review with no fallback. After the first review, the 200-step probe was changed to distinct production batches with checkpointed loose drift alignment, BatchNorm output dtype became observational, CUDA stage timing was made production-faithful, and wall projection became conservative. A focused re-review and final narrow review both returned `APPROVED`; the exact approved sources are recorded in `02-plan-review-implementation-addendum.md`.

## Experimental Adjustments

- **Externally review disposable gates before use**: Closed the only remaining reward-hacking surface identified in the plan review. (ref: `02-plan-review.md`, `02-plan-review-implementation-addendum.md`)
- **Assert actual TF32/autotune/layout state**: Prevents a consistently misconfigured FP32 control from granting a false BF16 funding pass. (ref: implementation addendum concern 1)
- **Use conservative wall and drift vetoes**: Maximum one eval per projected epoch and clean weak held-out batches avoid optimistic wall counting and overly strict cross-trajectory selection. (ref: final Claude `APPROVED` review)

## Run Log

### Run 1

Metadata:
- **Job ID**: N/A - production was blocked by preflight
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.5-gpt-5-6-sol-adversarial/run.log`
- **WandB**: N/A
- **Status**: failed preflight; production not launched
- **Started**: 2026-08-06 08:23 UTC
- **Ended**: 2026-08-06 08:26 UTC

Description:
- One seed-42, 300-counted-second local run of the reviewed width-3 BF16 candidate on the sole H20, permitted only after every preflight gate passes. The hypothesis is that BF16 funds at least 22,863 width-3 updates and the added capacity reaches at least 94.25% best test accuracy. There is no alternate width, precision, scaler, TF32 setting, or valid-run retry.

Observations:
- Attempt 1 stopped before importing the candidate because the ignored controller lacked the project root on `sys.path`; Claude approved the explicit root bootstrap before the permitted code-error retry.
- Attempt 2 reached the paired 200-distinct-production-batch trajectory and failed the registered candidate-only class-concentration veto: the BF16 arm exceeded 95% predictions in one class while the same-step FP32 control did not. The assertion preceded result serialization, so no step-specific JSON exists; the traceback identifies `numerical_gate` concentration line 268 in that exact executed source.
- The failed process was explicitly terminated after its assertion left DataLoader workers holding the interpreter open. Follow-up checks found no GPU compute process and no preflight/timing process. Timing, loader, evaluator-wall, and production gates were not run.

Key Metrics:
- Numerical collapse veto: **failed** - candidate concentration `>95%` with paired control `<=95%` during the 200-batch production-distribution trajectory.
- Parameter count/static state: **passed** - 2,412,730 FP32 parameters; only `train.py` tracked.
- External reviews: **passed** - idea, plan, controller, corrections, and import fix all reviewed by Claude with no fallback.
- Production accuracy: **not measured** - no `run.log` was created and the seed-42 production run was not launched.

## Verification Results

### Conditions Checked

- Baseline/scope/GPU: **passed** - baseline 94.15 at `7c1e7d8`; one idle 97,871 MiB H20; only `train.py` tracked and `data/` preserved.
- Static/structural: **passed** - syntax/diff checks, BF16 capability, width-3 parameter count, FP32 persistent state, and reviewed autocast scope.
- External controller review: **passed** - final exact controller and bootstrap fix approved by Claude.
- Paired numerical gate: **failed** - candidate-only `>95%` class concentration versus control `<=95%` during the real-batch trajectory.
- Three-arm timing/funding: **skipped - aborted after numerical failure**.
- Loader/lifecycle/wall gate: **skipped - aborted after numerical failure**.
- Production completion/metric: **skipped - production was not authorized**.

### Informational Metrics

- No production informational metrics; the formal accuracy run was not launched.

## Errors & Dead Ends

### 2026-08-06 - Controller could not import project module
- Error: `ModuleNotFoundError: No module named 'train'`
- Root cause: launching an ignored controller by path put `experiments/016/` on `sys.path`, not the project root; the candidate and all numerical gates were never reached.
- Source: numerical preflight attempt 1 traceback before module import, 2026-08-06 08:23 UTC.
- Do NOT retry: do not rely on the caller's cwd being added to `sys.path`; both controllers must explicitly prepend the resolved project root before importing `train`.

### 2026-08-06 - BF16 trajectory triggered class-concentration veto
- Error: `AssertionError: candidate_concentration > 0.95 and control_concentration <= 0.95`
- Root cause: under 200 distinct seed-fixed N1/M7+CutMix batches at LR 0.1, the width-3 BF16 arm entered candidate-only near-single-class prediction concentration while the paired FP32 control did not.
- Source: numerical preflight attempt 2 traceback at executed `preflight_bf16.py` line 268, 2026-08-06 08:26 UTC.
- Do NOT retry: do not rerun this exact BF16-width3 operating point, add scaling/clipping/warmup, narrow autocast, or substitute a width/precision variant inside EXP016.

## Human Notes

> The researcher requires external Claude adversarial review for both ideas and plans, with no fallback reviewer. Claude authentication was restored and all EXP-016 reviews completed successfully.
