# EXP-030: Early Drop-Path on the Added Stage-3 Block

## Execution

Overall Status & Info:
- **Created**: 2026-07-26
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-test-accuracy-030
- **Commit**: (pending - committed on loop success)
- **PR**: N/A - local-only run
- **Outcome**: failed - valid no-improvement

## Implementation Notes

### Summary

Added inverted per-example whole-residual drop-path to `PreActBlock`, configured only `layer3[2]` at fixed p=0.05 with a private CUDA generator seeded 28028, and routed its one-way cutoff through `maybe_disable_drop_path` immediately before the first hard-label forward. All other blocks default to the exact identity path, and the targeted block returns to that identity path at the accepted 65% mixup boundary.

### Surprises & Discoveries

The original draft duplicated the 65% cutoff in a new constant and left the transition inline, which would not have given the preflight an executable oracle for production behavior. Plan review caught this before implementation; the production controller now consumes the already-computed `use_mixup` predicate directly.

### Decisions

The private generator is a plain, non-registered block attribute created only after model device transfer. This preserves accepted initialization and `state_dict`, avoids global CPU/CUDA RNG changes, and lets p=0/eval bypass mask allocation and private-state advancement entirely.

## Experimental Adjustments

- **Separated primary, exposure, and corroboration verdicts**: Plan review established that final accuracy cannot override the goal's `best_test_acc`, and an otherwise valid `<130`-pass run still counts and cannot be rerun. (ref: `02-plan-review.md` concerns 1-2)
- **Used reciprocal-time exposure weighting**: The throughput gate models fixed time as `0.65/ms_early + 0.35/ms_hard`, matching the two time windows exactly. (ref: `02-plan-review.md` concern 4)

## Run Log

### Run 1

Metadata:
- **Job ID**: local timeout PID 1345164 (exec session 83358)
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.3-gpt-5-6-sol/run.log`
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-07-26 22:49:26 UTC
- **Ended**: 2026-07-26 22:55:08 UTC

Description:
- This will be the sole fixed-seed scored run of targeted early drop-path on the accepted EXP-027 learner. It tests whether sparsely suppressing only the added `layer3[2]` residual branch during the mixup/RandAugment window improves robustness while preserving exact accepted hard-tail and evaluation computation. The primary threshold is 94.42% best test accuracy; the preflight must first establish semantic isolation and at least 130 projected passes.

Observations:
- Startup confirmed one CUDA device, accepted `[2,2,3]` topology, 987,098 parameters, 300-second budget, and 195 batches per epoch. (source: `run.log` startup lines)
- Semantic preflight passed with 4.70% observed mask rate and exact p=0/eval/RNG/controller/worker-tail checks. Throughput preflight retained 0.998064 and projected 132.7499 passes with all CVs <=0.00794. (source: local preflight output before Run 1)
- The run completed normally with no traceback, OOM, worker error, or non-finite loss. Mixup and drop-path disabled together at epoch 86, step 16,586, 195.0 seconds; RandAugment disabled after iterator exhaustion at step 16,770, a 184-step lag. (source: `run.log` L40-L44)
- Evaluation cadence was unique at every fifth full epoch through 130 plus the final partial epoch 133. The treatment missed the primary metric and endpoint corroboration under normal exposure, so the exact family is closed without rerun or adjacent tuning. (source: `run.log` L6-L64)

Key Metrics:
- best_test_acc: 93.91%, delta -0.41 points from accepted and -0.51 from threshold (source: `run.log` L66)
- final_test_acc: 93.86%, delta -0.36 points from accepted final 94.22% (source: `run.log` L67)
- final_test_loss: 0.2887, delta +0.0364 from accepted 0.2523 (source: `run.log` L68)
- exposure: 25,922 steps = 132.72064 passes over 133 epochs (source: `run.log` L73-L74)
- timing: 300.0 counted seconds, 341.7 total seconds, 1.2 startup seconds (source: `run.log` L69-L71)
- resources: 1096.3 MiB peak VRAM and 987,098 parameters (source: `run.log` L72-L75)

## Verification Results

### Conditions Checked

- **Run validity - PASS**: exit 0, one finite summary, 300.0 counted seconds, 341.7 total seconds, no error signatures, 987,098 parameters, and no frozen-file drift. (source: `run.log` L65-L75; final git audit)
- **Exposure - PASS**: 132.72064 realized passes, above the preregistered 130-pass mechanism floor. (source: `run.log` L74)
- **Transition and evaluation cadence - PASS**: one paired mixup/drop-path transition at step 16,586 and one exhausted-iterator RandAugment transition 184 steps later; evaluations at epochs 5..130 by fives plus partial epoch 133. (source: `run.log` L6-L64)
- **Primary objective - FAIL**: 93.91% is below both baseline 94.32% and required threshold 94.42%. (source: `run.log` L66)
- **Endpoint corroboration - FAIL (secondary)**: 93.86% is below preregistered 94.32%. (source: `run.log` L67)

### Informational Metrics

- Best-final gap: 0.05 percentage points; accepted gap was 0.10. (source: `run.log` L66-L67)
- Loss delta: +0.0364 versus accepted 0.2523. (source: `run.log` L68)
- Preflight retention/projected exposure: 0.998064 / 132.7499 passes; realized exposure was 132.72064 passes. (source: local throughput output; `run.log` L74)

## Errors & Dead Ends

### 2026-07-26 - Dynamic-module source inspection failure
- Error: `TypeError: <class 'accepted_train.EarlyRandAugment'> is a built-in class`
- Root cause: `inspect.getsource` cannot locate classes created by executing the accepted `git show` source in an in-memory module.
- Source: semantic preflight attempt 1, `preflight.py:255`
- Do NOT retry: Do not use `inspect.getsource` on the dynamic oracle; compare current source snippets against the captured accepted source string.

### 2026-07-26 - Dynamic oracle cannot cross forkserver boundary
- Error: `PicklingError: Can't pickle <class 'accepted_train.EarlyRandAugment'>: No module named 'accepted_train'`
- Root cause: The exact in-memory accepted module is intentionally not installed/importable in spawned forkserver workers.
- Source: semantic preflight attempt 2, `preflight.py:215`
- Do NOT retry: Do not send dynamic-oracle classes to workers; use the established augmented-versus-clean replay check after exact source equivalence has passed.

## Human Notes

> Autopilot local-only execution; no user intervention requested.
