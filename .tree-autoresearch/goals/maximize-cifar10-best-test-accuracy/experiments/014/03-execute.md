# EXP-014: Calibrated stage-3 width-5 expansion

## Execution

Overall Status & Info:
- **Created**: 2026-08-06
- **Autonomy**: autopilot
- **Experiment Branch**: tree-autoresearch/maximize-cifar10-best-test-accuracy-exp-014
- **Base Node**: 011
- **Commit**: `527fd37`
- **Outcome**: failed

## Implementation Notes

### Summary

Changed only the final two 8x8 residual blocks from 256 to 320 channels, including the tail BatchNorm and classifier input, while preserving the six-block topology and every inherited training mechanism. Updated the printed architecture/stage-width metadata and added evaluation charged-time/progress plus terminal debiased training-loss diagnostics outside the charged per-step interval.

### Surprises & Discoveries

The implementation matched the planned shape edit directly: no hidden fixed-width helper or downstream consumer required an additional code path. A seed-reset CPU construction check reproduced every candidate state tensor exactly and the computed parameter count was exactly 3,827,290.

### Decisions

Kept MAC accounting external rather than embedding a self-verifying constant in production. Reused the loop's existing `debiased` scalar for terminal training loss and computed evaluation progress only after the evaluator returned, so neither diagnostic changes charged training work.

## Experimental Adjustments

- **Use direct indexing for transient EMA-state mutation**: The candidate's channels-last first convolution weight is not view-contiguous, so the harness now mutates `[0,0,0,0]` directly. This occurred before any numeric gate vector was emitted and changes no production code. (ref: `/tmp/exp014_gpu_verify.log`)
- **Use an experiment-owned bytecode cache**: The shared `/tmp/__pycache__` is not writable, so transient harness checks set `PYTHONPYCACHEPREFIX=/tmp/exp014_pycache`. This changes no harness or production semantics. (ref: preflight syntax-check stderr)

## Run Log

### Run 1

Metadata:
- **Job ID**: N/A - stopped at decisive preflight
- **Log file(s)**: `/tmp/exp014_cpu_verify.log`, `/tmp/exp014_gpu_verify.log`, `/tmp/exp014_preflight.log`; no `run.log` was created
- **WandB**: N/A
- **Status**: failed at preflight
- **Started**: 2026-08-06 09:47 UTC
- **Ended**: 2026-08-06 09:49 UTC

Description:
- One fixed-seed metric run of the preregistered 64/128/320 candidate will be launched only if all CPU/GPU correctness checks and the first complete paired accuracy-blind timing gate pass. It uses physical GPU 0 with a 600-second outer timeout and preserves raw output in `run.log`. No metric retry, alternate width, or hyperparameter change is permitted.

Observations:
- Implementation checkpoint passed: `py_compile`, `git diff --check`, exact tracked scope, 3,827,290 parameters, deterministic seed-reset state, finite CPU logits, and `(2,10)` output shape.
- The CPU contract audit passed with exact 461,556,864 MACs, 22 expected shape-different state keys, finite nonzero gradients for every parameter on batch 256, deterministic CutMix, and drop-path draw counts 6 active/0 terminal (source: `/tmp/exp014_cpu_verify.log`).
- The candidate-only physical-GPU-0 smoke passed at 652.138 MiB peak allocation. The SAM perturbation norm was 0.05000001 with exact restore and one BatchNorm update; 30 full-state EMA samples split 15/15 ordinary/SAM and swapped/restored exactly (source: `/tmp/exp014_gpu_verify.log`).
- The single complete preflight was stable but decisively rejected: parent drift 0.0077569 and ratio dispersion 0.0016938 passed, while median latency ratio 1.1609750 exceeded the 1.15 gate. No test loader was iterated and no metric run or `run.log` was created (source: `/tmp/exp014_preflight.log` `PREFLIGHT_JSON`).
- Claude Opus independently reproduced the ratios and every gate, found no blocker or integrity issue, and confirmed `crash/NaN` as the required tree encoding (source: `03-result-review.md`).

Key Metrics:
- parent weighted round medians: 12.692932, 12.720745, 12.655057, 12.622581, 12.634162 ms (source: `/tmp/exp014_preflight.log` `PREFLIGHT_JSON`)
- candidate weighted round medians: 14.763384, 14.768467, 14.667320, 14.686745, 14.663686 ms (source: `/tmp/exp014_preflight.log` `PREFLIGHT_JSON`)
- paired round ratios: 1.1631184, 1.1609750, 1.1590085, 1.1635295, 1.1606378; median 1.1609750; max 1.1635295 (source: `/tmp/exp014_preflight.log` `PREFLIGHT_JSON`)
- projection: 22,220.98 steps, 113 complete epochs, 137.815 EMA samples, 520.001 total seconds (source: `/tmp/exp014_preflight.log` `PREFLIGHT_JSON`)
- informational joint-process peak: 748.484 MiB; binding candidate-only peak: 652.138 MiB (source: `/tmp/exp014_preflight.log`; `/tmp/exp014_gpu_verify.log`)
- paired relative SAM: parent 0.0006286761, candidate 0.0005873253 (source: `/tmp/exp014_gpu_verify.log`)
- conditioning losses parent/candidate: step 1 2.467820/2.462616; step 25 1.913384/1.842300; step 50 2.139143/2.134604; step 100 1.489785/1.501453; step 200 1.121022/1.130681 (source: `/tmp/exp014_preflight.log` `PREFLIGHT_JSON`)

## Verification Results

### Conditions Checked

- **Physical GPU / visibility**: PASS - physical GPU 0 and sole visible CUDA device were NVIDIA H20; physical memory reported 97,871 MiB.
- **Source and architecture integrity**: PASS - parent snapshot matched commit `d68f73a`; only `train.py` changed; exact parameters/MACs/shapes/determinism and production mechanisms passed.
- **Candidate-only memory**: PASS - 652.138 MiB `<4096 MiB`.
- **One-shot paired preflight**: FAIL - median ratio 1.1609750 `>1.15`; all other numeric gates passed. This is a valid research rejection, not a retryable harness error.
- **Metric run and primary accuracy**: skipped - aborted immediately after the necessary preflight condition failed; `run.log` remained absent.

### Informational Metrics

- The candidate's step-200 conditioning loss was 0.009659 above parent while logit norm was 0.3341 higher; this finite short trace is diagnostic only and cannot establish a capacity mechanism.
- Candidate SAM perturbation was about 6.58% smaller relative to parameter norm than parent, as expected from the larger model under fixed Euclidean rho.
- The weighted ratio covers charged training paths, not the measured synthetic evaluation forwards. Its parent absolute calibration is about 9% slower than the historical run, so projections are approximate consistency checks rather than independent evidence.

## Errors & Dead Ends

### 2026-08-06 - channels-last diagnostic view was invalid
- Error: `RuntimeError: view size is not compatible with input tensor's size and stride`
- Root cause: The transient EMA audit attempted `.view(-1)` on a channels-last convolution weight.
- Source: execution transcript before the successful retry; the same-path successful log overwrote the raw traceback, while this durable entry was recorded immediately.
- Do NOT retry: Do not flatten channels-last parameters with `.view()`; use direct multidimensional indexing for an in-place diagnostic mutation.

### 2026-08-06 - shared temporary bytecode cache was not writable
- Error: `Permission denied: /tmp/__pycache__/exp014_preflight...pyc`
- Root cause: The shared `/tmp/__pycache__` directory has incompatible ownership/permissions.
- Source: Preflight syntax-check stderr before any GPU measurement.
- Do NOT retry: Set `PYTHONPYCACHEPREFIX=/tmp/exp014_pycache` for transient EXP-014 Python commands.

### 2026-08-06 - fixed width-320 candidate failed the decisive latency gate
- Error: `median candidate/parent ratio 1.1609750 > 1.15`
- Root cause: The 64/128/320 8x8-stage expansion increased production-weighted step latency by about 16.10%, just beyond the preregistered feasibility ceiling.
- Source: `/tmp/exp014_preflight.log` `PREFLIGHT_JSON`; all five paired ratios were 1.15901-1.16353.
- Do NOT retry: Do not rerun timing, substitute width 288, retune training, or launch a metric for EXP-014.

## Human Notes

> Autopilot session; no execution-phase intervention.
