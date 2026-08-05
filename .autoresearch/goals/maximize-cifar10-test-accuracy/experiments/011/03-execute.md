# EXP-011: One Extra 8x8 Residual Block

## Execution

Overall Status & Info:
- **Created**: 2026-07-24
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-test-accuracy-011
- **Commit**: (pending - committed on loop success)
- **PR**: N/A - user required fully local/offline execution
- **Outcome**: failed - valid near miss below required improvement margin

## Implementation Notes

### Summary

Starting from accepted commit `eb08811`, the implementation replaces the scalar block count with explicit `STAGE_BLOCKS=(2,2,3)`, strictly validates three positive integer counts, applies them independently to the three WRN stages, and logs exact widths/depths. It adds only one unchanged 128-to-128 `PreActBlock` at 8x8; FP32 numerics, stage widths, initialization, optimizer, schedule, mixup, seed, data, and evaluator behavior remain accepted.

### Surprises & Discoveries

The matched production-path preflight retained 92.77% throughput and projected 131.64 passes, closely matching EXP-010 despite the added sequential residual path. Its first invocation from `/tmp` needed the project root inserted into `sys.path`; this was diagnostic-only and occurred before any scored run.

### Decisions

Validation accepts tuple/list topology inputs but uses `type(count) is int`, rejecting Boolean values that Python otherwise treats as integers. The topology log avoids a conventional WRN depth label because nonuniform stage counts do not fit the usual `6n+4` naming formula.

## Experimental Adjustments

None.

## Run Log

### Run 1

Metadata:
- **Job ID**: local exec session 22094; launcher PID 1160374
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.3-gpt-5-6-sol/run.log`
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-07-24 16:14 UTC
- **Ended**: 2026-07-24 16:20 UTC

Description:
- One fixed-seed FP32 H20 run will test stage depths `[2,2,3]` at accepted widths `[32,64,128]`, with all optimization and regularization choices unchanged. The extra 8x8 residual block spends a similar static budget to EXP-010 on nonlinear depth rather than width. Measured retention projects 131.64 passes, above the 120 interpretation gate. Success requires at least 94.17%; no alternative topology, hyperparameter, precision, or seed fallback is permitted.

Observations:
- Static and semantic checks passed: `train.py`-only diff, byte-identical `prepare.py`, fail-closed evaluator, malformed-depth rejection, complete block/stride/projection topology, exact counts 691,674/987,098, `[256,10]` FP32 output, and finite production updates. (source: local preflight stdout)
- Matched preflight passed. Mixup windows were accepted 10.869903/10.761464/10.833464 and candidate 11.556629/11.660586/11.689949 ms; hard windows were accepted 10.445097/10.509247/10.455168 and candidate 11.300911/11.236514/11.310342 ms. CV ratios were 0.002690-0.004916; weighted aggregates 10.701060/11.534700 ms gave retention 0.927728 and 131.644560 projected passes; peak allocation was 1097.8 MiB. (source: local preflight stdout)
- Scored startup is healthy on `Device: cuda` with exact widths `[32,64,128]`, blocks `[2,2,3]`, and 987,098 parameters. (source: `run.log` startup)
- Mixup disabled exactly once at epoch 86, step 16,642, 195.0 counted seconds (65.0%) with LR 0.0612; hard-label throughput was about 22.8-23.0k images/s and loss remained finite. (source: `run.log` transition and late progress)
- The sole scored process exited 0 after 300.0 counted / 338.5 total seconds. Accuracy reached its best and final value, 94.15%, at terminal epoch 134 with loss 0.2782. (source: `run.log` terminal evaluation and summary)

Key Metrics:

- best_test_acc: 94.15% at epoch 134, delta +0.08 points versus 94.07% but 0.02 below required 94.17% (source: `run.log` line 60 and summary line 62)
- final_test_acc: 94.15%; final_test_loss: 0.2782; best/final gap: 0.00 points (source: `run.log` lines 60, 63-64)
- exposure: 25,961 steps = 132.92032 passes across 134 epochs, above the 120 interpretation gate and 93.67% of accepted 141.9 passes (source: `run.log` lines 69-70)
- total_seconds: 338.5; training_seconds: 300.0; startup_seconds: 1.1; peak_vram_mb: 1096.3; num_params: 987,098; evaluations: 27 unique epochs (source: `run.log` lines 65-71 and cadence audit)

## Verification Results

### Conditions Checked

- **Run completion and protocol**: PASS. Exit 0; one H20; exact `[32,64,128]` / `[2,2,3]` / 987,098 parameters; 300.0 counted and 338.5 total seconds; 132.92 passes; finite loss; 27 unique accepted-cadence evaluations; one correct transition; complete summary; and `train.py`-only diff. (source: preflight, `run.log`, and final diff audit)
- **Primary metric improvement**: FAIL. Best 94.15% is only +0.08 over baseline and below required 94.17%. Verification stopped; the valid near miss is not rerun. (source: `run.log` summary)

### Informational Metrics

Skipped by protocol after the primary margin failure; descriptive metrics are retained under Run 1.

## Errors & Dead Ends

### 2026-07-24 - Temporary preflight import path
- Error: `ModuleNotFoundError: No module named 'prepare'`
- Root cause: running a script stored in `/tmp` put `/tmp`, rather than the project root, first on Python's module search path.
- Source: first local preflight stdout before any model construction or scored run
- Do NOT retry: invoke a temporary preflight without explicitly inserting the project working directory into `sys.path`.

## Human Notes

> Autopilot run; no execution-phase intervention requested.
