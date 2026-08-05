# EXP-014: Safe Zero-Initialized Residual Endpoints

## Execution

Overall Status & Info:
- **Created**: 2026-07-26
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-test-accuracy-014
- **Commit**: (pending - committed on success)
- **PR**: N/A - fully local/offline
- **Outcome**: failed - valid accuracy regression

## Implementation Notes

### Summary
Accepted initialization remains unchanged, then exactly six `PreActBlock.conv2.weight` tensors are zeroed without consuming RNG. The graph, parameter count, optimizer, data, schedule, mixup, and evaluator are unchanged.

### Surprises & Discoveries
The architecture-specific dead-gradient warning was confirmed: zero pre-ReLU `bn2` scale receives zero gradient, while zero endpoint convolutions all receive positive-norm gradients immediately and open upstream branches on the second backward.

### Decisions
The constructor switch is strict Boolean and exists only for matched preflight. Production enables all six endpoints; no stage-selective or nonzero scale path exists.

## Experimental Adjustments
None.

## Run Log

### Run 1
Metadata:
- **Job ID**: local exec session 19914; launcher PID 1273681
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.3-gpt-5-6-sol/run.log`
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-07-26 15:36 UTC
- **Ended**: 2026-07-26 15:42 UTC

Description:
- One fixed-seed H20 run tests exact all-six residual endpoint zeroing with every accepted training choice unchanged. Preflight projects 142.03 passes. Success requires at least 94.17%; there is no initialization, LR, architecture, or seed fallback.

Observations:
- Fail-closed semantic checks passed exact 691,674 count, RNG/non-endpoint equality, six zero paths, correct identity/projection shortcuts, first-step endpoint opening, second-step upstream opening, and rejected BN-zero behavior. (source: preflight stdout)
- Timing windows had CV ratios 0.00054-0.00081; aggregates accepted/candidate 10.696490/10.686570 ms yielded retention 1.000928 and 142.031726 projected passes. (source: preflight stdout)
- Scored startup is healthy on CUDA with exactly six zero-initialized residual blocks and 691,674 parameters. Early loss is finite at accepted throughput. (source: `run.log` startup)
- Mixup disabled exactly once at epoch 92, step 17,860, 195.0 seconds with LR 0.0612. The process exited 0 after 300.0 counted / 341.4 total seconds. (source: `run.log` transition/summary)
- Best and final accuracy were 93.88% at terminal epoch 144 with loss 0.2660. (source: `run.log` terminal/summary)

Key Metrics:
- best/final: 93.88% / 93.88%; delta -0.19 from baseline; final loss 0.2660.
- exposure: 27,892 steps = 142.80704 passes; 144 epochs; 29 evaluations; 1,094.0 MiB peak; 691,674 parameters.

## Verification Results
### Conditions Checked
- **Protocol**: PASS - one H20, exact six endpoints/count, exit 0, 300.0/341.4 seconds, 142.81 passes, one transition, finite trajectory, accepted cadence.
- **Metric**: FAIL - 93.88% is below baseline and required 94.17%; no rerun.
### Informational Metrics

## Errors & Dead Ends
None.

## Human Notes
> Autopilot; no intervention.
