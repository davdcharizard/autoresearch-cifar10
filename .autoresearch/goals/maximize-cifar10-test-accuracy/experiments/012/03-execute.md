# EXP-012: Exact 8x8 Bottleneck Residual Refinement

## Execution

Overall Status & Info:
- **Created**: 2026-07-24
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-test-accuracy-012
- **Commit**: (pending - committed on loop success)
- **PR**: N/A - fully local/offline execution
- **Outcome**: failed - valid accuracy regression

## Implementation Notes

### Summary

Added one exact post-stage-3 pre-activation `128->64->64->128` identity bottleneck with accepted initialization. The accepted WRN initializes first; refinement construction and initialization run inside a restoring CPU RNG fork, preserving every accepted tensor and the post-construction RNG state. All training and evaluation logic is unchanged.

### Surprises & Discoveries

The adversarial plan review correctly found that late module registration alone would not preserve RNG because constructors consume random draws. Two-phase initialization fixed both weights and subsequent data RNG. The matched preflight retained 96.04% throughput, better than its 92% gate.

### Decisions

The refinement is optional only for matched preflight construction; production always uses fixed width 64. It is registered after accepted initialization but called before final BN. No endpoint zeroing or alternate ratio/placement is available.

## Experimental Adjustments

None.

## Run Log

### Run 1

Metadata:
- **Job ID**: local exec session 96547; launcher PID 1163657
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.3-gpt-5-6-sol/run.log`
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-07-24 16:46 UTC
- **Ended**: 2026-07-24 16:52 UTC

Description:
- One fixed-seed FP32 H20 run will test the exact rank-64 bottleneck with all accepted training choices unchanged. It adds 53,760 parameters and 3,407,872 MACs/image, using about 18% of EXP-011's full-block increment. Preflight projects 136.28 passes. Success requires 94.17%; no ratio, initialization, placement, optimizer, or seed fallback is permitted.

Observations:
- Static/semantic checks passed: fail-closed evaluator; exact 691,674/745,434 counts; complete topology, hook order, identity shortcut, MACs, finite update; byte-identical accepted state; and identical post-construction CPU RNG. (source: `preflight.py` stdout)
- Timing passed. Mixup accepted 10.821545/10.745329/10.745937 vs candidate 11.293030/11.176870/11.180744 ms; hard accepted 10.412158/10.409898/10.401312 vs candidate 10.854112/10.847595/10.952429 ms. CV ratios were 0.000449-0.004802; aggregates 10.628324/11.066423 ms gave 0.960412 retention and 136.282444 projected passes; peak allocation 1096.9 MiB. (source: `preflight.py` stdout)
- Scored startup is healthy on CUDA with exact `128->64->128` bottleneck and 745,434 parameters. (source: `run.log` startup)
- Mixup disabled exactly once at epoch 87, step 16,919, 195.0 seconds with LR 0.0612; loss remained finite. The process exited 0 after 300.0 counted / 339.8 total seconds. (source: `run.log` transition and summary)
- Best and final accuracy were 93.74% at terminal epoch 136 with loss 0.2873. (source: `run.log` terminal evaluation and summary)

Key Metrics:

- best_test_acc: 93.74%, delta -0.33 points from 94.07% baseline (source: `run.log` summary)
- final_test_acc/loss: 93.74% / 0.2873; best/final gap 0.00 (source: `run.log` summary)
- exposure: 26,462 steps = 135.48544 passes, above 130.5 projection gate and 95.48% of accepted 141.9 (source: `run.log` summary)
- total/training/startup: 339.8/300.0/1.1 seconds; peak VRAM 1094.4 MiB; 136 epochs; 745,434 parameters; 28 evaluations (source: `run.log` summary/cadence audit)

## Verification Results

### Conditions Checked

- **Run completion and protocol**: PASS. Exit 0, one H20, exact topology/count, 300.0 counted / 339.8 total seconds, 135.49 passes, finite loss, one transition, 28 unique accepted-cadence evaluations, and `train.py`-only source diff. (source: preflight, `run.log`, diff audit)
- **Primary metric improvement**: FAIL. 93.74% is 0.33 below baseline and 0.43 below required 94.17%; no rerun. (source: `run.log` summary)

### Informational Metrics

Skipped after primary condition failure; descriptive values are retained above.

## Errors & Dead Ends

### 2026-07-24 - Nested preflight import path
- Error: `ModuleNotFoundError: No module named 'prepare'`
- Root cause: Python placed the nested artifact directory rather than the project root on `sys.path`.
- Source: first preflight invocation before model construction
- Do NOT retry: run a nested diagnostic without inserting the project working directory into `sys.path`.

## Human Notes

> Autopilot; no execution-phase intervention.
