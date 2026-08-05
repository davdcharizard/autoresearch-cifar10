# EXP-016: Fixed-MAC Stage-Depth Redistribution

## Execution

Overall Status & Info:
- **Created**: 2026-07-26
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-test-accuracy-016
- **Commit**: (pending - committed on success)
- **PR**: N/A - local-only run
- **Outcome**: failed - valid run scored below the accuracy threshold

## Implementation Notes

### Summary

The production model first builds and initializes the accepted `[2,2,2]` graph, then removes `layer1[1]` and attaches one `128->128` block as `layer3[2]`. The added block is constructed and explicitly Kaiming-initialized inside a restored CPU RNG fork using preregistered seed 16016. All training behavior outside the final topology and descriptive startup log remains accepted.

### Surprises & Discoveries

Equal convolutional MACs did not imply equal H20 latency. Moving one block from 32x32 to 8x8 reduced weighted step time by about 17%, likely through lower activation traffic and more efficient dense-channel kernels.

### Decisions

The new block receives both constructor defaults and the exact accepted `_weights_init`, matching how existing blocks end after whole-model initialization. No adjacent allocation or seed was tested.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local exec session 53863
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.3-gpt-5-6-sol/run.log`
- **WandB**: N/A
- **Status**: completed (research no-improvement)
- **Started**: 2026-07-26 16:46:20 UTC
- **Ended**: 2026-07-26 16:52:15 UTC

Description:
- One fixed-seed H20 run tests whether redistributing six WRN blocks from `[2,2,2]` to `[1,2,3]` retains the positive dense stage-3 signal while recovering exposure at equal static MACs. Success requires a valid score of at least 94.17%; no topology or seed fallback is allowed.

Observations:
- Semantic preflight passed: depths `[1,2,3]`, 968,538 parameters, 101,106,944 MACs, seed-16016 oracle equality, common-state/RNG equality, correct shapes, and live new-block gradients. (source: semantic preflight stdout)
- Throughput preflight passed: weighted accepted/candidate times 10.671999/8.853656 ms, retention 1.205378, projected 171.043095 passes; every regime/path CV was below 0.0053. (source: throughput preflight stdout)
- Final scope/syntax audit passed and the scored log immediately reported CUDA, depths `[1,2,3]`, and 968,538 parameters. (source: pre-score commands and `run.log` startup)
- Mixup disabled exactly once at epoch 111, step 21,454, and 195.0 counted seconds; 35 evaluation epochs were unique and no error signature appeared. (source: `run.log` L6-L87)
- The run exited 0 after 300.0 counted and 346.2 total seconds. (source: local command exit and `run.log` L81-L82)

Key Metrics:
- best/final test accuracy: 93.82% at epoch 172, 0.25 points below the 94.07 baseline. (source: `run.log` L76, L78-L79)
- final test loss: 0.2778. (source: `run.log` L80)
- exposure: 33,535 steps = 171.6992 dataset-equivalent passes across 172 epochs. (source: `run.log` L85-L86)
- resources: 968.2 MiB peak VRAM, 968,538 parameters. (source: `run.log` L84, L87)

## Verification Results

### Conditions Checked

- **Completion and integrity**: PASS - exit 0, one H20, exact topology/count, 300.0 counted seconds, 346.2 total seconds, one transition, 35 unique evaluation epochs, and no runtime errors. (source: `run.log` L6-L87 and command exit)
- **Primary metric >=94.17%**: FAIL - best accuracy 93.82%, below both 94.07 baseline and 94.17 threshold. Verification stopped on this necessary-condition failure. (source: `run.log` L78 and results index)
- **Remaining conditions**: skipped after metric failure; scope, semantics, and production diff had already passed mandatory pre-score gates.

### Informational Metrics

- Skipped under the verification guard; values are preserved in Run 1 Key Metrics for analysis.

## Errors & Dead Ends

## Human Notes

> Autopilot local-only execution; no intervention requested.
